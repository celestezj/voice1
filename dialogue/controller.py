# -*- coding: utf-8 -*-
"""对话控制器：ASR(voice1) → LLM → TTS(voice0) 的单进程编排核心。

线程模型（全链路非阻塞，主线程只管采麦克风）：
- `feed_asr_sentence()` 在 ASR worker 线程被回调，只做快操作：累加本轮话、gen+=1、
  （LLM 在途则）`tts.interrupt()`、起新的 LLM 读流线程。绝不碰网络/长任务。
- `_llm_loop()` 专用线程阻塞读 SSE；每 token 先查代际（gen != 当前 → 弃流返回，
  生成器 close 关连接），按标点切句 `tts.submit()`（非阻塞入队）。
- TTS(voice0) 内部 queue 模式串行合成播放。

打断语义（用户已拍板）：
- 新 ASR 句若 LLM 在途 → 取消当前生成并重发「本轮累计」；`interrupt()` 只在显式打断
  LLM 吐词时联动调用（切掉被作废回复的音频与队列）。
- **post-commit barge**（`post_commit_window`，主程序默认 1.5s）：LLM 已生成完、但音频
  还没开播（本轮首句提交至今 < 窗口）时来新句 → 撤下刚 commit 的 (残句→答复)、残句+新句
  合并连同历史重发。**零固定延迟**——只在用户真补了句尾巴时才重答。
- LLM 已生成完且音频已开播（过窗口）时来新句 → 新轮，不打断语音。
- 句末合并窗口（`merge_window`，默认关）：断句后等窗口内补句才发 LLM（每轮固定延迟，
  主程序默认 0 不用它，post-commit barge 是其零延迟替代）。
"""
import threading
import time


class DialogueController:
    # LLM 切句边界 + 超长无标点硬切参数
    _BOUNDARY = "。！？…；\n"     # 句末边界（送 TTS 的切点）
    _COMMA = "，、："
    _HARD_MAX = 40                # 无标点累积超此长度 → 兜底硬切（保首包延迟）
    _COMMA_WINDOW = 12            # 硬切时回找逗号的最大回看长度

    def __init__(self, llm, tts, *, system_prompt=None, max_history_messages=None,
                 reply_hold=0.0, merge_window=0.0, post_commit_window=0.0,
                 max_context_tokens=40000, recent_keep=6, headroom=4000):
        self._llm = llm
        self._tts = tts
        self._system = system_prompt or (
            "你是语音助手。回答要口语化、简洁、适合语音播报：不要用 markdown、列表、"
            "符号或缩写；一次说 1-3 句话即可，必要时追问一句；不知道就直说。")
        self._max_history = max_history_messages  # 硬安全上限（None=由 token 预算管理）
        self._reply_hold = float(reply_hold)      # 首句 hold-off 秒（0=关）
        self._compress_threshold = int(max_context_tokens)   # 上下文压缩阈值（prompt tokens）
        self._recent_keep = int(recent_keep)      # 压缩后原样保留的最近消息条数
        self._headroom = int(headroom)            # 阈值预留余量（下一轮问题 + 安全）
        self._history = []           # 已提交的 user/assistant 轮（不含 system/摘要）
        self._summary = ""           # 已压缩的旧历史（构建请求时拼进 system）
        self._prompt_tokens = None   # 最近一次响应的精确上下文 token 数（usage.prompt_tokens）
        self._compress_running = False
        self._user_turn = ""         # 本轮累计（被打断时重发用）
        self._assistant_buf = ""     # 当前生成缓冲（未切句/未提交）
        self._assistant_full = ""    # 当前生成完整文本（_emit_sentences 只剥 buf 不动它，commit 用）
        self._gen = 0                # LLM 代际：每新句 +1，旧线程据此弃流
        self._stream_thread = None   # 在途 LLM 读流线程
        self._merge_window = float(merge_window)   # 句末合并窗口秒（0=关，立即发）
        self._merge_deadline = None                # 当前合并窗口截止（monotonic）
        self._merge_waiter = None                  # 合并窗口守护线程
        self._post_commit_window = float(post_commit_window)  # post-commit barge 窗口秒（0=关）
        self._turn_first_submit_ts = None          # 本轮首句提交时刻（post-commit 窗口锚点）
        self._tts_job = None         # 最近提交的 TTS Job（voice0 返回值，含 .done/.wait）
        self._tts_busy = False       # TTS 是否在播/待播（echo 门控依据）
        self._lock = threading.RLock()     # RLock：_llm_loop finally 在锁内 _submit_tts 会重入
        self._closed = False
        self._on_user = None
        self._on_ai_delta = None
        self._on_ai_sentence = None
        self._on_ai_done = None
        self._on_llm_start = None    # LLM 请求已发起（等待首 token，供控制台状态行）
        self._on_llm_error = None    # LLM 流抛异常（供控制台报错行）
        self._on_merge_rollback = None   # post-commit barge 撤答复（供控制台提示）

    # ---------------- 回调注册（供主程序/控制台接）----------------
    def register_callbacks(self, on_user=None, on_ai_delta=None,
                           on_ai_sentence=None, on_ai_done=None,
                           on_llm_start=None, on_llm_error=None,
                           on_merge_rollback=None):
        self._on_user = on_user
        self._on_ai_delta = on_ai_delta
        self._on_ai_sentence = on_ai_sentence
        self._on_ai_done = on_ai_done
        self._on_llm_start = on_llm_start
        self._on_llm_error = on_llm_error
        self._on_merge_rollback = on_merge_rollback   # post-commit barge 撤答复（供控制台提示）

    @property
    def history(self):
        with self._lock:
            return list(self._history)

    @property
    def tts_busy(self):
        """TTS 是否在播/待播（回声门控依据）。GIL 原子读，mic 回调每块直读无锁。"""
        return self._tts_busy

    def _build_messages_locked(self):
        """构建本次 LLM 请求的消息列表（调用方持锁）：system(+摘要)+历史+最新用户问题。"""
        system = self._system
        if self._summary:
            system = system + "\n\n【此前对话摘要】\n" + self._summary
        return ([{"role": "system", "content": system}]
                + list(self._history)
                + [{"role": "user", "content": self._user_turn}])

    # ---------------- "停下"硬停（ASR on_interrupt 回调，mic 线程）----------------
    def hard_stop(self):
        """用户说"停下"→ 立即终止 LLM 与 TTS 输出。

        - 在途 LLM 流弃（gen+1，作废回复**不 commit**）；
        - **被打断的问题保留**：commit 进历史（问题不丢）；
        - "停下"本身绝不进历史/LLM 输入——KWS 旁路吞掉触发块，根本不触发 on_sentence。
        """
        with self._lock:
            if self._closed:
                return
            self._gen += 1
            if self._user_turn:
                self._history.append({"role": "user", "content": self._user_turn})
                self._user_turn = ""
            self._assistant_buf = ""
            self._assistant_full = ""
            self._stream_thread = None
            self._merge_deadline = None    # 有挂起的合并窗口 → 作废（"停下"不续发）
            self._tts.interrupt()          # 立即切音频 + 清队列（快操作）
        self._maybe_compress()             # 历史变了，检查是否需要压缩

    # ---------------- TTS 提交 + 忙碌跟踪（voice0 Job.done，不改 voice0）----------------
    def _submit_tts(self, sentence):
        """提交给 TTS 并登记忙碌跟踪（首个任务起守护 watcher，排空后 _tts_busy 回落）。"""
        job = self._tts.submit(sentence)
        with self._lock:
            if self._turn_first_submit_ts is None:
                self._turn_first_submit_ts = time.monotonic()  # 本轮首句提交时刻（post-commit 锚点）
            first = self._tts_job is None
            self._tts_job = job
            self._tts_busy = True
        if first:
            threading.Thread(target=self._tts_watch, name="dialogue-tts-watch",
                             daemon=True).start()

    def _tts_watch(self):
        """守护：等最后一个 Job 播完/被取消（队列排空）→ _tts_busy 回落。"""
        while True:
            with self._lock:
                job = self._tts_job
            if job is None:
                time.sleep(0.02)
                continue
            job.wait()                     # 阻塞到该 job 播完或被打断（永不悬挂）
            with self._lock:
                if self._closed:
                    return
                if self._tts_job is job:
                    self._tts_busy = False
                    self._tts_job = None
                    return                 # 排空
                # 已有更新的 job → 继续盯它

    # ---------------- 历史压缩（事件驱动后台线程，不阻塞对话）----------------
    def _maybe_compress(self):
        """提交后检查上下文用量；超阈值且空闲 → 派一次性后台线程压缩旧历史。

        事件驱动（不常驻监控）：条件=「无在途 LLM 流 + 无压缩在跑 + 用量超阈值 +
        历史够多」。压缩是网络调用，放后台线程做，下一轮对话照常走当前快照。
        """
        with self._lock:
            if (self._closed or self._compress_running
                    or (self._stream_thread and self._stream_thread.is_alive())):
                return
            tokens = self._prompt_tokens
            if tokens is None:             # 无 usage 兜底：按字符估算
                tokens = self._llm.estimate_tokens(self._build_messages_locked())
            if tokens < self._compress_threshold - self._headroom:
                return
            if len(self._history) <= self._recent_keep:
                return
            self._compress_running = True
        threading.Thread(target=self._compress_task, name="dialogue-compress",
                         daemon=True).start()

    def _compress_task(self):
        with self._lock:
            history = list(self._history)
        recent = history[-self._recent_keep:]
        old = history[:-self._recent_keep]
        try:
            summary = self._llm.compress(old)
        except Exception as e:
            print("[dialogue] 历史压缩失败（保留原历史，不阻塞对话）: %s" % e, flush=True)
            return
        finally:
            with self._lock:
                self._compress_running = False
        with self._lock:
            if self._closed or not summary:
                return
            if self._history != history:   # 快照期间有新 commit → 放弃本次，下轮再压
                return
            self._summary = summary
            self._history = recent
            self._prompt_tokens = None     # 压缩后失效，等下次响应重定

    # ---------------- 入口：ASR on_sentence（ASR worker 线程）----------------
    def feed_asr_sentence(self, result):
        """新定稿句。快操作，不阻塞识别。

        三层防拆句：
        - 在途 barge：LLM 还在流时来新句 → gen+1 弃流 + `tts.interrupt()`，累计重发。
        - **post-commit barge**（`_post_commit_window`，主程序默认 1.5s）：LLM 已生成完、
          但音频还没开播（本轮首句提交至今 < 窗口）时来新句 → `_rollback_last_turn_locked`
          撤下刚 commit 的 (残句→答复)，残句+新句合并连同历史重发。零固定延迟。
        - 句末合并窗口（`_merge_window`，默认关）：断句后等窗口内补句才发 LLM（固定延迟，
          不用）。窗口=0 → 立即发。
        """
        text = (getattr(result, "text", None) or "").strip()
        if not text:
            return
        with self._lock:
            if self._closed:
                return
            in_flight = self._stream_thread is not None and self._stream_thread.is_alive()
            post_commit = (not in_flight and self._post_commit_window > 0
                           and self._tts_busy
                           and self._turn_first_submit_ts is not None
                           and time.monotonic() - self._turn_first_submit_ts < self._post_commit_window)
            if post_commit:
                self._rollback_last_turn_locked()   # 撤下 (残句→答复)，残句回到本轮累计
            self._user_turn = (self._user_turn + text) if self._user_turn else text
            self._gen += 1
            self._assistant_buf = ""          # 旧流作废：清缓冲与完整文本
            self._assistant_full = ""
            if in_flight or post_commit:
                self._tts.interrupt()         # 在途吐词 / 已答未开播 → 切掉作废音频
            self._stream_thread = None        # 在途流作废（gen 已变，旧线程自行退出）
            if self._merge_window > 0:
                self._merge_deadline = time.monotonic() + self._merge_window
                launch = False
            else:
                self._merge_deadline = None
                launch = True
        if post_commit and self._on_merge_rollback:
            self._on_merge_rollback()
        if self._on_user:
            self._on_user(result)
        if launch:
            self._launch_llm()
        else:
            self._start_merge_waiter()

    def _rollback_last_turn_locked(self):
        """撤下最近一轮已 commit 的 (user→assistant) 对，残句放回本轮累计。

        仅用于 post-commit barge：AI 已生成完但音频还没开播，用户补了句尾巴——把对残句
        的答复从历史撤掉，残句与新句合并后连同历史一起重发。调用方持锁。
        """
        if not self._history:
            return
        if self._history[-1]["role"] == "assistant":
            self._history.pop()
        if self._history and self._history[-1]["role"] == "user":
            frag = self._history.pop()["content"]
            self._user_turn = frag

    def _launch_llm(self):
        """把本轮累计发给 LLM（合并窗口过期 / 窗口=0 立即）。调用方不持锁。"""
        with self._lock:
            if self._closed or not self._user_turn:
                return
            if self._stream_thread is not None and self._stream_thread.is_alive():
                return                 # 防御：不应有在途流（feed 已置 None）
            self._merge_deadline = None
            self._turn_first_submit_ts = None  # 新一轮：首句提交时刻锚点重置
            gen = self._gen
            messages = self._build_messages_locked()   # system(+摘要)+历史+最新用户问题
        t = threading.Thread(target=self._llm_loop, args=(gen, messages),
                             name="dialogue-llm", daemon=True)
        with self._lock:
            self._stream_thread = t
        t.start()

    def _start_merge_waiter(self):
        with self._lock:
            if self._merge_waiter is not None and self._merge_waiter.is_alive():
                return                 # 已有守护线程盯着，新句会重置 deadline
            self._merge_waiter = threading.Thread(target=self._merge_wait,
                                                  name="dialogue-merge", daemon=True)
            self._merge_waiter.start()

    def _merge_wait(self):
        """守护：等合并窗口过期 → 发 LLM。新句重置 deadline，分片睡及时响应。"""
        while True:
            with self._lock:
                if self._closed:
                    return
                deadline = self._merge_deadline
            if deadline is None:
                return
            now = time.monotonic()
            if now >= deadline:
                self._launch_llm()
                return
            time.sleep(0.1)

    # ---------------- LLM 读流线程 ----------------
    def _llm_loop(self, gen, messages):
        try:
            if self._on_llm_start:
                self._on_llm_start()
            for delta in self._llm.stream_chat(messages):
                if gen != self._gen:          # 已被更新请求取代 → 弃流（生成器 close 关连接）
                    return
                with self._lock:
                    if gen != self._gen:
                        return
                    self._assistant_buf += delta
                    self._assistant_full += delta
                    buf = self._assistant_buf
                if self._on_ai_delta:
                    self._on_ai_delta(delta, buf)
                self._emit_sentences(gen)
            # 流正常结束 → 记录本次上下文的精确 token 用量（压缩触发依据）
            usage = getattr(self._llm, "last_usage", None)
            if isinstance(usage, dict) and usage.get("prompt_tokens"):
                self._prompt_tokens = usage["prompt_tokens"]
            else:
                self._prompt_tokens = self._llm.estimate_tokens(messages)
        except Exception as e:
            if gen == self._gen:
                print("[dialogue] LLM 出错: %s" % e, flush=True)
                if self._on_llm_error:
                    self._on_llm_error(e)
        finally:
            if gen != self._gen:
                return
            with self._lock:
                if gen != self._gen:
                    return
                full = self._assistant_full        # 完整回复（用于 commit/回调）
                tail = self._assistant_buf.strip()  # 未切句的残句也要播出来
                if tail:
                    self._submit_tts(tail)
                self._assistant_buf = ""
                self._assistant_full = ""
                self._commit_locked(full)
                self._stream_thread = None
            if self._on_ai_sentence and tail:
                self._on_ai_sentence(tail)
            if self._on_ai_done and full.strip():
                self._on_ai_done(full)
            self._maybe_compress()

    def _commit_locked(self, full_text):
        """完整回复才进历史（被打断的回复 gen 不对，根本到不了这里）。调用方持锁。"""
        if self._user_turn:
            self._history.append({"role": "user", "content": self._user_turn})
        t = full_text.strip()
        if t:
            self._history.append({"role": "assistant", "content": t})
        self._user_turn = ""
        # 历史长度由 token 预算（_maybe_compress）管理；_max_history 仅作硬安全上限
        if self._max_history and len(self._history) > self._max_history:
            self._history = self._history[len(self._history) - self._max_history:]

    # ---------------- 切句 → TTS ----------------
    def _emit_sentences(self, gen):
        """把缓冲里已到边界的句子切出来 submit 给 TTS（queue 串行播放）。

        提交放在持锁区间内：与打断路径的 `tts.interrupt()` 串行，杜绝"打断后又
        submit 出作废回复残留句子"的竞态。
        """
        while True:
            with self._lock:
                if gen != self._gen:
                    return
                cut = self._find_cut(self._assistant_buf)
                if cut is None:
                    return
                sentence = self._assistant_buf[:cut].strip()
                self._assistant_buf = self._assistant_buf[cut:]
                first = self._tts_job is None     # 本回合首句（尚无 TTS 任务在册）
            if not sentence:
                continue                          # 纯边界字符（如"。。"）丢弃后继续找
            if first and self._reply_hold > 0:
                time.sleep(self._reply_hold)      # 锁外：给用户续句打断的机会
                with self._lock:
                    if gen != self._gen:
                        return                    # hold 期间被 barge → 弃句（绝不播）
            self._submit_tts(sentence)
            if self._on_ai_sentence:
                self._on_ai_sentence(sentence)

    def _find_cut(self, buf):
        """返回首个可提交切点下标；无可提交（无边界且未超长）返回 None。"""
        n = len(buf)
        if n == 0:
            return None
        # 1) 最后一个句末边界（.。！？…；\n）
        last = -1
        for ch in self._BOUNDARY:
            i = buf.rfind(ch)
            if i > last:
                last = i
        if last >= 0 and len(buf[:last + 1].strip()) >= 2:
            return last + 1
        # 2) 超长无标点 → 硬切（优先回找逗号，否则硬切 _HARD_MAX）
        if n > self._HARD_MAX:
            start = max(0, n - self._COMMA_WINDOW)
            ci = -1
            for ch in self._COMMA:
                i = buf[start:].rfind(ch)
                if i >= 0 and i + start > ci:
                    ci = i + start
            if ci >= start:
                return ci + 1
            return self._HARD_MAX
        return None

    # ---------------- 生命周期 ----------------
    def close(self):
        with self._lock:
            self._closed = True
            self._gen += 1                    # 让在途线程弃流（不杀线程）
            self._merge_deadline = None       # 结束挂起的合并窗口（_merge_wait 见 _closed 退出）
