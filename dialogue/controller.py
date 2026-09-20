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
import re
import threading
import time

from .debug_log import dbg   # 临时：VOICE1_DEBUG_TTS=1 才落盘，定位后删
from .toolparse import ToolXmlParser   # LLM 模式工具：XML 流式解析（--tools，docs/llm-tools.md）


class DialogueController:
    # LLM 切句边界 + 超长无标点硬切参数
    _BOUNDARY = "。！？…；\n"     # 句末边界（送 TTS 的切点）
    _SOFT_CUT = " 　，、："    # 兜底硬切的可落点：空格分句缝 + 逗号类（非句末边界）
    # 句末语气词（白话口语天然句尾）：流式缓冲以它结尾且无标点 → 视为完整句先送出，
    # 否则"我再确认一下大后天的天气哈"这类无句号的过渡句卡在缓冲，工具调用期间出声前
    # 干等（2026-09-13 实测）。切错最多多一个停顿，无碍正确性。
    _PARTICLES = "哈吧呢嘛啊哦呀啦哟咯嘞呗呵"
    _HARD_MAX = 40                # 无标点累积超此长度 → 兜底硬切（保首包延迟）
    _SOFT_WINDOW = 20             # 硬切时在末 _SOFT_WINDOW 字符里回找软分句缝（绝不撕词）
    # 结论与流式文本的"未播残尾"短于此 → 视为已全量流式（branch a 不打断）。修金价冻结
    # （2026-09-14 实测）：ResultMessage 到达时流式缓冲还压着句末"。"没切出来，skip 只差 1
    # 字没盖满 → 旧逻辑落进 branch c，`tts.interrupt()` 把**已入队未开播**的整条结论音频
    # 全取消，重播又只有"。"（_find_cut 吐不出 <2 字句、_submit_tts 丢纯标点）→ 彻底静音。
    # 残尾 ≤ 此值 → 队列里就是结论本体，让它自然播完；残尾是纯标点/尾词，值不得打断。
    _TRIVIAL_TAIL = 6
    # agent 流式缓冲滞留超此秒数（无标点无语气词、agent 静默——工具调用期）→ 整体切句送出。
    # 过渡句"好，讲个新笑话给你"以"你"结尾无切点，会卡到最终结论才切（实测 3.7s 干等、
    # 出声前工具都跑完了）；静默超阈值说明 agent 在忙别的（工具/思考），当前缓冲大概率是
    # 完整过渡句 → 先出声（2026-09-14 实测金价过渡句后跟 \n 能立即切、笑话这句却卡死）。
    _AGENT_IDLE_FLUSH_MS = 1.5
    # 引号（中文/ASCII）：送 TTS 前剥掉——语音不念引号，`。”`/`？"`/`。" "` 这类"句末标点+
    # 引号"连在一起会让 TTS 前端对全角引号处理不稳、合成怪尾音（实测苏联笑话 2026-09-13）。
    _QUOTE_RE = re.compile(r"[“”‘’\"']")
    # 括号类（书名号/方括号/圆括号等）：语音不念，`？》】`/`《…` 这类"标点+括号"连标点同样
    # 出怪声（实测笑话标题《世界上哪个国家最大？》末尾 `？》】` 2026-09-13）。须在 _MOOD_SUB
    # 之后剥（先剥【心态：xxx】整体，否则心态文字会从括号里漏出来被念）。
    _BRACKET_RE = re.compile(r"[《》〈〉「」『』【】（）〔〕]")
    # 右引号/右括号/闭书名号：句末边界（？！。…\n）之后紧随的闭符并入前一句——否则
    # 换行切句会把 `"》》` 这类纯标点残留切成独立 Job，TTS 合成标点怪声（实测苏联笑话标题）。
    _CLOSERS = frozenset("”’\"'》」』】）〕〉")
    # 连续相同的句末/停顿标点 → 只留一个：`……`（鬼故事实测送 TTS 合成怪声）、`。。`、
    # `——` 等重复标点没有朗读意义，TTS 前端念重复标点不稳（2026-09-14 用户实测）。
    _PUNCT_RUN_RE = re.compile(r"([。！？…～、；：，,—])\1+")
    # 兜底剥尖括号段（`<…>`，长度≤64 防吞正文）：LLM 吐未知标签/残留标签时防止被 TTS 念出来。
    # 已知工具标签已被 toolparse 剥掉，这里只兜底未知/半截标签（幂等，同 _PUNCT_RUN_RE 位置）。
    _TAG_RE = re.compile(r"<[^<>]{1,64}>")

    # 心态标记：LLM 回复开头带【心态：xxx】（user_prompt.txt 约定），代表表情、不念出来。
    # 支持【】与 [] 两种括号；_MOOD_RE 取首个心态（on_mood 回调），_MOOD_SUB 只在送 TTS 时
    # 剥掉标记不读；标记保留在正文/历史/存档/控制台（它是模型的真实输出，必须带上）。
    _MOOD_RE  = re.compile(r"[【\[]心态[:：]\s*([^】\]]+)[】\]]")
    _MOOD_SUB = re.compile(r"[【\[]心态[:：][^】\]]*[】\]]")
    # 权限交互标记【询问】（agent 模式人格约定）：正文/存档保留，送 TTS 时剥掉括号不念。
    _ASK_RE   = re.compile(r"[【\[]询问[】\]]")
    _MOODS = {"平和", "开心", "兴奋", "惊喜", "温柔", "关切", "好奇", "期待",
              "无奈", "失望", "沮丧", "难过", "担心", "不满", "生气", "愤怒"}

    def __init__(self, llm, tts, *, system_prompt=None, max_history_messages=None,
                 reply_hold=0.0, merge_window=0.0, post_commit_window=0.0,
                 max_context_tokens=40000, recent_keep=6, headroom=4000,
                 mood_marker=True, agent=None, replay_echo_guard=1.5,
                 tools=None, tools_max_rounds=3, tools_timeout=None):
        # mood_marker=False → 本类的全部心态逻辑跳过（剥标记/解析/默认心态），
        # 行为与本次改动前完全一致；是否让 LLM 吐标记由 user_prompt.txt 里的约定决定。
        #
        # agent（ClaudeAgentClient，可选）：不为 None → **agent 模式**。自实现的历史/
        # 压缩/系统提示词全部旁路，上下文在 claude 会话里；controller 只做
        # "ASR 句 → agent → 最终结论 → TTS"。打断走 agent.abort()（等价 ESC，不 kill 进程）。
        self._llm = llm
        self._agent = agent
        self._tts = tts
        # 心态发射改由 SayTTS 播放链在"本句开播"瞬间执行（2026-09-19）：live2d 开时
        # tts 是 SayTTS 代理（有 mood_supported），controller 随 submit 带句首心态；
        # live2d 关时 tts 是裸 voice0（无此属性）→ 不传（无 live2d 也无处可发）。
        self._mood_supported = bool(getattr(tts, "mood_supported", False))
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
        self._replay_echo_guard = float(replay_echo_guard)  # 结论重播后的回声自屏蔽秒（0=关）
        self._replay_echo_guard_until = 0.0        # 回声自屏蔽截止（monotonic）：期间新 ASR 句
                                                   # 视为"被切音频的回声"丢弃——不打断重播
        self._replay_kws_until = 0.0               # （已停用 2026-09-14：KWS"停下"宽守卫删除，
                                                   #   保留字段仅防旧代码引用崩；见 kws_guard_active）
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
        self._mood_marker = bool(mood_marker)   # 心态标记总开关（False=全部跳过，行为同改动前）
        self._mood = None                # 本回复是否出现过心态标记（None=尚未；finally 判"没带标记"）
        self._mood_pos = 0               # _parse_mood_locked 增量扫描位置（只维护 _mood 状态，
                                         # 发射已挪到 SayTTS 播放链——见 _submit_tts/_leading_mood）
        # LLM 模式工具（--tools，docs/llm-tools.md）：默认关。开时 _llm_loop 走多轮循环——
        # 边流边解析 XML 标签→执行工具→结果回灌续轮（同一 gen/线程，barge-in 整轮作废）。
        self._tools = None               # {name: Tool} 或 None（关）
        self._tool_max_rounds = 3        # 工具续轮上限（防工具无限循环）
        self._tool_parser = None         # ToolXmlParser 实例（tools 关 = None → 单轮原路径）
        self._on_tool = None             # 工具执行完成回调（供控制台诊断行 [工具] …）
        self._tool_results_inflight = [] # 本轮工具结果（commit 时插在 user 与 assistant 之间，
                                         # 保证历史顺序：问题→[工具结果]→答复；硬停作废）
        # agent 模式专属状态（brain=agent 时使用）
        self._agent_evts = {}            # gen → Event（每回合一个，结果/作废唤醒对应收尾线程）
        self._agent_error = None         # 最近一次 agent 回合的错误文本（None=正常）
        self._assistant_display = ""     # agent 流式出字缓冲（仅控制台显示；TTS 仍取最终结论）
        # agent 流式送 TTS（--agent-stream-tts，默认关）：模型边生成边按句播报——
        # 含工具调用前的过渡句/思考段（实测「我把未来七天的天气捋一遍给你哈」只显示不播、
        # 出声前干等工具 7-8s 的体验问题）；最终结论一到立即 interrupt 打断重播。
        # 关 = 只播最终结论（旧行为，零变化）。
        self._agent_stream_tts = False   # 开关（set_agent 传入）
        self.set_tools(tools, max_rounds=tools_max_rounds, timeout=tools_timeout)
        self._agent_tts_buf = ""         # 流式增量待切句缓冲（仅 _agent_stream_tts 时使用）
        self._agent_tts_played = ""      # 流式已 submit 的 TTS 文本累计（_clean_for_tts 后；
                                         # 最终结论重播时按尾部重叠跳过已播前缀，防"阿阳"播两遍）
        self._agent_last_delta_ts = 0.0  # 最近一次流式增量时刻（monotonic）：idle 切句判定
                                         # （agent 工具调用期静默无增量，缓冲滞留超阈值先出声）
        self._pending_mood_announce = "" # 切句剥掉的纯心态标记段（【心态：xxx】）：攒着拼到
                                         # 下一个真实句子的控制台显示上——否则标记只活在预览里、
                                         # 定稿行被覆盖后控制台看不到（LLM/agent 模式均 2026-09-14）

    # ---------------- LLM 模式工具（--tools，docs/llm-tools.md）----------------
    def set_tools(self, tools=None, *, max_rounds=3, timeout=None):
        """启用/关闭 LLM 模式工具调用。tools: {name: Tool}，None/空 = 关（默认）。

        默认关 = 不注入提示、不挂解析器、`_llm_loop` 走单轮原路径——旧行为零变化。
        开启后：system 追加工具文档 → 模型在输出流里写 `<工具名 参数="值"/>` → 解析器
        捕获 → 工具执行（锁外 + 超时守卫）→ 结果回灌续轮（上限 max_rounds）。
        timeout: 非 None 时统一覆盖所有工具的 timeout 秒（`--tools-timeout`）。
        """
        self._tools = dict(tools) if tools else None
        self._tool_max_rounds = max(1, int(max_rounds))
        if timeout:
            for t in (self._tools or {}).values():
                t.timeout = float(timeout)
        self._tool_parser = ToolXmlParser(self._tools.keys()) if self._tools else None

    # ---------------- 回调注册（供主程序/控制台接）----------------
    def register_callbacks(self, on_user=None, on_ai_delta=None,
                           on_ai_sentence=None, on_ai_done=None,
                           on_llm_start=None, on_llm_error=None,
                           on_merge_rollback=None, on_tool=None):
        # 注意：心态发射已不在本类（挪到 SayTTS 播放链，主程序 idle_tts.set_mood_cb 注入）
        self._on_user = on_user
        self._on_ai_delta = on_ai_delta
        self._on_ai_sentence = on_ai_sentence
        self._on_ai_done = on_ai_done
        self._on_llm_start = on_llm_start
        self._on_llm_error = on_llm_error
        self._on_merge_rollback = on_merge_rollback   # post-commit barge 撤答复（供控制台提示）
        self._on_tool = on_tool                       # 工具执行完成（供控制台 [工具] 诊断行）

    def set_agent(self, agent, *, stream_tts=False):
        """绑定 agent 客户端并接管其结果回调（agent 模式大脑）。须在 agent.start() 前调用。

        stream_tts=True：agent 流式增量（含工具调用前的过渡句/思考）也按句送 TTS，
        最终结论到达立即打断重播；False=只播最终结论（默认，旧行为）。"""
        self._agent = agent
        self._agent_stream_tts = bool(stream_tts)
        if agent is not None:
            agent.set_callbacks(on_result=self._on_agent_result,
                                on_partial=self._on_agent_partial)

    @property
    def history(self):
        with self._lock:
            return list(self._history)

    def snapshot(self):
        """当前对话状态完整快照（供主程序本地持久化）。

        锁内只做浅拷贝（list/str，微秒级），真正的磁盘写入由调用方在锁外做——
        因此周期性存档**不阻塞** LLM 读流线程。字段即 LLM 可见的完整输入：
        system（系统提示词，永不压缩）+ summary（压缩摘要，拼在 system 后）+
        history（已 commit 轮次）+ 未提交的进行中内容。
        agent 模式：历史旁路（上下文在 claude 会话），额外带 agent_session_id 供审计。
        """
        with self._lock:
            snap = {
                "system": self._system,
                "summary": self._summary,
                "history": list(self._history),
                "user_turn": self._user_turn,
                "assistant_full": self._assistant_full,
            }
        if self._agent is not None:
            try:
                snap["agent_session_id"] = self._agent.session_id
            except Exception:
                pass
        return snap

    @property
    def tts_busy(self):
        """TTS 是否在播/待播（回声门控依据）。GIL 原子读，mic 回调每块直读无锁。"""
        return self._tts_busy

    @property
    def turn_active(self):
        """一轮对话是否仍在进行：LLM 读流线程在途（跨句停顿仍算）或 TTS 队列在播/待播。

        `_tts_busy` 在流中句与句之间也会短暂回落（最近提交的 Job 播完即回落），不能单独
        代表"一轮播完"——live2d「一轮播完自动复位」用本属性判真实播完（防 LLM 句中停顿
        被误判为播完、气泡中途收掉/表情提前归位）。GIL 原子读，无锁。"""
        return (self._stream_thread is not None and self._stream_thread.is_alive()) \
            or self._tts_busy

    def _build_messages_locked(self):
        """构建本次 LLM 请求的消息列表（调用方持锁）：system(+摘要)+历史+最新用户问题。"""
        system = self._system
        if self._summary:
            system = system + "\n\n【此前对话摘要】\n" + self._summary
        if self._tools:
            system = system + "\n\n" + self._tools_prompt()
        return ([{"role": "system", "content": system}]
                + list(self._history)
                + [{"role": "user", "content": self._user_turn}])

    def _tools_prompt(self):
        """生成注入 system 的工具文档（docs/llm-tools.md §5.2，照 Alife UpdatePrompt 精简）。

        调用方持锁（只读 self._tools）。system 永远放消息最前，工具文档拼在摘要之后。
        """
        lines = [
            "## 工具调用",
            "你可以通过输出 XML 标签调用工具来获取实时信息：",
            "- 调用方式：<工具名 参数=\"值\"/>（自闭合）。可一次调用多个。",
            "- 可用工具：",
        ]
        for t in sorted(self._tools.values(), key=lambda x: x.name):
            lines.append("  " + t.to_prompt_doc())
            if t.explanation:
                lines.append("    " + t.explanation)
            if t.present:
                # 结果"呈现方式"指令（如笑话要完整逐字讲）：提前告知，收到结果后按它呈现
                lines.append("    [呈现要求] " + t.present)
        lines.append("- 每轮对话都可以调用工具，且可以随时再次调用：用户追问新日期/新城市等"
                     "此前结果没覆盖的信息时，重新调用对应工具获取，不要用旧结果硬答，"
                     "也不要说\"没有数据/查不到\"——先调用工具试试。")
        lines.append("- 工具结果里的数据是权威事实，直接据此准确回答一次：不要复述/重复已说过的"
                     "内容，不要编造数据里没有的具体数字（用户实测 2026-09-16：模型重复答了"
                     "两版且自相矛盾）。")
        lines.append("- 调用前先说一句过渡语（用户听得到），然后输出标签，等收到 [工具结果] 后继续回答。")
        lines.append("- 注意：& < > 等字符要用 &amp; &lt; &gt; 转义；标签本身不会被用户听到。")
        return "\n".join(lines)

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
            dbg("HARD_STOP gen=%d" % self._gen)
            if self._user_turn:
                self._history.append({"role": "user", "content": self._user_turn})
                self._user_turn = ""
            self._assistant_buf = ""
            self._assistant_full = ""
            self._assistant_display = ""
            self._agent_tts_buf = ""       # agent 流式 TTS 缓冲作废
            self._agent_tts_played = ""    # 流式已播累计同样作废
            self._pending_mood_announce = ""
            self._tool_results_inflight = []   # 在途工具结果作废（不 commit）
            self._stream_thread = None
            self._merge_deadline = None    # 有挂起的合并窗口 → 作废（"停下"不续发）
            self._tts.interrupt()          # 立即切音频 + 清队列（快操作）
        if self._agent is not None:
            # agent 模式："停下"= ESC。中断当前回合（进程/会话存活、历史保留），
            # "停下"本身绝不进 agent 上下文（KWS 旁路吞掉触发块）。非阻塞。
            self._agent.abort()
        self._maybe_compress()             # 历史变了，检查是否需要压缩

    # ---------------- TTS 提交 + 忙碌跟踪（voice0 Job.done，不改 voice0）----------------
    def _clean_for_tts(self, sentence):
        """剥掉语音不念的标记：引号/【询问】/心态/括号。返回清理后文本（幂等）。

        agent 流式去重用（_agent_tts_played 累计的是清理后文本），与 _submit_tts 同源。
        """
        sentence = self._QUOTE_RE.sub("", sentence)    # 剥引号：`。”`→`。`、`？"`→`？`（TTS 不念引号）
        sentence = self._ASK_RE.sub("", sentence)      # 【询问】标记剥掉不念（正文/存档保留）
        if self._mood_marker:                          # 心态标记【心态：xxx】只在送 TTS 时剥掉不读
            sentence = self._MOOD_SUB.sub("", sentence)
        sentence = self._BRACKET_RE.sub("", sentence)  # 剥括号：`？》】`→`？`、`《…`→`…`（TTS 不念括号）
        sentence = self._PUNCT_RUN_RE.sub(r"\1", sentence)  # 连续相同标点（……、——等）留一个：
                                                       # 念重复标点合成怪声（鬼故事"过去……"实测）
        sentence = self._TAG_RE.sub("", sentence)     # 兜底剥未知/残留 XML 标签（--tools 兜底）
        return sentence

    @staticmethod
    def _tail_overlap(played, result):
        """result 开头在 played 尾部已播的重叠字符数（0=无重叠）。

        只匹配 **played 尾部**——过渡句（工具调用前的"我把…捋一捋哈"，不在最终结论里）
        自然被排除；最终结论开头被流式 partial 整句播过的部分返回其长度，供重播跳过。
        保守：找不到尾部匹配 → 0（全量重播，宁重复不丢内容）。
        """
        if not played or not result:
            return 0
        n = min(len(played), len(result))
        for i in range(n, 0, -1):
            if played.endswith(result[:i]):
                return i
        return 0

    @staticmethod
    def _lcp(a, b):
        """a 与 b 的最长公共前缀长度。"""
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    @staticmethod
    def _overlap_ratio(a, b):
        """b 的内容有多少已在 a 中出现过（最长公共子序列长度占比，保序、容忍错位）。

        结论本体是否已基本被流式播过的判据：流式切句/丢标点让 `played` 与 `clean_full`
        中段错位（鬼故事实测同段文本，particle 切句吞句号、`。」` 独立段被丢），
        `_tail_overlap`（比 played 尾部）匹配不上 → skip=0 → branch c 整段重播
        "好啊阿阳"两遍。LCS 不要求连续，几处错位不影响 → 比率高（≥0.6）= 已全量播过。
        一维滚动 DP，O(n·m)，文本几百字，每回合一次可忽略。
        """
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return 0.0
        prev = [0] * (m + 1)
        for i in range(n):
            cur = [0] * (m + 1)
            ai = a[i]
            for j in range(m):
                if ai == b[j]:
                    cur[j + 1] = prev[j] + 1
                else:
                    cur[j + 1] = prev[j + 1] if prev[j + 1] >= cur[j] else cur[j]
            prev = cur
        return prev[m] / m

    def _submit_tts(self, sentence):
        """提交给 TTS 并登记忙碌跟踪（首个任务起守护 watcher，排空后 _tts_busy 回落）。"""
        if not any(ch.isalnum() for ch in sentence):
            return      # 纯标点/空白段（换行残留的 `"》》`、`"` 等）不送 TTS——避免合成标点怪声
        # 播放时间轴心态发射（2026-09-19）：心态标记文本到达即发 = 全挤在 LLM 流结束的
        # ~1s 里、音频却要播几十秒——live2d 表情全程卡最后一个标签。改成本句心态在提交
        # 前提取（此刻标记还没被 _clean_for_tts 剥掉，_with_pending_mood 已把它拼到句首），
        # 随 submit 带给 SayTTS 播放链，在"本句实际开播"瞬间发射（说话框同款 job.done 时序）。
        mood = self._leading_mood(sentence)
        sentence = self._clean_for_tts(sentence)
        if not any(ch.isalnum() for ch in sentence):
            return      # 剥净后纯标点/纯标记（如独立 `。”`、`【心态：xxx】`）也不送
        dbg("TTS_SUBMIT text=%r mood=%r" % (sentence[:24], mood))
        # KWS「停下」宽守卫已于 2026-09-14 停用（见 kws_guard_active 注释）：原来按估算播放
        # 时长顺延屏蔽"停下"，实测把用户真"停下"整段吞掉（222157 日志 6~7 次守卫命中全是
        # 用户在重复说"停下"），真"停下"随时生效，播放期随时可打断。
        if self._mood_supported:
            job = self._tts.submit(sentence, mood=mood)   # SayTTS 播放链：本句开播时发心态
        else:
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

    def echo_guard_active(self):
        """结论重播后回声自屏蔽窗口内（agent 结论被打断重播后 ~1.5s）：新 ASR 定稿句
        大概率是"被切音频的回声"（重播刚起、AI 上一句尾音还在房间里绕）→ 调用方丢弃，
        否则这条幻影句会落在 post-commit 窗口里把重播打断（2026-09-14 实测冻结根因：
        "金价卡在'给你'"——过渡句被切、尾音回声喂 ASR → 幻影句 → gen+1 取消重播、
        GPU 上重播两句都在合成中 → 全部 stale 跳过 → 无音频 + 气泡冻住）。"""
        return self._replay_echo_guard > 0 and time.monotonic() < self._replay_echo_guard_until

    def kws_guard_active(self):
        """KWS「停下」自屏蔽 —— **已停用**（2026-09-14 用户实测回归，恒 False）。

        原 ⑦ 设计：AI 自播/重播期回声门控把 mic 喂给"停下"KWS，AI 自己的音频可能自触发 →
        hard_stop 自杀，故按估算播放时长（len/5+3s/句）屏蔽"停下"。实测推翻：
        - `sessions/debug_tts_20260914_222157.log`：金价/鬼故事播放期的守卫命中（6~7 次）
          **全是用户真在说"停下"**（间隔 2.5~12s 的重复尝试），全部被吞——用户实测
          "说了好多次都没反应"。"播放期只听'停下'"的文档契约被打破。
        - AI 音频自触发"停下"（需 phonetics 恰好匹配 tíng xià）在**所有真实日志零实例**；
          2026-09-14 金价冻结真根因是 branch c 尾差打断（⑧），已修，非 KWS 自触发。
        - KWS 无法从声学区分"AI 回声"与"真·停下"，任何按播放时长的宽守卫都会连真"停下"
          一起吞。
        停用后真"停下"随时生效（含流式播放/自然播放/重播全程）；若将来 AI 音频真自触发，
        正解是 AEC 回声消除（从 mic 信号减掉喇叭参考），不是宽守卫。保留方法签名 + main 的
        `_on_interrupt` 检查点，便于将来按 `replay_echo_guard` 窄回声窗口门控复用。
        """
        return False

    # ---------------- 历史压缩（事件驱动后台线程，不阻塞对话）----------------
    def _maybe_compress(self):
        """提交后检查上下文用量；超阈值且空闲 → 派一次性后台线程压缩旧历史。

        事件驱动（不常驻监控）：条件=「无在途 LLM 流 + 无压缩在跑 + 用量超阈值 +
        历史够多」。压缩是网络调用，放后台线程做，下一轮对话照常走当前快照。
        agent 模式整条旁路（上下文在 claude 会话里，claude 自己压缩）。
        """
        if self._agent is not None:
            return
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
    def feed_asr_sentence(self, result, *, barge_audio=False):
        """新定稿句。快操作，不阻塞识别。

        barge_audio=True：**文本输入源用**——只要 TTS 还在播（哪怕 LLM 已吐完、仅音频在播）
        也立即 interrupt 切掉作废音频。语音定稿句默认 False：LLM 已答完、仅音频在播时**不**
        打断（让回答播完、新回复排队）——语音/文本打断语义不同，用户拍板。

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
            if self.echo_guard_active():
                # 结论重播后的回声自屏蔽窗口内 → 丢弃（AI 被切音频的回声，非用户意图）。
                # 只拦"定稿句"，麦克风采集/partial 不受影响；窗口过后恢复正常。
                dbg("FEED DROP(echo guard) gen=%d %r" % (self._gen, text[:20]))
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
            dbg("FEED gen=%d text=%r" % (self._gen, text[:24]))
            self._assistant_buf = ""          # 旧流作废：清缓冲与完整文本
            self._assistant_full = ""
            self._assistant_display = ""
            self._agent_tts_buf = ""          # agent 流式 TTS 缓冲作废（新句取代旧回合）
            self._agent_tts_played = ""       # 流式已播累计同样作废
            self._pending_mood_announce = ""  # 攒着的心态标记段同样作废
            if in_flight or post_commit or (barge_audio and self._tts_busy):
                self._tts.interrupt()         # 在途吐词 / 已答未开播 / 文本强打断 → 切掉作废音频
            self._stream_thread = None        # 在途流作废（gen 已变，旧线程自行退出）
            if self._merge_window > 0:
                self._merge_deadline = time.monotonic() + self._merge_window
                launch = False
            else:
                self._merge_deadline = None
                launch = True
        if post_commit and self._on_merge_rollback:
            self._on_merge_rollback()
        if self._agent is not None and in_flight:
            # agent 模式：新句取代在途回合 → 先中断（等价 ESC）再重发累计；非阻塞。
            self._agent.abort()
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
        # 撤答复连带撤工具结果（顺序：问题 → [工具结果] → 答复）：残句+新句重发时不带旧工具结果
        while (self._history and self._history[-1]["role"] == "user"
               and self._history[-1]["content"].startswith("[工具结果]")):
            self._history.pop()

    def _launch_llm(self):
        """把本轮累计发给 LLM（合并窗口过期 / 窗口=0 立即）。调用方不持锁。

        agent 模式分支：不发消息列表，直接 `agent.submit()`（上下文在 claude 会话），
        结果异步经 `_on_agent_result` 回来，由 `_agent_stream_thread` 收尾。
        """
        with self._lock:
            if self._closed or not self._user_turn:
                return
            if self._stream_thread is not None and self._stream_thread.is_alive():
                return                 # 防御：不应有在途流（feed 已置 None）
            self._merge_deadline = None
            self._turn_first_submit_ts = None  # 新一轮：首句提交时刻锚点重置
            if self._mood_marker:
                self._mood = None              # 新一轮：心态标记重新解析
                self._mood_pos = 0             # 扫描位置复位（从头扫新回合的标记）
            gen = self._gen
            if self._agent is not None:
                text = self._user_turn
                evt = threading.Event()
                self._agent_evts[gen] = evt    # 该回合的收尾唤醒事件
                self._assistant_display = ""
                self._agent_tts_played = ""    # 新回合：流式已播累计清零（去重用）
                self._pending_mood_announce = ""  # 新回合：攒着的心态标记段清零
                agent_launch = (text, gen, evt)
                messages = None
            else:
                agent_launch = None
                messages = self._build_messages_locked()   # system(+摘要)+历史+最新用户问题
        if agent_launch is not None:
            text, gen, evt = agent_launch
            if self._on_llm_start:
                self._on_llm_start()           # 控制台 "→ LLM 请求中…"（agent 也叫这行）
            self._agent.submit(text, ctx=gen)  # 非阻塞；结果经 on_result 回来
            t = threading.Thread(target=self._agent_stream_thread, args=(gen, evt),
                                 name="dialogue-agent", daemon=True)
            with self._lock:
                self._stream_thread = t
            t.start()
            return
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
            parser = self._tool_parser               # tools 关 = None → 单轮原路径（零变化）
            max_rounds = self._tool_max_rounds if self._tools else 1
            round_no = 0
            while True:
                round_no += 1
                if parser is not None:
                    parser.reset()                   # 每轮全新生成，标签不跨轮
                pending = []                         # 本轮捕获的工具结果（回灌文本）
                round_full0 = len(self._assistant_full)  # 本轮开头正文长度：工具调用前是否已吐
                                                         # 过正文（判定"调用前已说过渡"）
                pre_transition = False                  # 本轮工具调用前模型吐过正文 → 过渡句已出声
                for delta in self._llm.stream_chat(messages):
                    if gen != self._gen:             # 已被更新请求取代 → 弃流（生成器 close 关连接）
                        return
                    if parser is not None:
                        clean, calls = parser.feed(delta)   # 边流边解析（工具标签剥掉）
                        delta = clean                # 正文（已剥标签）走原管线
                    else:
                        calls = ()
                    with self._lock:
                        if gen != self._gen:
                            return
                        self._assistant_buf += delta
                        self._assistant_full += delta
                        if self._mood_marker:
                            self._parse_mood_locked()
                        buf = self._assistant_buf
                    if delta and self._on_ai_delta:  # 纯标签 delta（clean 空）不刷新预览
                        self._on_ai_delta(delta, buf)
                    self._emit_sentences(gen)
                    if calls:
                        # 本轮工具调用前模型已吐过正文（可能已切句出声，或整段在缓冲里即将送
                        # 出）→ 过渡句已/将被用户听到。结果回灌轮据此注入"不重复开场过渡"。
                        # 此刻判（而非回灌时判）：_assistant_full 之后还会继续累加本轮后续文本。
                        if len(self._assistant_full) > round_full0:
                            pre_transition = True
                        # 工具调用前的过渡句先出声：_find_cut 只切句末标点，过渡句
                        # （"好的我来查一下"）无边界会滞留到最终答案才播——工具执行期
                        # 用户干等。此处把累积缓冲整段送出，TTS 开播后再跑工具。
                        with self._lock:
                            if gen != self._gen:
                                return
                            tail = self._assistant_buf.strip()
                            self._assistant_buf = ""
                        if tail:
                            tail = self._with_pending_mood(tail)   # 拼回心态标记（TTS 剥掉不念）
                            self._submit_tts(tail)
                            if self._on_ai_sentence and any(ch.isalnum() for ch in tail):
                                self._on_ai_sentence(tail)
                        for c in calls:              # 工具执行在锁外（可能走网络，毫秒~超时）
                            pending.append(self._run_tool(c))
                if not pending:
                    break                            # 没调工具 = 单轮 = 旧行为
                # 呈现要求：本轮调用的工具里带 present（如笑话"完整逐字讲"）→ 覆盖通用
                # "不要重复已说过的内容"（那条是防数据工具复述摘要，对笑话是反效果——
                # 2026-09-19 实测 DeepSeek 把笑话原文压成一句评论"这个太损了…"）。"不要
                # 编造"对所有工具保留（笑话也不许另编）。
                presents = [t.present for c in calls
                            if (t := self._tools.get(c.name)) is not None and t.present]
                msg = ("[工具结果]\n以下为工具返回的权威数据，据此直接回答用户、一次说清即可："
                       "不要编造数据里没有的数字。")
                if pre_transition:
                    # 调用前过渡句已出声（用户听得到），回灌轮别再重复开场（2026-09-19 笑话
                    # 实测"好呀…给你讲一个"+"阿阳想听笑话呀…"两段过渡）。只对"调用前真吐过
                    # 正文"的轮加——纯 `<get_time/>` 无过渡时不注入，模型照常自己开头。
                    msg += (" 调用工具前你已经说过一句过渡语了——拿到结果后**直接开始说内容**，"
                            "不要再重复一遍开场过渡（如“我查一下”“我给你讲一个”）。")
                msg += (" 呈现要求：%s。" % "；".join(dict.fromkeys(presents)).rstrip("。")
                        if presents else " 不要重复已说过的内容。")
                msg += "\n" + "\n".join(pending)
                if round_no >= max_rounds:
                    # 到轮数上限：最后一批工具结果仍回灌进历史（工具已执行、侧效应已发生），
                    # 但不再起新的 LLM 轮（防工具无限循环）。
                    with self._lock:
                        if gen != self._gen:
                            return
                        self._tool_results_inflight.append(msg)
                    break
                # 工具续轮：结果回灌为一条 user 消息，追加本地 messages 再流一轮。
                # 不重新 _launch_llm——同一 gen、同一条流线程，barge-in/post-commit/回声门控
                # 把整轮（含工具续轮）看作一轮，语义正确；工具结果暂存 inflight，
                # commit 时插在 user 与 assistant 之间进 _history（供存档/压缩）。
                with self._lock:
                    if gen != self._gen:
                        return
                    self._tool_results_inflight.append(msg)
                    round1_assistant = self._assistant_full.strip()
                # 续轮上下文：把本轮助手正文作为 assistant 消息喂给下一轮（OpenAI 工具轮同款
                # 协议：assistant 过渡 → user 工具结果 → assistant 直接续正文）。模型必须看
                # 到自己调用工具前已说过的过渡句——否则它不知道、拿到结果后又自己开场一遍
                # （2026-09-19 实测"我给你讲个苏联笑话"+"来 给你讲个苏联笑话"两段过渡，
                # 纯提示词禁令拦不住：模型眼里的对话只有 question → [工具结果]，它不认为
                # 自己开过场）。纯 `<get_time/>` 无正文时不加空 assistant 消息。
                if round1_assistant:
                    messages.append({"role": "assistant", "content": round1_assistant})
                messages.append({"role": "user", "content": msg})
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
                full = self._assistant_full        # 完整回复（保留心态标记：进历史/回调，TTS 才剥）
                tail = self._assistant_buf.strip()  # 未切句的残句也要播出来
                if tail:
                    # tail 直通路径不经过 _emit_sentences，须手动拼回攒着的心态标记段
                    # （LLM 无标点整段落 tail 时标记曾丢失，二轮起控制台看不到，2026-09-14）
                    tail = self._with_pending_mood(tail)
                    self._submit_tts(tail)
                self._assistant_buf = ""
                self._assistant_full = ""
                if self._mood_marker and self._mood is None:
                    self._mood = "平和"            # LLM 没带标记 → 默认心态
                self._commit_locked(full)
                self._stream_thread = None
            if self._on_ai_sentence and tail and any(ch.isalnum() for ch in tail):
                self._on_ai_sentence(tail)
            if self._on_ai_done and full.strip():
                self._on_ai_done(full)
            self._maybe_compress()

    # ---------------- 工具执行（--tools，LLM 模式）----------------
    def _run_tool(self, call):
        """执行一个已捕获的工具调用，返回回灌给 LLM 的结果文本。

        在 LLM 流线程、**锁外**调用（工具可能走网络）；`Tool.run()` 自带超时/异常/截断守卫，
        绝不拖死语音流线程。诊断经 `on_tool` 回调（供主程序控制台打 `[工具] …含耗时`）。
        """
        name, attrs = call.name, call.attrs
        tool = (self._tools or {}).get(name)
        if tool is None:
            return "调用 <%s> 失败：[未知工具 %s]" % (name, name)
        t0 = time.time()
        ok, text = tool.run(attrs)
        dt = time.time() - t0
        if self._on_tool:
            try:
                self._on_tool(name, attrs, text, dt, ok)
            except Exception:
                pass                             # 诊断回调失败不影响主链路
        label = "<%s/>" % name if not attrs else "<%s %s/>" % (
            name, " ".join('%s="%s"' % (k, v) for k, v in attrs.items()))
        if not ok:
            return "调用 %s 失败：%s" % (label, text)
        return "%s 返回：%s" % (label, text)

    # ---------------- agent 模式（brain=agent，旁路历史/压缩/系统提示词）----------------
    def _on_agent_partial(self, ctx, delta):
        """agent 流式出字：控制台显示（默认）+ 可选流式送 TTS（_agent_stream_tts 开时）。
        在 agent 循环线程执行，快操作（持锁累加 + 控制台刷新 + 非阻塞入队），不阻塞生成。
        """
        with self._lock:
            if self._closed or ctx != self._gen:
                return
            self._assistant_display += delta
            self._assistant_full += delta        # 流式累计完整文本：心态标记实时解析
            if self._mood_marker:
                self._parse_mood_locked()        # 心态实时切 live2d 表情（与 LLM 路径同源）
            disp = self._assistant_display
            dbg("PARTIAL ctx=%s buf=%d delta=%r" % (ctx, len(self._agent_tts_buf), delta[:20]))
            if self._agent_stream_tts:
                self._agent_tts_buf += delta
                self._agent_last_delta_ts = time.monotonic()   # idle 切句判定基准
                self._flush_agent_stream_locked()   # 边生成边按句送 TTS
        if self._on_ai_delta:
            self._on_ai_delta(delta, disp)

    def _agent_mood_pending(self, buf):
        """buf 含未闭合的心态标记（跨 delta 到达中）→ 暂不切句，等标记闭合。

        心态标记【心态：xxx】可能被 delta 切开（如「【心态」「：」「期待】」），
        未闭合就切句送出，_MOOD_SUB 剥不掉半截标记，TTS 会把「【心态」念出来。
        """
        if not self._mood_marker:
            return False
        if self._MOOD_RE.search(buf):
            return False                       # 已有完整闭合标记，可切
        return bool(re.search(r"[【\[]心态", buf))   # 有标记字样但未闭合

    def _flush_agent_stream_locked(self):
        """agent 流式增量按句切出送 TTS（仅 _agent_stream_tts 时调用）。调用方持锁。

        心态标记未闭合不切句；切出的句子走 `_submit_tts`（剥心态/引号/括号，纯标点丢弃）。
        在 agent 循环线程执行——submit 非阻塞入队（微秒级），绝不阻塞 agent 生成。
        """
        while self._agent_tts_buf:
            if self._agent_mood_pending(self._agent_tts_buf):
                return                          # 心态标记跨 delta 到达中，等补齐
            cut = self._find_cut(self._agent_tts_buf)
            if cut is None:
                # 无标点无标记 → 若缓冲以句末语气词结尾且够长，视为完整句先送出
                # （白话口语"…哈/哦/吧"是天然句尾，工具调用期间不再干等；切错只多一停顿）。
                b = self._agent_tts_buf
                if len(b) >= 4 and b[-1] in self._PARTICLES \
                        and not self._agent_mood_pending(b):
                    self._agent_tts_buf = ""
                    dbg("FLUSH particle %r" % b.strip()[:20])
                    self._announce_agent_sentence(b.strip())
                return
            sentence = self._agent_tts_buf[:cut].strip()
            self._agent_tts_buf = self._agent_tts_buf[cut:]
            dbg("FLUSH cut@%d %r" % (cut, sentence[:20]))
            self._announce_agent_sentence(sentence)

    def _with_pending_mood(self, sentence):
        """把攒着的纯心态标记段（_pending_mood_announce）拼到真实句子的显示文本上并清零。

        `_find_cut` 会把句首 `【心态：xxx】` 单独切走（作切点），攒到下一个真实句子拼回
        显示——否则标记只活在流式预览里、定稿行看不到（2026-09-14）。`_submit_tts` 会再
        剥掉不念，控制台/live2d/历史保留。**所有"切句→送 TTS→通知定稿"的出口都走它**
        （`_emit_sentences` / `_announce_agent_sentence` / 两个 `tail` 直通路径），漏一处
        就丢标记（实测 LLM 无标点整段落 tail、二轮起标记消失）。
        """
        s = self._pending_mood_announce + sentence
        self._pending_mood_announce = ""
        return s

    def _announce_agent_sentence(self, sentence):
        """agent 流式切出的句子：送 TTS + 累计已播 + 通知控制台定稿行。

        与 LLM 路径 `_emit_sentences` 的 on_ai_sentence 语义一致——控制台每句一行完整
        文本（首句带 `[ts]` 首答时间，由主程序 on_ai_sentence 打印）。否则 agent 流式
        句只走 on_ai_delta 的**累计全文预览**，屏幕会留下"过渡句+结论拼一行、带省略号
        截断"的残留（用户实测 2026-09-14）。纯心态标记/纯标点段（剥净后无可念内容）
        → 不送 TTS、不打控制台行（独立 `【心态：开心】` 不刷一行）。
        """
        if not any(ch.isalnum() for ch in self._clean_for_tts(sentence)):
            if "心态" in sentence:
                # 纯心态标记段：攒着拼下句显示。连续相同标记去重（实测 agent 结论开头
                # 自带【心态：xxx】，而它到达前又单独流式吐过同款标记 → 不查重会拼成
                # 【心态：开心】【心态：开心】… 显示重复、live2d 表情重复触发）。
                if not self._pending_mood_announce.endswith(sentence):
                    self._pending_mood_announce += sentence
            return
        sentence = self._with_pending_mood(sentence)  # 标记拼回显示（_submit_tts 会剥掉不念）
        self._submit_tts(sentence)
        self._agent_tts_played += self._clean_for_tts(sentence)  # 累计已播（去重用）
        if self._on_ai_sentence:
            self._on_ai_sentence(sentence)

    def _idle_flush_agent_stream_locked(self):
        """agent 静默期流式缓冲兜底切句（仅 _agent_stream_tts 时调用）。调用方持锁。

        触发条件：缓冲非空 + 心态标记已闭合 + 最近一次增量已停滞 ≥ _AGENT_IDLE_FLUSH_MS。
        情形 = agent 在跑工具/思考（长时间无 delta），缓冲里滞留的是一句完整过渡句
        （如"好，讲个新笑话给你"——"你"不是句末语气词也没有标点，正常切句永远等不到
        切点，实测卡 3.7s 干等）。此刻把它整段送出声：过渡句先播、工具跑完结论到达时
        `_tail_overlap` 会把过渡句从结论前缀排除（去重），不会播两遍。
        """
        if not self._agent_tts_buf:
            return
        if self._agent_mood_pending(self._agent_tts_buf):
            return                               # 心态标记跨 delta 到达中，等补齐
        if time.monotonic() - self._agent_last_delta_ts < self._AGENT_IDLE_FLUSH_MS:
            return                               # 还在活跃产出（agent 打字中），不打断节奏
        sentence = self._agent_tts_buf.strip()
        self._agent_tts_buf = ""
        dbg("FLUSH idle %.2fs %r" % (time.monotonic() - self._agent_last_delta_ts,
                                     sentence[:20]))
        self._announce_agent_sentence(sentence)   # 过渡句算"已播"，结论去重

    def _on_agent_result(self, ctx, text, is_error):
        """agent 最终结论回来（agent 循环线程）。ctx 作废（被打断/被取代）→ 只唤醒不碰状态。

        锁内只做状态填充 + evt.set()（最后一步才唤醒，保证收尾线程读到就绪状态）。
        _agent_stream_tts 开时：最终结论到达按三态收尾（见函数内注释）——结论整段已进
        流式队列 → 不打断自然播完（**打断会把已入队未开播的结论音频全取消 → 完全静音**，
        2026-09-13 实测）；已播全是结论前缀 → 不打断只补送剩余；已播含过渡句/思考段 →
        立即 interrupt 打断 + 从 skip 重播剩余（`_tail_overlap` 尾部重叠——过渡句不在
        结论里自动排除，否则"阿阳"整句播两遍）。**去重重叠只对"最后一个心态标记之后"的
        结论本体算**——ResultMessage 全文常以过渡句开头（"我再确认一下大后天的天气哈【心态：
        开心】阿阳…"），对全文算会把过渡句误当"已播的结论开头"跳过打断（2026-09-13 实测：
        结论都出来了还不打断）。
        """
        with self._lock:
            if not self._closed and ctx == self._gen:
                if is_error:
                    self._agent_error = text or "agent 出错了"
                else:
                    self._agent_error = None
                    self._assistant_full = text or ""
                    raw = text or ""
                    if self._agent_stream_tts:
                        # 结论以最后一个心态标记为界：标记前是过渡句/思考段（要打断），
                        # 标记后才是结论本体。去重重叠只对结论本体算——若对整段全文算，
                        # ResultMessage 含过渡句开头时（"…哈【心态：开心】阿阳…"），
                        # 过渡句会被误当"已播的结论开头"跳过打断（2026-09-13 实测）。
                        concl_start = 0
                        for m in self._MOOD_RE.finditer(raw):
                            concl_start = m.end()
                        clean_full = self._clean_for_tts(raw[concl_start:])
                        played = self._agent_tts_played
                        # 换行归一化（2026-09-14 实测笑话场景）：流式切句 `_find_cut` 在
                        # `\n` **边界处切**（`\n` 归属下句/被吞），入队的句子文本**不含** `\n`；
                        # 而 clean_full 保留 `\n` → 两侧字符错位 → `_tail_overlap` 算不出重叠
                        # （skip=0）→ 明明结论几乎全量流式，却误落 branch c 取消+整段重播。
                        # 比对齐：`\n` 对 TTS 发音无影响，比较/去重前剥掉即可（实录见
                        # debug_tts_20260914_183135.log：skip=0 而实际 6 句已流式全进队）。
                        nclean = clean_full.replace("\n", "")
                        nplayed = played.replace("\n", "")
                        # 结论本体已基本被流式播过（LCS 占比 ≥0.6，容忍流式切句/丢标点的
                        # 中段错位）→ 视为全量覆盖不打断不重播。根因（2026-09-14 鬼故事实测）：
                        # ResultMessage 全文含过渡句前缀（agent 只在开头带一次心态标记 →
                        # concl_start 掐不到过渡句），而过渡句+整篇都在 played **开头**——
                        # `_tail_overlap` 只比 played 尾部匹配不上（skip=0）→ branch c
                        # interrupt + 整段重播"好啊阿阳"两遍。LCS 保序容忍错位，比率高即
                        # 内容几乎全在 played 里（同一份文本中段切句错位仍 ~0.7）。
                        # 阈值 0.6：只流式了结论开头一点（<60%）不算——那是真没播完要补送。
                        if self._overlap_ratio(nplayed, nclean) >= 0.6:
                            skip = len(nclean)
                        else:
                            skip = max(self._tail_overlap(nplayed, nclean),
                                       self._lcp(nplayed, nclean))
                        remainder = nclean[skip:] if skip < len(nclean) else ""
                        clean_full = nclean                # 后续用归一化版（remainder/长度/守卫）
                        # 四态（2026-09-13/14 实测教训：打断会把"已入队未开播"的结论音频
                        # 全取消 → 完全静音；故只有"已播含过渡句且残尾值得重播"才打断）：
                        #  a) 结论整段已进流式队列（skip 盖满）→ 不打断，让队列自然播完
                        #  a') 结论几乎全量流式、只差极短残尾（≤_TRIVIAL_TAIL）→ 不打断：
                        #      实测残尾是流式缓冲没切出的句末"。"（skip 155/156），旧逻辑
                        #      落 branch c 把整条队列 interrupt 掉、重播又吐不出 1 字"。"，
                        #      = 金价冻结"说了阿阳就卡住"真根因（2026-09-14）。队列里就是
                        #      结论本体，让它自然播完；残尾是纯标点/尾词，值不得打断。
                        #  b) 已播全是结论前缀（skip==len(played)，无过渡句）→ 不打断，
                        #     只补送未播的剩余结论（残句在 remainder 里补齐）
                        #  c) 已播含过渡句/思考段且残尾有实义（skip<len(played)）→ 立即
                        #     打断旧播放，从 skip 重播剩余结论（"阿阳"只播一遍的保证）
                        self._agent_tts_buf = ""          # 流式缓冲作废（残句由下方补齐）
                        self._agent_tts_played = ""       # 本回合去重完毕，清累计
                        self._pending_mood_announce = ""  # 攒着的心态标记段同样作废
                        dbg("RESULT ctx=%s raw=%r" % (ctx, raw[:30]))
                        dbg("RESULT concl_start=%d clean_full=%r" % (concl_start, clean_full[:30]))
                        dbg("RESULT played=%r skip=%d len_clean=%d remainder=%r"
                            % (played[:30], skip, len(clean_full), remainder[:20]))
                        covered = skip >= len(clean_full) or \
                            len(remainder) <= self._TRIVIAL_TAIL
                        dbg("RESULT branch=%s interrupt=%s" %
                            ("a" if covered else ("c" if skip < len(played) else "b"),
                             not covered and skip < len(played)))
                        if covered:
                            self._assistant_buf = ""      # 整段已进队/残尾可忽略 → 无需重播
                            # （KWS「停下」守卫已停用：自然播放期真"停下"随时生效）
                        else:
                            self._assistant_buf = remainder  # 补送/重播未播部分
                            if skip < len(played):
                                self._tts.interrupt()     # 切掉过渡句/思考段旧播放
                                # 被切音频尾音此刻还在房间里绕，重播又马上起——回声会被门控
                                # grace 喂进 ASR 闭成幻影句落在 post-commit 窗口打断重播
                                # （实测冻结根因）。短窗口内丢弃新 ASR 句 = 自屏蔽。
                                self._replay_echo_guard_until = \
                                    time.monotonic() + self._replay_echo_guard
                                # （KWS「停下」守卫已停用：重播期真"停下"也随时生效）
                    else:
                        self._assistant_buf = raw
                    if self._mood_marker:
                        self._mood_pos = 0        # _assistant_full 刚被权威全文替换：从头重扫，
                        self._parse_mood_locked() #  不漏结果里未被流式发射过的心态（重复幂等）
            elif not self._closed:
                # 结果回来时代际已变（新句/幻影句抢在结果前）→ 本回合结果被丢弃。打点以便
                # 区分"重播未提交"（结果被丢）与"重播提交后被杀"（结果处理了、后续又被打断）。
                dbg("RESULT DISCARD ctx=%s gen=%s" % (ctx, self._gen))
            evt = self._agent_evts.get(ctx)
            if evt is not None:
                evt.set()                          # 唤醒该回合的收尾线程

    def _agent_stream_thread(self, gen, evt):
        """agent 回合收尾线程：等结果（或回合作废）→ 切句送 TTS / 报错。

        与 LLM 路径 finally 一致：完整文本按句送 TTS、心态兜底「平和」；**不 commit 历史**
        （上下文在 claude 会话里，agent 模式旁路自实现历史），只清本轮累计。

        _agent_stream_tts 开时在等待期间做 **idle 切句**：agent 工具调用期无流式增量
        （静默数秒），缓冲里滞留的完整过渡句（如"好，讲个新笑话给你"以"你"结尾无标点无
        语气词，没有切点）会干等到最终结论才出声（实测 2026-09-14：笑话过渡句卡 3.7s）。
        用带超时的 evt.wait 循环，静默超过 _AGENT_IDLE_FLUSH_MS → 把整段缓冲先送出声。
        """
        while True:
            # 0.5s 分片睡：结果/回合作废到达 evt 立即唤醒；否则每 0.5s 检查一次 idle
            if evt.wait(0.5):
                break
            with self._lock:
                if self._closed or gen != self._gen:
                    self._agent_evts.pop(gen, None)   # 回合作废：清掉本回合唤醒事件再退
                    return
                if self._agent_stream_tts:
                    self._idle_flush_agent_stream_locked()
        dbg("STREAM_THREAD woke gen=%d" % gen)
        with self._lock:
            self._agent_evts.pop(gen, None)
            if self._closed or gen != self._gen:
                dbg("STREAM_THREAD ABORT gen=%d cur=%d closed=%s" % (gen, self._gen, self._closed))
                return                             # 回合已作废（被打断/被新句取代）
            if self._agent_error:
                err = self._agent_error
                self._agent_error = None
                self._stream_thread = None
                tail = ""
                full = ""
            else:
                err = None
                full = self._assistant_full        # 完整回复（保留心态/【询问】标记：审计用）
                self._emit_sentences(gen)          # 按句送 TTS（_submit_tts 剥心态/【询问】）
                tail = self._assistant_buf.strip()  # 未切句的残句也要播出来
                if tail:
                    # tail 直通路径须手动拼回攒着的心态标记段（与 LLM finally 同修，
                    # _emit_sentences 只处理了它切出的句子、尾句标记不拼回同样丢失）
                    tail = self._with_pending_mood(tail)
                    self._submit_tts(tail)
                self._assistant_buf = ""
                self._assistant_full = ""
                self._assistant_display = ""
                self._agent_tts_buf = ""
                self._agent_tts_played = ""
                self._pending_mood_announce = ""
                dbg("STREAM_THREAD done gen=%d" % gen)
                if self._mood_marker and self._mood is None:
                    self._mood = "平和"            # agent 没带标记 → 默认心态
                self._user_turn = ""               # 本轮累计清空（不 commit 历史）
                self._stream_thread = None
        if err is not None:
            if self._on_llm_error:
                self._on_llm_error(err)            # 控制台 "× LLM 出错"（agent 也叫这行）
            return
        if self._on_ai_sentence and tail and any(ch.isalnum() for ch in tail):
            self._on_ai_sentence(tail)
        if self._on_ai_done and full.strip():
            self._on_ai_done(full)

    def _commit_locked(self, full_text):
        """完整回复才进历史（被打断的回复 gen 不对，根本到不了这里）。调用方持锁。"""
        if self._user_turn:
            self._history.append({"role": "user", "content": self._user_turn})
        # 工具结果插在 user 与 assistant 之间（顺序：问题→[工具结果]→答复）
        for msg in self._tool_results_inflight:
            self._history.append({"role": "user", "content": msg})
        self._tool_results_inflight = []
        t = full_text.strip()    # 心态标记是模型真实输出，保留在历史/存档/上下文里
        if t:
            self._history.append({"role": "assistant", "content": t})
        self._user_turn = ""
        # 历史长度由 token 预算（_maybe_compress）管理；_max_history 仅作硬安全上限
        if self._max_history and len(self._history) > self._max_history:
            self._history = self._history[len(self._history) - self._max_history:]

    # ---------------- 心态标记（user_prompt 约定的【心态：xxx】：保留在正文，仅送 TTS 时剥掉不读）----------------
    def _leading_mood(self, sentence):
        """提取句子的心态（随 submit 带给 SayTTS 播放链，在"本句实际开播"瞬间发射）。

        在 `_clean_for_tts` **之前**调用——此刻标记还在句子里（`_with_pending_mood` 已把
        攒着的纯标记段拼回句首）。取**首个** `_MOOD_RE` 匹配：句首标签是主流；句中/句尾
        标签也兜底（`_find_cut` 把标签闭合处当切点，理论上标签总在句首，但工具过渡句等
        tail 直通路径不经过 _find_cut）。无标记 → None（继承当前心态，不切表情）。
        超纲词兜底「平和」（与 _parse_mood_locked 同口径）。mood_marker 关 → 恒 None。
        """
        if not self._mood_marker:
            return None
        m = self._MOOD_RE.search(sentence)
        if not m:
            return None
        mood = (m.group(1) or "").strip()
        return mood if mood in self._MOODS else "平和"

    def _parse_mood_locked(self):
        """从流式文本里维护 `_mood` 状态（判"本回复是否带心态标记"）。调用方持锁。

        标记**保留**在 _assistant_buf/_assistant_full 里（它是模型的真实输出：控制台打印、
        历史存档、LLM 上下文都要带上），只在 `_submit_tts` 送 TTS 那一刻被 _MOOD_SUB 剥掉。
        **发射已挪到 SayTTS 播放链**（_submit_tts 提交前 `_leading_mood` 提取、随 submit
        带入，在"本句实际开播"瞬间发 live2d 表情——文本到达即发会让全部心态挤在 LLM 流
        结束的 ~1s 里、音频播几十秒时表情全程卡最后一个标签，2026-09-19）。本函数只维护
        `_mood`（None=还没出现标记，流末 finally 兜底「平和」）与 `_mood_pos`（增量扫描
        位置，新回合 _launch_llm 复位；_assistant_full 被 agent 结果整体替换/清零时兜底
        从头扫）。
        """
        if not self._mood_marker:
            return
        full = self._assistant_full
        pos = self._mood_pos
        if pos > len(full):
            pos = 0                       # _assistant_full 被替换/清零，位置失效 → 从头扫
        while True:
            m = self._MOOD_RE.search(full, pos)
            if not m:
                break
            mood = (m.group(1) or "").strip()
            self._mood = mood if mood in self._MOODS else "平和"   # 超纲词兜底（状态标记）
            pos = m.end()
        self._mood_pos = pos

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
            if not any(ch.isalnum() for ch in self._clean_for_tts(sentence)):
                if "心态" in sentence:
                    # 纯心态标记段（如"【心态：期待】"）攒着：_emit_sentences 每 delta 调
                    # 一次，局部变量跨调用即丢（实测 LLM 流式首 delta 带标记、定稿行却无），
                    # 必须用实例属性 _pending_mood_announce（agent 流式同源，见
                    # _announce_agent_sentence）跨调用攒着拼到下一个真实句子显示上。
                    # 连续相同标记去重（同 _announce_agent_sentence，防"已流式吐过的标记
                    # + 结论开头自带的同款标记"拼成双份显示）。
                    if not self._pending_mood_announce.endswith(sentence):
                        self._pending_mood_announce += sentence
                continue                          # 纯标点段（"。。"）丢弃后继续找
            sentence = self._with_pending_mood(sentence)  # 标记拼回显示（_submit_tts 会剥掉不念）
            if first and self._reply_hold > 0:
                time.sleep(self._reply_hold)      # 锁外：给用户续句打断的机会
                with self._lock:
                    if gen != self._gen:
                        return                    # hold 期间被 barge → 弃句（绝不播）
            self._submit_tts(sentence)
            if self._on_ai_sentence:
                self._on_ai_sentence(sentence)

    def _absorb_closers(self, buf, i):
        """切点后紧随的闭引号/闭括号并入前一句（可跨换行：corpus 原文 `？\n"》》`
        分行时 `"》》` 的闭符归前句，不残留纯标点独立段——否则换行切句把它切成
        独立 Job，TTS 合成标点怪声）。"""
        n = len(buf)
        while i < n and (buf[i] in self._CLOSERS or buf[i] in self._BOUNDARY):
            i += 1
        return i

    def _find_cut(self, buf):
        """返回首个可提交切点下标；无可提交（无边界且未超长）返回 None。"""
        n = len(buf)
        if n == 0:
            return None
        # 1) 首个句末边界（.。！？…；\n）——按句切，绝不把多句并成一个 Job。
        #    旧实现取【最后一个】边界：LLM 流式逐字出字时二者等价；但 agent 模式
        #    整段回复一次性落地（_on_agent_result 整段塞入 _assistant_buf），
        #    取末边界会把整段并成一句 TTS，live2d 说话框逐句链式就失去意义
        #    （实测宋词 163 字整首塌成 1 个 Job）。
        #    先跳过开头一连串边界字符（\n 换行/句末标点）：agent 输出常以换行分段，
        #    段首 \n 若被当首边界会因「>=2 字」守卫被拒、回退取末边界再次坍塌。
        start_idx = 0
        while start_idx < n and buf[start_idx] in self._BOUNDARY:
            start_idx += 1
        first = -1
        for ch in self._BOUNDARY:
            i = buf.find(ch, start_idx)
            if i >= 0 and (first < 0 or i < first):
                first = i
        # 1a) 心态标记处也可作切点：模型常在句中切换心态（"…哈【心态：开心】阿阳…"），
        #     不当切点会把前后句粘成一个 TTS Job（合并气泡/超长句，live2d 逐句跟播失效）。
        #     **切在标签前**（标签领衔下一句）：标签语义修饰其后的内容；按闭合处切会把
        #     标签粘在前句尾巴，而 `_leading_mood` 只取句首首个心态 → 句中后一个心态被吞
        #     （实测 "…摊低成本对吧 我懂【心态：温柔】但你别硬加…" 温柔随前句 submit 后
        #     丢失，播放链只发担心/关切/期待，2026-09-19）。标签在缓冲开头（start==0）
        #     则按闭合处切、抽成纯标记段攒着拼回下句（_with_pending_mood）。
        m = self._MOOD_SUB.search(buf, start_idx)
        if m is not None:
            start = m.start()
            end = m.end()
            if first < 0 or end <= first:
                if start > 0 and len(buf[:start].strip()) >= 2:
                    return self._absorb_closers(buf, start)
                if len(buf[:end].strip()) >= 2:
                    return self._absorb_closers(buf, end)
        if first >= 0 and len(buf[:first + 1].strip()) >= 2:
            return self._absorb_closers(buf, first + 1)
        # 1b) 无合格首边界 → 退回取最后一个边界（旧行为，防句中停顿被拆）
        last = -1
        for ch in self._BOUNDARY:
            i = buf.rfind(ch)
            if i > last:
                last = i
        if last >= 0 and len(buf[:last + 1].strip()) >= 2:
            return self._absorb_closers(buf, last + 1)
        # 2) 超长无标点 → 兜底硬切（保首包延迟）。绝不撕词：先回找末 _SOFT_WINDOW
        #    字符里的软分句缝（空格/逗号类——LLM 按空格分短句，切在缝上停顿自然）；
        #    找不到才硬切 _HARD_MAX。硬切落在词中间会把词撕开（"钟|表"、"黑眼|圈"），
        #    每段独立合成、queue 缝里插停顿，听感"黑眼…停顿…圈"（实测）。
        if n > self._HARD_MAX:
            start = max(0, n - self._SOFT_WINDOW)
            ci = -1
            for ch in self._SOFT_CUT:
                i = buf[start:].rfind(ch)
                if i >= 0 and start + i > ci:
                    ci = start + i
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
        if self._agent is not None:
            self._agent.close()               # 优雅断开 claude（进程正常结束）+ 停循环线程
