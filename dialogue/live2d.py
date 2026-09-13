# -*- coding: utf-8 -*-
"""voice1 → live2d 桌宠通道发射器（emotion 表情 + say 头顶说话框）。

live2d（desktop_pet.py）的 `--control-port` 服务：原始 TCP，每行一个 JSON，
UTF-8 + `ensure_ascii=False` + `\n` 结尾，服务端无响应。协议键 = 独立通道，
**给值 = 设置、`null` = 清除/复位**：

    {"emotion":"开心"}      切情绪（16 种，voice1 心态名与 live2d 完全一致，恒等映射）
    {"emotion":null}        复位默认情绪 平和
    {"say":"你好呀"}        头顶显示说话框（文字一直保持，直到显式清除）
    {"say":null}            隐藏说话框
    {"emotion":null,"say":null}   恢复初始状态（一条同时复位表情+收框）

本类只发 emotion 与 say，不碰 mouth（嘴自动开合由 live2d `--listen`/`--lipsync`
负责；voice1 若发 `{"mouth":..}` 反会被 live2d 音频能量线程覆盖）。

启用/禁用与发送语义（用户拍板）：
- 未给 `--live2d-port` → 不启用（构造即返回，无 worker）。
- **启动测活失败** → 打印一条告知用户，彻底禁用、不重试。
- **测活成功** → 真实启用（`enabled=True`），并立即补发一条**恢复初始状态**
  `{"emotion":null,"say":null}`——清掉桌宠上次遗留的气泡/情绪（启动复位）。
- **运行中发送失败**（live2d 中途退出）→ **继续如常发送**，每次失败只打印提醒
  （"live2d server 连接失败，请检查"），**不置禁用**——live2d 可能重启回来。
- `emit(mood)` 发情绪；`say(text)` 发说话框文本（逐句跟读，覆盖显示）；
  `reset()` = 恢复初始状态（表情复位 + 收框，供"拜拜/超时回休眠 / 停下 / 播完自动复位"）。

线程纪律：各触发点多在 LLM 线程持锁内（`on_mood`）或 TTS watcher 线程——公开方法
只入队（微秒级，**绝不做网络 IO**）；实际发送在常驻 daemon worker（FIFO 串行保序：
先情绪后句子文本，句与句、reset 之间不乱序）。`emit` 每轮一次、`say` 逐句，FIFO
即够，无需覆盖去重。

连接策略（用户拍板）：**常驻长连接 + 惰性重连**——worker 持一条 socket 串行发送
（`_lock` 保护，与 close 并发安全），发送前不检查状态，`sendall` 抛 `OSError` 即关旧
重建重发一次；**不探测、无心跳**（live2d 可能重启回来，下次发送自动连上）。长连接让
桌面端 `client_count` 恒为 1（"有 AI 在驱动"语义真正成立，见 desktop_pet 的
`--look-at-cursor`）。
"""

import json
import queue
import socket
import threading

_DEFAULT_HOST = "127.0.0.1"   # live2d control 服务只绑回环
_RESET = {"emotion": None, "say": None}   # 恢复初始状态：表情复位平和 + 隐藏说话框


class Live2dEmitter:
    """voice1 → live2d 通道发射器。构造即测活；`port=None` → 全部短路禁用。"""

    def __init__(self, port=None, host=_DEFAULT_HOST):
        self._host = host
        self._port = port
        self._enabled = False
        self._q = queue.Queue()          # FIFO 待发 payload（dict），worker 串行消费
        self._stop = threading.Event()
        self._sock = None                # 常驻长连接（worker 持锁惰性建立/重建）
        self._lock = threading.Lock()    # 保护 _sock：worker 与 close 并发安全
        if port is None:
            return                       # 未给端口 → 禁用，无 worker
        try:
            # 测活 + 启动复位一次完成：成功静默（这条恢复消息就位）；失败须上抛给 __init__
            self._send_sync(_RESET, log=False, raise_on_error=True)
        except OSError as e:
            print("[live2d] 表情联动关闭：连接 %s:%d 失败（%s）。请确认已先启动 "
                  "desktop_pet.py --control-port %d" % (host, port, e, port), flush=True)
            return                       # 测活失败：彻底禁用，不重试
        self._enabled = True
        threading.Thread(target=self._run, name="live2d-emit",
                         daemon=True).start()

    @property
    def enabled(self):
        return self._enabled

    def emit(self, mood):
        """切情绪。心态解析到（`on_mood` 回调，LLM 线程持锁内）→ 非阻塞入队。"""
        self._push({"emotion": mood})

    def say(self, text):
        """头顶说话框显示/跟读文本。送 TTS 的每个实际播放句（剥心态标记后）→ 非阻塞入队。"""
        self._push({"say": text})

    def reset(self):
        """恢复初始状态：一条 `{"emotion":null,"say":null}`（表情复位平和 + 收框）。
        触发点：启动 / 拜拜 / 静默超时回休眠 / "停下"打断 / 一轮播完自动复位（后者受
        `--live2d-idle-reset` 开关控制，由调用方决定是否调本方法）。"""
        self._push(_RESET)

    def _push(self, payload):
        if not self._enabled:
            return
        self._q.put(payload)

    def _run(self):
        """常驻发送 worker：FIFO 串行消费；空闲每 0.5s 醒一次查 stop。"""
        while not self._stop.is_set():
            try:
                payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if payload is None or self._stop.is_set():
                return                 # close 哨兵
            self._send_sync(payload)

    def _send_sync(self, payload, log=True, raise_on_error=False):
        """长连接惰性发送一条 JSON（持 `_lock`，worker 与 close 串行）。

        **惰性重连**（用户拍板）：发送前**不检查连接状态**——直接 `sendall`；抛 `OSError`
        （断线/RST）则关旧连接、重建新连接、**重发一次**。不探测、无心跳（live2d 可能重启，
        下次发送自动连上）。重发再失败按语义处理：

        - `raise_on_error=True`：失败上抛（构造测活用，__init__ 捕获后禁用并打印总提示）。
        - `log=False`：失败静默（close 收尾不刷提醒）。
        - 其余（worker 运行中）：失败**打印提醒但不上抛、不置禁用**——只提醒
          "live2d server 连接失败，请检查"。
        """
        data = (json.dumps(payload, ensure_ascii=False)
                .encode("utf-8") + b"\n")
        with self._lock:
            try:
                self._ensure_sock()
                self._sock.sendall(data)
                return True
            except OSError:
                self._drop_sock()        # 连接已断 → 关旧
            try:
                self._ensure_sock()      # 重建
                self._sock.sendall(data) # 重发一次
                return True
            except OSError as e:
                self._drop_sock()
                if raise_on_error:
                    raise
                if log:
                    print("[live2d] live2d server 连接失败，请检查（%s:%d：%s）"
                          % (self._host, self._port, e), flush=True)
                return False

    def _ensure_sock(self):
        """惰性建立/复用常驻长连接（调用方持 `_lock`）。1s 超时：连接失败/发送阻塞
        都不让 worker 卡死（协议无响应，发送缓冲理论上可满）。"""
        if self._sock is None:
            self._sock = socket.create_connection((self._host, self._port), timeout=1)
            self._sock.settimeout(1)

    def _drop_sock(self):
        """关掉失效连接（调用方持 `_lock`）。断线重建 / close 收尾用。"""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def close(self):
        """程序退出：丢弃队列残留，同步补发一条恢复初始状态（失败静默）再停 worker、关连接。"""
        if self._enabled:
            self._send_sync(_RESET, log=False)
            self._enabled = False
        self._stop.set()
        try:
            self._q.put_nowait(None)     # 唤醒 worker 退出
        except queue.Full:
            pass
        with self._lock:
            self._drop_sock()
