# -*- coding: utf-8 -*-
"""心态 → live2d 表情发射器（voice1 侧展示联动，只发 emotion 不碰 mouth）。

live2d（desktop_pet.py）的 `--control-port` 服务：原始 TCP，每行一个 JSON，
UTF-8 + `ensure_ascii=False` + `\n` 结尾，服务端无响应。`emotion` 的合法键
与 voice1 的 16 个心态名**完全一致**（平和/开心/…/愤怒），恒等映射，无需
转换表。嘴的自动开合由 live2d 自己的 `--listen`/`--lipsync` 负责，本类
只发 `{"emotion": ...}`——发 `{"mouth": ...}` 反会被 live2d 音频能量线程覆盖。

启用/禁用语义（用户拍板）：
- 未给 `--live2d-port` → 不启用（构造即返回，无 worker）。
- **启动测活失败** → 打印一条告知用户，彻底禁用、不重试。
- **测活成功** → 真实启用（`enabled=True`），并立即初始化发一条「平和」归位。
- **运行中发送失败**（live2d 中途退出）→ **继续如常发送**，每次失败只打印
  提醒（"live2d server 连接失败，请检查"），**不置禁用**——live2d 可能重启回来。
- `reset()`：对话结束（拜拜/静默超时）归位「平和」；`close()`：程序退出同步
  补发「平和」再停 worker。

线程纪律：`on_mood` 在 LLM 线程持锁内触发 → `emit()` 只写状态 + 唤醒 event
（微秒级，**绝不做网络 IO**）；实际发送在常驻 daemon worker（单线程串行、
最新值覆盖），保证发送顺序且不阻塞 LLM 读流。
"""

import json
import socket
import threading

_DEFAULT_HOST = "127.0.0.1"   # live2d control 服务只绑回环


class Live2dEmitter:
    """心态 → live2d 表情。构造即测活；`port=None` → 全部短路禁用。"""

    def __init__(self, port=None, host=_DEFAULT_HOST):
        self._host = host
        self._port = port
        self._enabled = False
        self._wanted = None            # 最新目标心态（覆盖语义，单 worker 串行消费）
        self._event = threading.Event()
        self._stop = threading.Event()
        if port is None:
            return                     # 未给端口 → 禁用，无 worker
        try:
            # 测活 + 初始化归位一次完成：失败须上抛给 __init__（成功静默——
            # 这条「平和」就是初始表情，无需再打印一条）
            self._send_sync("平和", log=False, raise_on_error=True)
        except OSError as e:
            print("[live2d] 表情联动关闭：连接 %s:%d 失败（%s）。请确认已先启动 "
                  "desktop_pet.py --control-port %d" % (host, port, e, port), flush=True)
            return                     # 测活失败：彻底禁用，不重试
        self._enabled = True
        threading.Thread(target=self._run, name="live2d-emit",
                         daemon=True).start()

    @property
    def enabled(self):
        return self._enabled

    def emit(self, mood):
        """心态解析到（`on_mood` 回调，LLM 线程持锁内）→ 提交最新表情。非阻塞。"""
        if not self._enabled:
            return
        self._wanted = mood
        self._event.set()

    def reset(self):
        """归位：对话结束（超时/拜拜/程序退出）发「平和」。enabled 内部短路。"""
        self.emit("平和")

    def _run(self):
        """常驻发送 worker：串行消费最新心态；空闲每 0.5s 醒一次查 stop。"""
        while not self._stop.is_set():
            if not self._event.wait(0.5):
                continue
            if self._stop.is_set():
                return
            self._event.clear()
            mood = self._wanted
            if mood is not None:
                self._send_sync(mood)

    def _send_sync(self, mood, log=True, raise_on_error=False):
        """短连接发送一条情绪。

        - `raise_on_error=True`：失败上抛（构造测活用，__init__ 捕获后禁用并打印总提示）。
        - `log=False`：失败静默（close 收尾不刷提醒）。
        - 其余（worker 运行中）：失败**打印提醒但不上抛、不置禁用**——live2d 中途退出后
          可能重启回来，故继续如常发送（用户拍板），只提醒"live2d server 连接失败，请检查"。
        """
        try:
            payload = (json.dumps({"emotion": mood}, ensure_ascii=False)
                       .encode("utf-8") + b"\n")
            with socket.create_connection((self._host, self._port), timeout=1) as s:
                s.sendall(payload)
            return True
        except OSError as e:
            if raise_on_error:
                raise
            if log:
                print("[live2d] live2d server 连接失败，请检查（%s:%d：%s）"
                      % (self._host, self._port, e), flush=True)
            return False

    def close(self):
        """程序退出：enabled 时同步补发「平和」归位（失败静默）再停 worker。"""
        if self._enabled:
            self._send_sync("平和", log=False)
            self._enabled = False
        self._stop.set()
        self._event.set()
