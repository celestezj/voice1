# -*- coding: utf-8 -*-
"""示例：麦克风实时识别（sounddevice 16k 采集 → 常驻引擎 → 逐句回调）。

用法（中文输出需 UTF-8 编码，Ctrl+C 退出）：
    PYTHONIOENCODING=utf-8 python examples/record_mic.py [--backend paraformer] [--device cuda]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sounddevice as sd
from asr import RealtimeASR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="paraformer", help="paraformer|whisper|sherpa")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = ap.parse_args()

    asr = RealtimeASR(backend=args.backend, device=args.device, profile=True)
    asr.on_sentence(lambda r: print("[%.2fs] %s (ttfb=%.3fs)"
                                    % (r.audio_end, r.text, r.ttfb)))
    print("说话吧…（Ctrl+C 退出）", flush=True)

    def cb(indata, frames, t, status):
        # indata: (frames, 1) float32，取单声道列 → 常驻引擎（wall 轴时间戳）
        asr.ingest(indata[:, 0], source_ts=time.monotonic())

    try:
        with sd.InputStream(samplerate=16000, channels=1, callback=cb):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        asr.close()
        print("已退出。", flush=True)


if __name__ == "__main__":
    main()
