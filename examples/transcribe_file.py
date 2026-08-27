# -*- coding: utf-8 -*-
"""示例：命令行转写单个 wav（逐句打印文本 + 时间戳 + 尾字延迟）。

用法（中文输出需 UTF-8 编码）：
    PYTHONIOENCODING=utf-8 python examples/transcribe_file.py 音频.wav [--backend paraformer] [--device cuda]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asr import RealtimeASR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="音频文件路径")
    ap.add_argument("--backend", default="paraformer", help="paraformer|whisper|sherpa")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = ap.parse_args()

    asr = RealtimeASR(backend=args.backend, device=args.device, profile=True)
    t0 = time.time()
    res = asr.ingest_file(args.wav)
    for r in res:
        print("[%.2f-%.2fs] %s  (ttfb=%.3fs)"
              % (r.audio_start, r.audio_end, r.text, r.ttfb))
    print("== 共 %d 句，识别耗时 %.2fs" % (len(res), time.time() - t0))
    asr.close()


if __name__ == "__main__":
    main()
