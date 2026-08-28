# -*- coding: utf-8 -*-
"""示例：命令行转写单个音频文件（wav/mp3/flac/ogg；逐句打印文本 + 时间戳 + 尾字延迟）。

`--streaming`：流式逐块出字（on_partial 边说边出 + 句末 flush 定稿，CER 更优）；
whisper 不支持流式 → 自动降级整句并告警。

用法（中文输出需 UTF-8 编码）：
    PYTHONIOENCODING=utf-8 python examples/transcribe_file.py 音频.mp3 [--backend paraformer] [--device cuda] [--streaming]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asr import RealtimeASR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="音频文件路径（wav/mp3/flac/ogg）")
    ap.add_argument("--backend", default="paraformer",
                    help="paraformer(默认,流式)|paraformer-offline(离线高精度,整句)|"
                         "whisper|whisper-large(large-v3-turbo,同音字最强)|sherpa")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--streaming", action="store_true",
                    help="流式逐块出字（on_partial 边说边出 + 句末 flush 定稿）")
    ap.add_argument("--hotword-file", default=None,
                    help="热词文件路径（每行一个纠错项；引擎级拼音纠错，如 神庙→神妙，"
                         "所有后端统一生效）")
    args = ap.parse_args()

    asr = RealtimeASR(backend=args.backend, device=args.device,
                      streaming=args.streaming, profile=True,
                      hotword_file=args.hotword_file)
    if args.streaming:
        last = [""]
        def _partial(p):
            if p.text and p.text != last[0]:      # 累计文本未变则跳过（避免逐块刷屏）
                last[0] = p.text
                print("[流式出字] %s" % p.text, flush=True)
        asr.on_partial(_partial)
    t0 = time.time()
    res = asr.ingest_file(args.wav)
    for r in res:
        print("[%.2f-%.2fs] %s  (ttfb=%.3fs)"
              % (r.audio_start, r.audio_end, r.text, r.ttfb))
    print("== 共 %d 句，识别耗时 %.2fs" % (len(res), time.time() - t0))
    asr.close()


if __name__ == "__main__":
    main()
