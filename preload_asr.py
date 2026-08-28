# -*- coding: utf-8 -*-
"""preload_asr：预下载全部后端权重到项目 `.cache/`（仅首次联网，之后运行期零网络）。

用法：`python preload_asr.py [--device cpu]`
覆盖：
  - paraformer → FunASR paraformer-zh-streaming（modelscope）→ `.cache/modelscope/`
  - whisper    → faster-whisper medium（hf-mirror）→ `.cache/hf/`
  - sherpa     → sherpa-onnx zipformer（ghfast.top）→ `.cache/sherpa_models/`

预下载阶段用 cpu 即可（权重缓存与设备无关）；下载后运行期不再联网。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr.core.backend import get_backend

_MODELS = ["paraformer", "paraformer-offline", "whisper", "whisper-large", "sherpa"]


def main():
    ap = argparse.ArgumentParser(description="预下载 ASR 后端权重")
    ap.add_argument("--device", default="cpu", help="预下载阶段用 cpu 即可（权重缓存与设备无关）")
    args = ap.parse_args()

    ok, fail = [], []
    for name in _MODELS:
        t0 = time.time()
        try:
            b = get_backend(name, device=args.device)
            b.load()
            b.close()
            ok.append(name)
            print("[preload] %-10s OK  (%.1fs)" % (name, time.time() - t0), flush=True)
        except Exception as e:
            fail.append(name)
            print("[preload] %-10s FAIL: %r" % (name, e), flush=True)

    print("完成: OK=%s FAIL=%s" % (ok, fail), flush=True)
    print("缓存: paraformer→.cache/modelscope, whisper→.cache/hf, sherpa→.cache/sherpa_models", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
