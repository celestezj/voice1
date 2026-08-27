# -*- coding: utf-8 -*-
"""打断词旁路代码 case（T12）：用户说「停下」→ 即时作废全部排队识别任务。

演示内容：
  1. `interrupt_words=["停下"]` 启用旁路 KWS（sherpa-onnx zipformer 3.3M int8）。
  2. 连喂 N 句正常语音制造积压 → 中途喂「停下」→ KWS 流式命中 → `interrupt()`
     作废排队任务、触发块丢弃不识别。
  3. 正在识别的句子完成 → `stale=True` 不进普通回调（profile 仍收集）。
  4. 打断后继续喂新句 → 正常识别（新会话恢复）。

用法（中文输出需 UTF-8）：
    PYTHONIOENCODING=utf-8 python examples/demonstrate_interrupt.py \
        句1.wav 句2.wav 句3.wav ... --stop 停下.wav [--device auto] [--slow 0.6]

参数：
    --stop   打断词音频（例如 melo 合成的「停下」）；缺省尝试 examples/assets 无则跳过打断演示
    --slow   识别人为减速（秒/句），模拟繁忙 worker 让积压真实存在；
             默认 0.6s，显式 --slow 0 关闭（快 GPU 下队列可能不积压，演示效果打折）
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from asr import RealtimeASR
from asr.core.audio import read_audio, resample_to

SR = 16000


def load16k(path):
    sr, data = read_audio(path)
    if sr != SR:
        data = resample_to(data, sr, SR)
    data = np.ascontiguousarray(data, dtype=np.float32)
    # 峰值归一化到 0.10（约 -20dB）：VAD 能量门限 -35dB、最短句 250ms，
    # 过小的音频会被当噪声丢弃（T12 实测：s01「你好。」仅 2 帧过阈 → 不识别）。
    # 归一化上限取 0.10 是 T12 实测的"双检都工作"区间：
    #   归一过大（≥0.15）会把静音文件的噪声底放大，KWS 反而漏检「停下」——
    #   KWS 对喂入音频的响度/信噪比敏感（模型按典型麦克风电平训练）。
    peak = np.abs(data).max()
    if peak > 1e-6:
        data = data * (0.10 / peak)
    return data


def silence(sec):
    return np.zeros(int(SR * sec), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+", help="积压句音频（至少 2 个，自动按 260ms 间隔拼接）")
    ap.add_argument("--stop", help="打断词音频（如 melo 合成的「停下」wav）")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--slow", type=float, default=0.6, help="识别人为减速秒/句（模拟繁忙 worker）")
    args = ap.parse_args()

    if len(args.wavs) < 2:
        print("积压句至少给 2 个音频文件，否则没有『排队任务』可作废。")
        return

    # 1) 启用打断词旁路
    asr = RealtimeASR(backend="paraformer", device=args.device,
                      interrupt_words=["停下"], profile=True, debug=False)
    delivered = []
    asr.on_sentence(lambda r: delivered.append(r))

    # 模拟繁忙 worker（可选）：真实场景中断打的价值正是 worker 来不及处理积压
    if args.slow > 0:
        orig = asr._backend.recognize
        asr._backend.recognize = (lambda a, _o=orig, _s=args.slow:
                                  (time.sleep(_s), _o(a))[1])

    # 2) 拼积压流：句1 句2 ...（句间 260ms 静音）
    backlog = [load16k(p) for p in args.wavs]
    parts = []
    for k, s in enumerate(backlog):
        parts.append(s)
        if k < len(backlog) - 1:
            parts.append(silence(0.26))
    stream = np.concatenate(parts)
    n = int(SR * 0.1)
    print("== 积压音频 %.2fs（%d 句）→ 100ms 块喂入（背压限速）"
          % (len(stream) / SR, len(backlog)))

    stop = threading.Event()

    def feeder():
        for i in range(0, len(stream), n):
            if stop.is_set():
                return
            asr.ingest(stream[i:i + n])

    threading.Thread(target=feeder, daemon=True).start()
    time.sleep(1.2)                     # 让积压排到「停下」之前

    # 3) 喂「停下」+ 尾随静音（麦克风持续采样；KWS 需尾音收尾解码），监控打断触发
    if args.stop and os.path.exists(args.stop):
        ting = load16k(args.stop)
        gen0 = asr._gen
        t0 = time.monotonic()
        fired = False
        for seg in [ting] + [silence(0.1)] * 20:
            asr.ingest(seg)             # 背压时阻塞 = 实时流语义
            if asr._gen != gen0:
                fired = True
                stop.set()
                break
        print("== 「停下」命中=%s（打断触发耗时 %.0fms）" % (fired, (time.monotonic() - t0) * 1000))
    else:
        print("== 未提供 --stop 音频，跳过打断演示（仅识别积压）")

    time.sleep(2.0)                     # 等积压消化/作废

    # 4) 汇报
    print("\n== 普通回调收到 %d 条（打断前完成的句子）:" % len(delivered))
    for r in delivered:
        print("   #%d %s (stale=%s)" % (r.idx, r.text, r.stale))
    stale = [r for r in asr._sentences if r.stale]
    print("== profile 中 stale=%d 条（打断时正在识别/含触发词的句子，被抑制不进回调）:"
          % len(stale))
    for r in stale:
        print("   #%d %s (stale=%s)" % (r.idx, r.text, r.stale))
    print("== 积压 %d 句，产出结果 %d 条 → %d 条排队任务被作废"
          % (len(backlog), len(asr._sentences), len(backlog) - len(asr._sentences)))

    # 5) 恢复：打断后继续喂新句
    if args.stop and os.path.exists(args.stop):
        before = len(delivered)
        new_stream = np.concatenate([backlog[0], silence(0.3)])
        for i in range(0, len(new_stream), n):
            asr.ingest(new_stream[i:i + n])
        time.sleep(1.5)
        print("\n== 打断后继续喂新句，回调新增 %d 条（恢复）" % (len(delivered) - before))
        for r in delivered[before:]:
            print("   #%d %s" % (r.idx, r.text))

    asr.close()


if __name__ == "__main__":
    main()
