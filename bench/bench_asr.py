# -*- coding: utf-8 -*-
"""bench_asr：内容级验收（voice1 硬指标裁判）。

对 (backend, device) 组合跑全部语料，产出并对照硬指标：
  - 严格 CER（仅去标点）         硬指标：CER<5%
  - 规范 CER（去标点+繁简统一+中文/阿拉伯数字归一）——形态差异归因辅助
  - RTF = 识别墙钟 / 音频总时长   硬指标：RTF<0.3(GPU) / <1(CPU)
  - 平均尾字延迟 ttfb           硬指标：<0.5s

结果写 UTF-8 报告 `reports/bench_{tag}.txt`，打印摘要表。
用法：
  python bench/bench_asr.py                       # 全部组合
  python bench/bench_asr.py --backend paraformer --device cuda --tag T9
"""
import argparse
import json
import os
import re
import sys
import time
import wave

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from asr import RealtimeASR

CORPUS = os.path.join(_ROOT, "assets", "corpus")
REPORTS = os.path.join(_ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)

# 硬指标（ADR 立项定案，口径写死）
HARD = dict(cer=0.05, rtf_gpu=0.3, rtf_cpu=1.0, ttfb=0.5)

_PUNC = re.compile(r"[，。！？、；：,.\"'“”‘’（）()《》<>：:·…—\-]+")


def _zh_num(s):
    _D = "零一二三四五六七八九"
    return "".join(_D[int(c)] for c in s)


def _norm_nums(s):
    s = re.sub(r"(\d+)\s*%", lambda m: "百分之" + _zh_num(m.group(1)), s)
    return re.sub(r"\d+", lambda m: _zh_num(m.group(0)), s)


# 常用简繁映射（覆盖语料实测出现的字；完整表 T11 若需可换 opencc）
_T2S = str.maketrans(
    "國我們妳怎麼點臺獨數裏爲說後時樣師習辦公務導戰勝軟體檔案網頁這是來開萬與兩邊間風關體學歷級練員約離後現",
    "国我们你怎么点台独数里为说后时样师习办公务导战胜软件档案网页这是来开万与两边间风关体学历级练员约离后现",
)


def cer_strict(ref, hyp):
    return _cer(_PUNC.sub("", ref), _PUNC.sub("", hyp))


def cer_norm(ref, hyp):
    def n(s):
        s = _PUNC.sub("", s)
        s = _norm_nums(s)
        return s.translate(_T2S)
    return _cer(n(ref), n(hyp))


def _cer(ref, hyp):
    import difflib
    sm = difflib.SequenceMatcher(None, ref, hyp)
    dist = max(len(ref), len(hyp)) - sum(b.size for b in sm.get_matching_blocks())
    return dist / max(len(ref), 1)


def wav_dur(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def run(backend, device, tag, tail=250):
    with open(os.path.join(CORPUS, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    t_load = time.time()
    asr = RealtimeASR(backend=backend, device=device, vad_silence_tail_ms=tail, profile=True)
    load_s = time.time() - t_load

    rows, total_dur, total_wall = [], 0.0, 0.0
    ttfb_all, cs, cn, cnt = [], 0.0, 0.0, 0
    for entry in manifest:
        ref, wav = entry["text"], os.path.join(_ROOT, entry["wav"])
        dur = wav_dur(wav)
        t0 = time.time()
        res = asr.ingest_file(wav)
        dt = time.time() - t0
        hyp = "".join(r.text for r in res)
        s_cer, n_cer = cer_strict(ref, hyp), cer_norm(ref, hyp)
        ttfb_all.extend(r.ttfb for r in res)
        total_dur += dur
        total_wall += dt
        cs += s_cer
        cn += n_cer
        cnt += 1
        rows.append((os.path.basename(wav), dur, dt, s_cer, n_cer, ref, hyp))
    asr.close()

    n = max(cnt, 1)
    avg_ttfb = (sum(ttfb_all) / len(ttfb_all)) if ttfb_all else 0.0
    rtf = total_wall / total_dur if total_dur else 0.0
    return dict(backend=backend, device=device, load_s=load_s, tail=tail,
                strict_cer=cs / n, norm_cer=cn / n, rtf=rtf, ttfb=avg_ttfb,
                total_dur=total_dur, total_wall=total_wall, rows=rows)


def report(res, tag):
    ok_cer = res["strict_cer"] < HARD["cer"]
    ok_rtf = res["rtf"] < (HARD["rtf_gpu"] if res["device"].startswith("cuda") else HARD["rtf_cpu"])
    ok_ttfb = res["ttfb"] < HARD["ttfb"]
    lines = []
    lines.append("=== bench %s | %s/%s | VAD tail=%dms | 加载 %.1fs | 音频 %.1fs / 墙钟 %.1fs ==="
                 % (tag, res["backend"], res["device"], res["tail"], res["load_s"],
                    res["total_dur"], res["total_wall"]))
    lines.append("硬指标: 严格CER<%.2f %s | RTF<%.2f(%s) %s | ttfb<%.2f %s"
                 % (HARD["cer"], "达标" if ok_cer else "未达",
                    HARD["rtf_gpu"] if res["device"].startswith("cuda") else HARD["rtf_cpu"],
                    res["device"], "达标" if ok_rtf else "未达",
                    HARD["ttfb"], "达标" if ok_ttfb else "未达"))
    lines.append("严格CER=%.3f | 规范CER=%.3f | RTF=%.3f | 平均ttfb=%.3fs"
                 % (res["strict_cer"], res["norm_cer"], res["rtf"], res["ttfb"]))
    lines.append("")
    lines.append("%-8s %6s %7s %7s %7s  ref" % ("wav", "dur", "wall", "sCER", "nCER"))
    for name, dur, dt, sc, nc, ref, hyp in res["rows"]:
        lines.append("%-8s %5.1f %6.2f %7.3f %7.3f  %s" % (name, dur, dt, sc, nc, ref))
        lines.append("          hyp: %s" % hyp)
    path = os.path.join(REPORTS, "bench_%s_%s_%s.txt" % (tag, res["backend"], res["device"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return ok_cer and ok_rtf and ok_ttfb, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="all",
                    choices=["all", "paraformer", "paraformer-offline", "whisper", "whisper-large", "sherpa"])
    ap.add_argument("--device", default="all", choices=["all", "cpu", "cuda"])
    ap.add_argument("--tag", default="T9")
    ap.add_argument("--tail", type=int, default=250, help="VAD silence_tail_ms")
    args = ap.parse_args()

    combos = [("paraformer", "cuda"), ("paraformer", "cpu"),
              ("paraformer-offline", "cuda"), ("paraformer-offline", "cpu"),
              ("whisper", "cuda"), ("whisper", "cpu"),
              ("whisper-large", "cuda"), ("whisper-large", "cpu"),
              ("sherpa", "cpu")]
    if args.backend != "all":
        combos = [c for c in combos if c[0] == args.backend]
    if args.device != "all":
        combos = [c for c in combos if c[1] == args.device]

    summary = []
    for backend, device in combos:
        print("[bench] %s/%s (tail=%d) …" % (backend, device, args.tail), flush=True)
        res = run(backend, device, args.tag, args.tail)
        ok, path = report(res, args.tag)
        summary.append((backend, device, ok, res["strict_cer"], res["rtf"], res["ttfb"], path))
        print("  -> %s" % path, flush=True)

    print("")
    print("=== 汇总（严格CER | RTF | ttfb）===")
    for backend, device, ok, sc, rtf, ttfb, path in summary:
        print("  %-10s %-4s %s | CER=%.3f RTF=%.3f ttfb=%.3fs | %s"
              % (backend, device, "PASS" if ok else "FAIL", sc, rtf, ttfb, path))


if __name__ == "__main__":
    main()
