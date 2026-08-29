# -*- coding: utf-8 -*-
"""冒烟：真实唤醒词 KWS 模型加载 + 静音 feed 零误触发（验证 SLEEP 分支检测器链路）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from asr.kws.interrupt import get_interrupt_detector

det = get_interrupt_detector("sherpa", words=["小爱小爱"])
det.load()
print("[冒烟] 唤醒检测器加载 OK, 词=%s, 关键词文件=%s"
      % (det._words, os.path.basename(det._keywords_file)))
hits = 0
for _ in range(100):                     # 喂 100 块静音（约 2 秒）
    if det.feed(np.zeros(320, dtype=np.float32)):
        hits += 1
assert hits == 0, "静音不应触发唤醒: %d" % hits
det.close()
print("[冒烟] 静音 100 块零误触发 OK")
