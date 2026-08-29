# voice-asr 环境锁定（版本快照 · 2026-08-27）

从 `voice-tts`（voice0）克隆，独立重装冲突包，**不干扰 voice0 的 voice-tts 环境**。
克隆后关键包版本见下；如遇冲突在 voice-asr 内单独重装。

## 关键包版本

| 包 | 版本 | 角色 |
|---|---|---|
| torch / torchaudio | 2.11.0+cu126 | CUDA 12.6，GPU 推理 |
| numpy | 2.2.6 | 音频缓冲 |
| scipy | 1.15.3 | `resample_poly` 重采样 |
| funasr | 1.4.4 | paraformer 主力后端 |
| modelscope | 1.39.1 | FunASR 模型源 |
| faster_whisper | 1.2.1 | whisper 可选项 |
| ctranslate2 | 4.8.1 | faster-whisper 运行时 |
| sherpa_onnx | 1.13.6 | CPU 轻量基线 |
| onnxruntime | 1.23.2 | sherpa 依赖 |
| huggingface_hub | 0.36.2 | whisper 模型源 |
| soundfile | 0.14.0 | 非 WAV（mp3/flac/ogg）解码（`read_audio`） |

## 从零复现

> **一键安装**：`python setup_env.py`（仓库根）自动完成下面 1-3 步 + 端到端验证——
> 复用 voice0 一键脚本建 voice-tts 底座，克隆出 **voice0/voice1 共用的唯一环境 voice-asr**，
> 装 ASR 依赖、预下载权重。下面命令行仅为手动分解，正常直接用脚本即可。

```bash
# 1) 一键安装（推荐）
python setup_env.py
conda activate voice-asr

# 2) 权重预下载（脚本已做；仅首次联网，之后运行期零网络）
python preload_asr.py

# 3) 跑验收
python bench/bench_asr.py --backend paraformer --device cuda --tag my
```

## 镜像源（Windows/大陆网络）

- **FunASR 模型**：modelscope 直连（模块内 `MODELSCOPE_CACHE` 已指 `.cache/modelscope`）。
- **whisper 权重**：`HF_ENDPOINT=https://hf-mirror.com`（huggingface.co 直连被墙；模块内已设 + `HF_HOME=.cache/hf`）。
- **sherpa 权重**：ghfast.top 代理 GitHub releases（模块内已设，缓存 `.cache/sherpa_models`）。
- **Windows 编码**：跑带中文输出的命令加 `PYTHONIOENCODING=utf-8`（GBK 会崩）。
