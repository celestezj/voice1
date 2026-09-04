#!/usr/bin/env bash
# ============================================================
#  voice1 一键启动语音对话（ASR + DeepSeek LLM + voice0 TTS + live2d）
#
#  前置：
#    - conda 环境 voice-asr（与 voice0 共享，含 torch）
#    - dialogue/config.local.json 里填好 LLM key（或 --llm-config 指定别的文件）
#    - （可选）live2d 桌宠已启动并 --listen（本脚本不负责拉起桌宠）
#
#  跑法：bash start_dialogue.sh
#        或追加参数覆盖默认值：bash start_dialogue.sh --vad-tail 600
# ============================================================
cd "$(dirname "$0")"

# 中文输出必须 UTF-8（Windows 默认 GBK 会让 python 直接崩）
export PYTHONIOENCODING=utf-8

# 当前不在 voice-asr 环境 → 尝试激活（需 conda 在 PATH）
if ! python -c "import sys; sys.exit(0 if 'voice-asr' in sys.executable else 1)" 2>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate voice-asr 2>/dev/null
fi

if ! python -c "import sys; sys.exit(0 if 'voice-asr' in sys.executable else 1)" 2>/dev/null; then
    echo "[错误] 当前不是 voice-asr 环境，也没能自动激活。请先 conda activate voice-asr 再跑。"
    exit 1
fi

echo "[启动] voice-asr 环境 OK，开始语音对话（Ctrl+C 退出）..."
exec python examples/voice_dialogue.py \
    --asr-device cuda --tts-device cuda --vad-tail 300 --vad-threshold-db -42 \
    --system-prompt dialogue/user_prompt.txt --llm-config dialogue/config.local.json \
    --tts-normalize rms --live2d-port 5000 "$@"
