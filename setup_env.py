# -*- coding: utf-8 -*-
"""voice1 一键安装脚本 —— ASR + DeepSeek LLM + voice0 TTS 全链路（语音对话）。

给全新用户：已装 conda + git、从未装过本项目。本脚本把两项目装进**同一个环境**
`voice-asr`，因为语音对话程序 `examples/voice_dialogue.py` 在**同一进程**里同时
跑 voice1 的 ASR 和 voice0 的 TTS，必须一个"超集"环境同时装下两边的依赖。

流程（全部幂等，重复执行自动跳过已完成步骤）：
  1  前置检查：conda / git / NVIDIA 驱动（决定 torch 装 cu126 还是 CPU 版）
  2  定位 voice0（约定在 voice1 的**同级目录**）：缺则提示/自动克隆
  3  跑 voice0 的一键脚本（`--skip-cosy`）→ 建 voice-tts 底座（torch + melo +
     锁定依赖 + melo 权重预下载 + 端到端合成验证）——**TTS 侧完全复用 voice0**
  4  克隆 voice-tts → `voice-asr`（两项目共用的唯一环境；复用 torch 不重下）
  5  往 voice-asr 装 ASR 侧依赖（funasr/modelscope/sherpa-onnx/faster-whisper/
     soundfile/requests，版本锁定）
  6  预下载 ASR 权重（preload_asr.py，仅首次联网；之后运行期零网络）
  7  端到端验证：voice-asr 里 ASR + TTS 双端 import 全通，打印用法

用法（任意 python 即可，conda base 也行）：
    python setup_env.py
不硬编码 conda 路径，由 `conda info --base` 动态获取；长时间无输出时每 20s 提示
"仍在进行"，避免误以为卡死。
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV0 = "voice-tts"                                   # voice0 脚本产出的底座环境
ENV1 = "voice-asr"                                   # 两项目共用的唯一环境（本脚本的产物）
VOICE0_DIR = os.path.join(os.path.dirname(ROOT), "voice0")   # voice0 约定在 voice1 同级目录
VOICE0_URL = "https://github.com/celestezj/voice0"
PY_VER = "3.10"

# ASR 侧依赖（voice0 的 voice-tts 底座不含这些；版本锁定自 docs/environment-voice1.md）
ASR_DEPS = ["funasr==1.4.4", "modelscope==1.39.1", "faster_whisper==1.2.1",
            "ctranslate2==4.8.1", "sherpa_onnx==1.13.6", "onnxruntime==1.23.2",
            "soundfile==0.14.0", "requests"]

_STEPS = []


# ---------------------------------------------------------------------------
# 子进程执行：中文输出强制 UTF-8（Windows GBK 会崩）+ HF 镜像兜底
# ---------------------------------------------------------------------------
def _env():
    e = dict(os.environ)
    e.setdefault("PYTHONIOENCODING", "utf-8")
    e.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return e


def run(args, cwd=ROOT, capture=False):
    cmd = args if isinstance(args, (list, tuple)) else args.split()
    if capture:
        return subprocess.run(cmd, cwd=cwd, env=_env(), capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
    return subprocess.run(cmd, cwd=cwd, env=_env())


def run_live(args, label, cwd=ROOT, heartbeat=20.0):
    """流式跑网络/耗时命令；期间每 heartbeat 秒打一条"仍在进行"提示。"""
    stop = threading.Event()
    t0 = time.time()

    def beat():
        while not stop.wait(heartbeat):
            sys.stderr.write("\n  ⏳ %s：仍在进行中（已用时 %d 秒），请耐心等待…\n"
                             % (label, int(time.time() - t0)))
            sys.stderr.flush()

    th = threading.Thread(target=beat, daemon=True)
    th.start()
    try:
        return run(args, cwd=cwd, capture=False)
    finally:
        stop.set()
        th.join(timeout=1)


def die(msg):
    print("\n[错误] %s" % msg)
    sys.exit(1)


def run_ok_else(p, msg):
    if p.returncode != 0:
        die(msg)


# ---------------------------------------------------------------------------
# 环境检测（conda 动态定位，不硬编码路径）
# ---------------------------------------------------------------------------
_BASE = None


def conda_base():
    global _BASE
    if _BASE:
        return _BASE
    p = run(["conda", "info", "--base"], capture=True)
    if p.returncode == 0 and p.stdout.strip():
        _BASE = p.stdout.strip()
        return _BASE
    p2 = subprocess.run("conda info --base", shell=True, capture_output=True,
                        text=True, env=_env())
    if p2.returncode == 0 and p2.stdout.strip():
        _BASE = p2.stdout.strip()
        return _BASE
    return None


def env_exists(name):
    base = conda_base()
    return bool(base) and os.path.isdir(os.path.join(base, "envs", name))


def env_python(name):
    """指定环境的 python 绝对路径；环境不存在时返回 None。"""
    base = conda_base()
    if not base:
        return None
    ed = os.path.join(base, "envs", name)
    cand = os.path.join(ed, "python.exe") if os.name == "nt" else os.path.join(ed, "bin", "python")
    return cand if os.path.isfile(cand) else None


def env_py_cmd(name, args):
    """在指定环境里跑 python 的命令列表（args 不含 python 本身）。"""
    py = env_python(name)
    if py:
        return [py] + args
    return ["conda", "run", "-n", name, "--no-capture-output", "python"] + args


def env_py(name, args, capture=False):
    """在指定环境里跑 python，返回 CompletedProcess。"""
    return run(env_py_cmd(name, args), capture=capture)


# ---------------------------------------------------------------------------
# 显卡检测：驱动支持 CUDA≥12.6 → 装 cu126 torch；否则 CPU 版（melo/ASR 照常）
# ---------------------------------------------------------------------------
def nvidia_info():
    driver = cuda = None
    p = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture=True)
    if p.returncode == 0 and p.stdout.strip():
        driver = p.stdout.strip().splitlines()[0].strip()
    h = run(["nvidia-smi"], capture=True).stdout
    m = re.search(r"CUDA\s*(?:Version|版本)[\s:：]*(\d+\.\d+)", h)
    if m:
        cuda = float(m.group(1))
    return driver, cuda


def nvidia_cuda_ok(driver, cuda):
    if cuda is not None:
        return cuda >= 12.6
    if driver:
        try:
            return float(driver.split()[0]) >= (561.33 if os.name == "nt" else 560.28)
        except ValueError:
            return False
    return False


def gpu_usable():
    driver, cuda = nvidia_info()
    return driver is not None and nvidia_cuda_ok(driver, cuda)


def _nv_desc(driver, cuda):
    if driver is None:
        return "未检测到（无 nvidia-smi）"
    return "驱动 %s（支持 CUDA %s）" % (driver, ("%.1f" % cuda) if cuda else "未知")


def ghfast(url):
    return url.replace("https://github.com/", "https://ghfast.top/https://github.com/")


def git_clone(url, dst):
    """克隆并返回是否成功；直连失败自动换 ghfast.top 镜像重试一次。"""
    args = ["git", "clone", url, dst]
    if run(args).returncode == 0:
        return True
    print("  直连失败，改用 ghfast.top 镜像重试...")
    return run(["git", "clone", ghfast(url), dst]).returncode == 0


# ---------------------------------------------------------------------------
# 幂等检测：先查"已完成"再决定跳不跳
# ---------------------------------------------------------------------------
def voice0_setup_present():
    return os.path.isfile(os.path.join(VOICE0_DIR, "setup_env.py"))


def asr_deps_pinned():
    """ASR 侧依赖是否已按版本锁定（importlib.metadata 查版本，不依赖 __version__ 属性）。"""
    code = (
        "from importlib.metadata import version,sys;"
        "want={'funasr':'1.4.4','modelscope':'1.39.1','faster_whisper':'1.2.1',"
        "'ctranslate2':'4.8.1','sherpa_onnx':'1.13.6','onnxruntime':'1.23.2',"
        "'soundfile':'0.14.0'};"
        "ok=all(version(k)==v for k,v in want.items());"
        "sys.exit(0 if ok else 1)"
    )
    p = env_py(ENV1, ["-c", code], capture=True)
    return p.returncode == 0


def melo_importable():
    """voice-asr 里能 import melo（证明 voice0 的 TTS 也在这个共享环境里可用）。"""
    return env_py(ENV1, ["-c", "import melo"], capture=True).returncode == 0


# ---------------------------------------------------------------------------
# 安装步骤（状态机）
# ---------------------------------------------------------------------------
def step(name):
    def deco(fn):
        _STEPS.append((name, fn))
        return fn
    return deco


def ask_yesno(prompt):
    while True:
        try:
            ans = input(prompt + " [y/N] ").strip().lower()
        except EOFError:
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("", "n", "no"):
            return False
        print("  请输入 y 或 n。")


@step("前置检查（conda / git / 显卡 / 驱动）")
def s1():
    if not conda_base():
        die("未找到 conda。请先安装 Miniconda/Anaconda，再从 Anaconda Prompt 运行本脚本。")
    if not shutil.which("git"):
        die("未找到 git。请先安装 Git for Windows（默认安装选项即可）。")
    print("  conda: %s" % conda_base())
    print("  git:   %s" % shutil.which("git"))
    driver, cuda = nvidia_info()
    print("  NVIDIA: %s" % _nv_desc(driver, cuda))
    if driver is None:
        print("          → 未检测到 nvidia-smi：装 CPU 版 torch（ASR/TTS 可跑，较慢）")
    elif nvidia_cuda_ok(driver, cuda):
        print("          → 驱动支持 CUDA≥12.6：装 cu126 版 torch（GPU 加速可用）")
    else:
        print("          → 驱动不支持 CUDA 12.6：装 CPU 版 torch；想用 GPU 请升级驱动")


@step("定位 voice0（约定在 voice1 同级目录）")
def s2():
    if voice0_setup_present():
        print("  voice0 已就位：%s" % VOICE0_DIR)
        return
    print("  语音对话程序 `examples/voice_dialogue.py` 从 voice1 的**同级目录**读取 voice0（TTS 侧），")
    print("  当前解析位置：%s" % VOICE0_DIR)
    if not ask_yesno("未找到 voice0，是否自动克隆 https://github.com/celestezj/voice0 到这里？"):
        die("请手动把 voice0 克隆到 voice1 的同级目录（%s）后重新运行本脚本。" % VOICE0_DIR)
    if os.path.isdir(VOICE0_DIR) and not voice0_setup_present():
        print("  %s 存在但不是完整的 voice0（缺 setup_env.py），删除后重克隆..." % VOICE0_DIR)
        shutil.rmtree(VOICE0_DIR)
    if not git_clone(VOICE0_URL, VOICE0_DIR):
        die("voice0 克隆失败（直连与 ghfast.top 镜像都失败）。请检查网络后重跑。")


@step("跑 voice0 一键脚本（--skip-cosy）→ 建 %s 底座" % ENV0)
def s3():
    """TTS 侧完全复用 voice0：torch + melo + 锁定依赖 + melo 权重预下载 + 合成验证。"""
    if env_exists(ENV0):
        print("  环境 %s 已存在，跳过。若想补装 cosy：python %s/setup_env.py" % (
            ENV0, VOICE0_DIR))
        return
    p = run_live([sys.executable, os.path.join(VOICE0_DIR, "setup_env.py"), "--skip-cosy"],
                 "voice0 一键安装（首次约 10-30 分钟，含 torch+melo 权重）",
                 cwd=VOICE0_DIR)
    if p.returncode != 0:
        die("voice0 一键安装失败（TTS 底座没建好，voice-asr 无法继续）。请查看上方输出后重跑。")


@step("克隆 %s → %s（两项目共用的唯一环境）" % (ENV0, ENV1))
def s4():
    if env_exists(ENV1):
        print("  环境 %s 已存在，跳过。" % ENV1)
        return
    if not env_exists(ENV0):
        die("前置步骤应已建好 %s，但当前不存在。请重跑本脚本。" % ENV0)
    run_ok_else(run_live(["conda", "create", "-n", ENV1, "--clone", ENV0, "-y"],
                         "克隆环境 %s → %s" % (ENV0, ENV1)),
                "%s 克隆失败（%s → %s）。" % (ENV1, ENV0, ENV1))


@step("往 %s 装 ASR 侧依赖（版本锁定）" % ENV1)
def s5():
    if asr_deps_pinned():
        print("  ASR 依赖已按版本锁定，跳过。")
        return
    print("  安装：%s" % " ".join(ASR_DEPS))
    run_ok_else(run_live(env_py_cmd(ENV1, ["-m", "pip", "install"] + ASR_DEPS),
                         "安装 ASR 依赖"),
                "ASR 依赖安装失败（请确认网络/版本兼容）。")


@step("预下载 ASR 权重（%s/preload_asr.py，仅首次联网）" % ENV1)
def s6():
    p = run_live(env_py_cmd(ENV1, [os.path.join(ROOT, "preload_asr.py"), "--device", "cpu"]),
                 "预下载 ASR 权重（权重已缓存则秒过）")
    if p.returncode != 0:
        die("ASR 权重预下载失败（检查网络）。失败后端见上方 FAIL 列表。")


@step("端到端验证（%s 里 ASR + TTS 双端 import）" % ENV1)
def s7():
    code = (
        "import funasr,sherpa_onnx,sounddevice,requests,json;"
        "from importlib.metadata import version;"
        "print('  ASR 侧 OK: funasr %s / sherpa_onnx %s / sounddevice %s' % ("
        "version('funasr'),version('sherpa_onnx'),version('sounddevice')))"
    )
    if env_py(ENV1, ["-c", code], capture=False).returncode != 0:
        die("ASR 侧 import 验证未通过（%s 环境异常）。请查看上方输出。" % ENV1)
    if not melo_importable():
        print("  ⚠ voice-asr 里 import melo 失败——voice0 的 TTS 可能未在此环境装好，"
              "语音对话的 TTS 侧会报错。建议重跑 voice0 的 setup_env.py。")
    else:
        print("  TTS 侧 OK: melo 可在 %s 里 import（voice0/voice1 共享环境验证通过）" % ENV1)


BANNER = r"""
============================================================
  voice1 一键安装 —— 语音对话全链路（voice1 ASR + voice0 TTS）
  - 复用 voice0 一键脚本做 TTS 底座（不重复装 torch/melo）
  - voice0 / voice1 共用同一个环境 voice-asr（语音对话同时用两边）
  - 重复执行自动跳过已完成步骤；长时间无输出脚本会提示"仍在进行"
============================================================
"""


def main():
    print(BANNER)
    for i, (name, fn) in enumerate(_STEPS, 1):
        print("\n[%d/%d] %s" % (i, len(_STEPS), name))
        print("-" * 72)
        fn()
    print("\n" + "=" * 72)
    print("安装完成！voice0 / voice1 共用环境：")
    print("  conda activate voice-asr")
    print("  # voice1：麦克风实时识别（GPU）")
    print("  PYTHONIOENCODING=utf-8 python examples/record_mic.py --device cuda")
    print("  # 语音对话（voice1 ASR + DeepSeek LLM + voice0 TTS，GPU）")
    print("  PYTHONIOENCODING=utf-8 python examples/voice_dialogue.py --asr-device cuda "
          "--tts-device cuda --vad-tail 300 --system-prompt dialogue/user_prompt.txt "
          "--tts-normalize agc")
    print("  # voice0：TTS demo（同一环境直接跑）")
    print("  cd %s && python speak_example.py" % VOICE0_DIR)


if __name__ == "__main__":
    main()
