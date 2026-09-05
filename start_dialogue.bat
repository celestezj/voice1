@echo off
chcp 65001 >nul
title voice1 dialogue
setlocal enabledelayedexpansion

:: 1. script dir = work dir (no hardcoded path)
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
echo [info] work dir: %SCRIPT_DIR%

:: 1b. brain 模式：start_dialogue.bat [llm|agent] [voice_dialogue 额外参数...]
::    默认不传 = llm（现状零改动）；agent = 本地 claude 常驻会话
set "BRAIN=llm"
if /i "%1"=="agent" (
    set "BRAIN=agent"
    shift
) else if /i "%1"=="llm" (
    set "BRAIN=llm"
    shift
)
echo [info] brain: %BRAIN%

:: 收集其余参数原样透传给 voice_dialogue.py
set "EXTRA="
:extra_loop
if "%1"=="" goto extra_done
set "EXTRA=%EXTRA% %1"
shift
goto extra_loop
:extra_done

:: 2. conda env name: 默认 voice-asr，conda 定位后自动探测（优先 voice-asr，缺失回退 voice-tts）
set "CONDA_ENV=voice-asr"

set "CONDA_ROOT="

:: 3. locate conda root (no machine-specific paths)
:: method1: uninstall registry via PowerShell (custom paths, highest priority)
echo [probe1] uninstall registry: find Anaconda/Miniconda...
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-ItemProperty -Path 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*','HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue | Where-Object {$_.DisplayName -match 'Anaconda|Miniconda'} | Select-Object -First 1 -ExpandProperty InstallLocation" 2^>nul') do (
    set "TEST_ROOT=%%i"
    if exist "!TEST_ROOT!\Scripts\activate.bat" (
        set "CONDA_ROOT=!TEST_ROOT!"
        goto conda_find_ok
    )
)

:: method2: Anaconda-specific registry keys
echo [probe2] Anaconda registry keys...
set "REG_LIST=HKCU\Software\Continuum\Anaconda HKLM\Software\Continuum\Anaconda HKLM\Software\WOW6432Node\Continuum\Anaconda"
for %%r in (%REG_LIST%) do (
    for /f "tokens=2,*" %%a in ('reg query "%%r" /v InstallPath 2^>nul ^| findstr InstallPath') do (
        set "TEST_ROOT=%%b"
        if exist "!TEST_ROOT!\Scripts\activate.bat" (
            set "CONDA_ROOT=!TEST_ROOT!"
            goto conda_find_ok
        )
    )
)

:: method3: where conda (needs PATH)
echo [probe3] where conda (PATH)...
for /f "delims=" %%i in ('where conda 2^>nul') do (
    set "TEST_SCRIPT=%%i"
    for %%p in ("!TEST_SCRIPT!\..\..") do (
        set "TEST_ROOT=%%~fp"
        if exist "!TEST_ROOT!\Scripts\activate.bat" (
            set "CONDA_ROOT=!TEST_ROOT!"
            goto conda_find_ok
        )
    )
)

:: method4: fallback common install paths
echo [probe4] common install paths...
for %%p in (
    %USERPROFILE%\anaconda3
    %USERPROFILE%\miniconda3
    C:\ProgramData\Anaconda3
    C:\ProgramData\Miniconda3
    C:\anaconda3
) do (
    if exist "%%p\Scripts\activate.bat" (
        set "CONDA_ROOT=%%p"
        goto conda_find_ok
    )
)

:: all probes failed: ask user for path (empty = quit)
echo.
echo ########################################################
echo # All probes failed!
echo # Enter Anaconda root (e.g. D:naconda), press Enter
echo # empty + Enter to quit
echo ########################################################
set /p "CONDA_ROOT=Anaconda root:"
if "!CONDA_ROOT!"=="" (
    echo User cancelled, exiting
    pause
    exit /b 1
)
if not exist "!CONDA_ROOT!\Scripts\activate.bat" (
    echo [error] Scriptsctivate.bat not found
    pause
    exit /b 1
)

:conda_find_ok
echo [ok] Anaconda root: !CONDA_ROOT!
set "CONDA_ACTIVATE=!CONDA_ROOT!\Scripts\activate.bat"

:: 3. conda env：**优先 voice-asr**（voice-tts 的严格超集，含全部 ASR + agent 依赖如
::    claude_agent_sdk）；缺失才用 voice-tts（共享 voice0 基座，缺依赖就地补装）；
::    绝不新建 voice-asr。CONDA_ROOT 已定位，直接探测 python.exe 是否存在。
if exist "!CONDA_ROOT!\envs\voice-asr\python.exe" (
    set "CONDA_ENV=voice-asr"
) else (
    set "CONDA_ENV=voice-tts"
)
echo [info] conda env: !CONDA_ENV!

:: 4. activate env
call "!CONDA_ACTIVATE!" "!CONDA_ROOT!"
call conda activate "!CONDA_ENV!"

:: python output needs UTF-8 (default GBK would crash it)
set "PYTHONIOENCODING=utf-8"

:: verify we are really in the target conda env (warn if not)
python -c "import sys; sys.exit(0 if '%CONDA_ENV%' in sys.executable else 1)" >nul 2>nul
if errorlevel 1 (
    echo [error] python not in %CONDA_ENV%, check env name / conda
    pause
    exit /b 1
)

echo.
echo ==============================================
echo  Environment ready, starting voice dialogue
echo  Exit: Ctrl + C
echo ==============================================
echo.

:: 5. brain 模式相关参数：llm 走本地配置；agent 有本地历史则续上次会话
set "LLMCFG=--llm-config dialogue\config.local.json"
set "AGENT_RESUME="
if "%BRAIN%"=="agent" (
    set "LLMCFG="
    if exist "sessions\agent_session_id.txt" set "AGENT_RESUME=--agent-resume"
    if defined AGENT_RESUME (
        echo [info] agent: 检测到本地 claude 历史，续上次会话
    ) else (
        echo [info] agent: 无本地会话历史，将新建会话
    )
)

set "CMD=python examples\voice_dialogue.py --asr-device cuda --tts-device cuda --vad-tail 300 --vad-threshold-db -42 --system-prompt dialogue\user_prompt.txt %LLMCFG% --tts-normalize rms --live2d-port 5000 --brain %BRAIN% %AGENT_RESUME% %EXTRA%"
echo [run] %CMD%
%CMD%

echo.
echo -------- dialogue ended, press any key to close --------
pause
