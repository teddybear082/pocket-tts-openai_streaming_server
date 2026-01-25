@echo off
setlocal EnableDelayedExpansion

title Pocket TTS Server Launcher

echo.
echo ========================================================
echo        Pocket TTS OpenAI Streaming Server Launcher
echo ========================================================
echo.
:: 0. Hugging Face Authentication
set "HF_TOKEN=your_token_here_if_you_want_to_hardcode_it"
if "%HF_TOKEN%"=="your_token_here_if_you_want_to_hardcode_it" (
    set /p "HF_TOKEN=Enter Hugging Face Token (leave blank if already logged in): "
)

if not "%HF_TOKEN%"=="" (
    echo [INFO] Setting Hugging Face Token...
    set "HF_TOKEN=%HF_TOKEN%"
)

:: 1. Activate Virtual Environment
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] venv not found at .\venv. Attempting to run with system python...
)

echo.
echo Please configure the server (Press ENTER to use defaults):
echo.

:: 2. Host
set "HOST=0.0.0.0"
set /p "INPUT_HOST=Host IP [%HOST%]: "
if not "%INPUT_HOST%"=="" set "HOST=%INPUT_HOST%"

:: 3. Port
set "PORT=5002"
set /p "INPUT_PORT=Port [%PORT%]: "
if not "%INPUT_PORT%"=="" set "PORT=%INPUT_PORT%"

:: 4. Model Path
set "MODEL_PATH="
set /p "INPUT_MODEL=Model Path/Variant (Optional, default=built-in): "
if not "%INPUT_MODEL%"=="" set "MODEL_PATH=--model_path ^"%INPUT_MODEL%^""

:: 5. Voices Directory
set "DEFAULT_VOICES=%~dp0voices"
set /p "INPUT_VOICES=Voices Directory [%DEFAULT_VOICES%]: "

if "!INPUT_VOICES!"=="" (
    set "VOICES_DIR_ARG=--voices_dir "!DEFAULT_VOICES!""
) else (
    set "VOICES_DIR_ARG=--voices_dir "!INPUT_VOICES!""
)

:: 6. Streaming Default
:: Changed: Defaults to ON. Only unsets if the user types 'N'.
set "STREAM_ARG=--stream"
set /p "INPUT_STREAM=Enable Streaming? (Y/N) [Y]: "
if /i "%INPUT_STREAM%"=="N" set "STREAM_ARG="

echo.
echo ========================================================
echo Starting Pocket TTS Server...
echo Host: %HOST%
echo Port: %PORT%
if defined MODEL_PATH echo Model: %MODEL_PATH%
if defined VOICES_DIR echo Voices: %VOICES_DIR%
if defined STREAM_ARG echo Streaming: Enabled
echo ========================================================
echo.

:: 7. Run Command
python pocket_tts_openai_server.py --host %HOST% --port %PORT% %MODEL_PATH% %VOICES_DIR_ARG% %STREAM_ARG%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server exited with error code %ERRORLEVEL%.
    pause
)
