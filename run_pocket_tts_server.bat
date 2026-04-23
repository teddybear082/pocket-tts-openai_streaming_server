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
set "PORT=49112"
set /p "INPUT_PORT=Port [%PORT%]: "
if not "%INPUT_PORT%"=="" set "PORT=%INPUT_PORT%"

:: 4. Model Path
set "MODEL_PATH="
set /p "INPUT_MODEL=Model Config Path (.yaml)/Variant Name (Optional, default=built-in): "
if not "%INPUT_MODEL%"=="" set "MODEL_PATH=--model-path ^"%INPUT_MODEL%^""

:: 5. Voices Directory
set "DEFAULT_VOICES=%~dp0voices"
set /p "INPUT_VOICES=Voices Directory [%DEFAULT_VOICES%]: "

if "!INPUT_VOICES!"=="" (
    set "VOICES_DIR_ARG=--voices-dir "!DEFAULT_VOICES!""
) else (
    set "VOICES_DIR_ARG=--voices-dir "!INPUT_VOICES!""
)

:: 6. Streaming Default
:: Changed: Defaults to ON. Only unsets if the user types 'N'.
set "STREAM_ARG=--stream"
set /p "INPUT_STREAM=Enable Streaming? (Y/N) [Y]: "
if /i "%INPUT_STREAM%"=="N" set "STREAM_ARG="

:: 7. Text Preprocessing Default
:: Defaults to ON. Only unsets if the user types 'N'.
set "TEXT_PREPROCESS_ARG=--text-preprocess"
set /p "INPUT_PREPROCESS=Enable Text Preprocessing? (Y/N) [Y]: "
if /i "%INPUT_PREPROCESS%"=="N" set "TEXT_PREPROCESS_ARG="

:: 8. Language
set "LANGUAGE_ARG="
set /p "INPUT_LANGUAGE=Language (english, french_24l, german_24l, portuguese, italian, spanish_24l - leave blank for default): "
if not "%INPUT_LANGUAGE%"=="" set "LANGUAGE_ARG=--language %INPUT_LANGUAGE%"

:: 9. Quantization
set "QUANTIZE_ARG="
set /p "INPUT_QUANTIZE=Enable int8 Quantization? (Y/N) [N]: "
if /i "%INPUT_QUANTIZE%"=="Y" set "QUANTIZE_ARG=--quantize"

echo.
echo ========================================================
echo Starting Pocket TTS Server...
echo Host: %HOST%
echo Port: %PORT%
if defined MODEL_PATH echo Model: %MODEL_PATH%
if defined VOICES_DIR echo Voices: %VOICES_DIR%
if defined STREAM_ARG echo Streaming: Enabled
if defined TEXT_PREPROCESS_ARG echo Text Preprocessing: Enabled
if defined LANGUAGE_ARG echo Language: %INPUT_LANGUAGE%
if defined QUANTIZE_ARG echo Quantization: Enabled
echo ========================================================
echo.

:: 10. Run Command
python server.py --host %HOST% --port %PORT% %MODEL_PATH% %VOICES_DIR_ARG% %STREAM_ARG% %TEXT_PREPROCESS_ARG% %LANGUAGE_ARG% %QUANTIZE_ARG%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server exited with error code %ERRORLEVEL%.
    pause
)
