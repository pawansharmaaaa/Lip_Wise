@echo off

REM Check if ffmpeg is installed
where ffmpeg >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ffmpeg is not installed. Checking for winget...

    REM Check if winget is installed
    where winget >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo winget is not installed. Please install winget or manually install ffmpeg.
        exit /b
    )

    REM Install FFMPEG
    winget install ffmpeg
)

REM Check if CUDA is installed
if not defined CUDA_PATH (
    echo CUDA is not installed. Please install CUDA.
    echo CPU will be used for inference.
)

REM Create a virtual environment
python -m venv .lip-wise

REM Activate the virtual environment
call .lip-wise\Scripts\activate

REM Install requirements
pip install -r requirements.txt

REM Copy basicsr archs
copy archs\* .lip-wise\Lib\site-packages\basicsr\archs