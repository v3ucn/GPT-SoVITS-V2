SET FFMPEG_PATH=%cd%\runtime\ffmpeg\bin
SET PATH=%FFMPEG_PATH%;%PATH%
runtime\python.exe GPT_SoVITS/inference_webui_full.py zh_CN
pause
