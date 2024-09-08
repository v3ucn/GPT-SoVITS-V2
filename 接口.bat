SET FFMPEG_PATH=%cd%\runtime\ffmpeg\bin
SET PATH=%FFMPEG_PATH%;%PATH%
runtime\python.exe api_v2.py
pause
