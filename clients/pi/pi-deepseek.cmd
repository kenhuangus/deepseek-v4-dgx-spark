@ECHO off
SETLOCAL
SET "PATH=%USERPROFILE%\.pi\agent\bin;%PATH%"

REM 1. Check if dgx-spark-llm endpoint is reachable on localhost port 8888
curl.exe -fsS -m 2 http://localhost:8888/v1/models >NUL 2>&1
IF %ERRORLEVEL% EQU 0 GOTO launch

REM 2. Check if reachable via host alias dgx-spark-llm
curl.exe -fsS -m 2 http://dgx-spark-llm:8888/v1/models >NUL 2>&1
IF %ERRORLEVEL% EQU 0 GOTO launch

ECHO [ERROR] DeepSeek-V4.1-Flash is unreachable on dgx-spark-llm:8888.
ECHO Please verify that llama-server is running on dgx-spark-llm.
EXIT /B 1

:launch
pi --provider dgx-spark-llm --model deepseek-v4.1-flash %*
