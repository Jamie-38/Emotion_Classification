@echo off
REM Activate the Conda environment
call E:\MiniConda\Scripts\activate.bat E:\MiniConda\envs\aligner

REM Run the MFA alignment command
python -m montreal_forced_aligner align %1 %2 %3 %4 %5
