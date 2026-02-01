@echo off
setlocal

REM Run MFA using the currently active Python environment.
REM Requirement: montreal-forced-aligner must be installed in this environment.

python -m montreal_forced_aligner align %1 %2 %3 %4

endlocal
