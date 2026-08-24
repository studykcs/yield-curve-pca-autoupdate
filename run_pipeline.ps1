$ErrorActionPreference = "Continue"
Set-Location "C:\Users\swgtl\Desktop\pythonprojectwithclaude"

$python = "C:\Users\swgtl\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$log = "output\pipeline.log"
New-Item -ItemType Directory -Force -Path output | Out-Null

"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -Append -Encoding utf8 $log
& $python collect.py    2>&1 | Out-File -Append -Encoding utf8 $log
& $python anomaly.py    2>&1 | Out-File -Append -Encoding utf8 $log
& $python dashboard.py  2>&1 | Out-File -Append -Encoding utf8 $log
