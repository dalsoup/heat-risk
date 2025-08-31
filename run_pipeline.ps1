# run_pipeline.ps1
# 실행: powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
param(
  [switch]$Retrain = $false  # -Retrain 넣으면 6단계까지 수행
)

$ErrorActionPreference = "Stop"

# --- 경로/환경 ---
$proj = "C:\Users\21137\OneDrive\Desktop\heat-risk"   # 프로젝트 루트
$py   = "C:\Users\21137\AppData\Local\Programs\Python\Python312\python.exe"  # 또는 venv의 python
$logDir = Join-Path $proj "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "run_$ts.log"

# 로깅 헬퍼
function Log($msg) { $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"; "$stamp  $msg" | Tee-Object -FilePath $log -Append }

Push-Location $proj
try {
  Log "[START] heat-risk hourly pipeline"

  Log "1) fetch_weather.py"
  & $py fetch_weather.py            2>&1 | Tee-Object -FilePath $log -Append

  Log "2) build_features_hourly.py"
  & $py build_features_hourly.py    2>&1 | Tee-Object -FilePath $log -Append

  Log "3) (mock) build_mock_labels.py"
  & $py build_mock_labels.py        2>&1 | Tee-Object -FilePath $log -Append

  Log "4) build_mock_individual_streams.py"
  & $py build_mock_individual_streams.py   2>&1 | Tee-Object -FilePath $log -Append

  Log "5) build_personal_feature_table.py"
  & $py build_personal_feature_table.py    2>&1 | Tee-Object -FilePath $log -Append

  if ($Retrain) {
    Log "6) train_personal_risk.py"
    # (원하면 환경변수도 여기서 전달 가능)
    & $py train_personal_risk.py     2>&1 | Tee-Object -FilePath $log -Append
  } else {
    Log "skip 6) retrain (Retrain switch not set)"
  }

  Log "[DONE]"
}
catch {
  Log "[ERROR] $($_.Exception.Message)"
  exit 1
}
finally {
  Pop-Location
}
