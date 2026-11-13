# Windows-side logger during timed runs: CPU effective frequency (% of base 2.9 GHz), CPU load, AC status.
# WSL cannot see host frequency/thermals; this log is the evidence for/against thermal throttling.
# Stops when the stop-file exists.
param([string]$Out, [string]$StopFile, [int]$Interval = 10)
Add-Type -AssemblyName System.Windows.Forms
"time,perf_pct_of_base,est_mhz,cpu_util_pct,ac" | Out-File -Encoding utf8 $Out
while (-not (Test-Path $StopFile)) {
  $c = Get-Counter '\Processor Information(_Total)\% Processor Performance','\Processor Information(_Total)\% Processor Utility' -ErrorAction SilentlyContinue
  $perf = [math]::Round($c.CounterSamples[0].CookedValue, 1)
  $util = [math]::Round($c.CounterSamples[1].CookedValue, 1)
  $ac = [System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus
  "$(Get-Date -Format s),$perf,$([math]::Round(2900*$perf/100)),$util,$ac" | Out-File -Append -Encoding utf8 $Out
  Start-Sleep -Seconds $Interval
}
