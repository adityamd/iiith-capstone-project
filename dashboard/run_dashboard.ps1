$ErrorActionPreference = 'Stop'
$Candidates = @(
    $env:MLDL_PYTHON,
    (Join-Path $env:USERPROFILE 'anaconda3\envs\MLDL\python.exe'),
    (Join-Path $env:USERPROFILE 'miniconda3\envs\MLDL\python.exe'),
    'C:\ProgramData\anaconda3\envs\MLDL\python.exe',
    'C:\ProgramData\miniconda3\envs\MLDL\python.exe'
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

$MldlPython = $Candidates | Select-Object -First 1
if (-not $MldlPython) {
    throw 'Could not locate the MLDL Python environment. Activate MLDL or set MLDL_PYTHON to its python.exe path.'
}
$MldlRoot = Split-Path -Parent $MldlPython
$RepositoryRoot = Split-Path -Parent $PSScriptRoot

Set-Location -LiteralPath $RepositoryRoot
$env:PATH = "$MldlRoot;$MldlRoot\Library\mingw-w64\bin;$MldlRoot\Library\usr\bin;$MldlRoot\Library\bin;$MldlRoot\Scripts;$env:PATH"
& $MldlPython -m streamlit run dashboard\app.py
