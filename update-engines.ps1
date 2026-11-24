# update-engines.ps1 - 원클릭 엔진 갱신 빌드 (Windows PowerShell 5.1 / PowerShell 7)
# update-engines.sh 와 같은 동작: 순차 빌드 Strata -> Strata-Vision -> Strata-Poly -> Strata-Poly-Vision -> HY-4 -> Inkling
#   MSVC(vcvars64) 환경 자동 활성화, .venv-build(없으면 .venv) 자동 활성화, 로그는 화면 + build-logs/ 에 저장.
#   CMake 캐시의 컴파일러가 바뀌면 기존 빌드를 보존하고 별도 빌드 폴더를 사용한다.
#
# 사용법:
#   .\update-engines.ps1                       # 6개 전부
#   .\update-engines.ps1 -Pull                 # 먼저 git pull --ff-only
#   .\update-engines.ps1 strata poly           # 일부만 (strata | vision | poly | poly-vision | hy4 | inkling), 항상 위 순서
#   .\update-engines.ps1 -NoDeploy             # 빌드만, engine/ 은 건드리지 않음
# 환경 변수(선택): VCVARS, CUDA_ROOT, CUDA_ARCH, JOBS, VISION_CUDA=ON, LLAMA_DIR, VISION_LLAMA_DIR
# 배포 후 BUILD.json 도 strata launch(terminal/setup.py)와 같은 방식으로 갱신한다 (update-build-json.py).
# (더블클릭/CMD 용: update-engines.bat)

[CmdletBinding()]
param(
    [switch]$Pull,
    [switch]$NoDeploy,
    [Parameter(ValueFromRemainingArguments = $true)]
    [ValidateSet('strata', 'vision', 'poly', 'poly-vision', 'hy4', 'inkling')]
    [string[]]$Steps
)

$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}
$Root = $PSScriptRoot
Set-Location $Root
$LogDir = Join-Path $Root 'build-logs'
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$Jobs = if ($env:JOBS) { $env:JOBS } else { [Environment]::ProcessorCount }
$VisionCuda = if ($env:VISION_CUDA) { $env:VISION_CUDA } else { 'OFF' }
$LlamaDir = if ($env:LLAMA_DIR) { $env:LLAMA_DIR } else { Join-Path $Root 'third_party\llama.cpp' }
$VisionLlamaDir = if ($env:VISION_LLAMA_DIR) { $env:VISION_LLAMA_DIR } else { Join-Path $Root 'third_party\llama.cpp-69f201a' }
$VisionSrc = Join-Path $Root 'terminal\tools\vision'   # 0.2.16 이후 레이아웃: 루트에는 tools/vision 이 없다
if (-not $Steps -or $Steps.Count -eq 0) { $Steps = @('strata', 'vision', 'poly', 'poly-vision', 'hy4', 'inkling') }

$Results = New-Object System.Collections.Generic.List[string]
function Banner($t) { Write-Host "`n==== $t ====" -ForegroundColor Cyan }
function Fail($t) { throw $t }
function Fwd($p) { $p -replace '\\', '/' }

function Find-Vcvars {
    if ($env:VCVARS) { return $env:VCVARS }
    $vsw = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vsw) {
        $p = & $vsw -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
        if ($p) {
            $v = Join-Path $p 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path $v) { return $v }
        }
    }
    foreach ($base in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
        $c = Get-ChildItem (Join-Path $base 'Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat') -ErrorAction SilentlyContinue |
            Sort-Object FullName | Select-Object -Last 1
        if ($c) { return $c.FullName }
    }
}

function Import-Msvc {
    $vc = Find-Vcvars
    if (-not $vc -or -not (Test-Path $vc)) { Fail 'vcvars64.bat 를 찾지 못했습니다. $env:VCVARS 로 지정하세요.' }
    Banner "MSVC 환경 활성화: $vc"
    $tmp = [IO.Path]::ChangeExtension([IO.Path]::GetTempFileName(), '.bat')
    Set-Content -Path $tmp -Encoding ASCII -Value @('@echo off', "call `"$vc`" >nul 2>&1", 'if errorlevel 1 exit /b 1', 'set')
    $dump = & cmd.exe /c $tmp
    $code = $LASTEXITCODE
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    if ($code -ne 0) { Fail 'vcvars64.bat 실행 실패' }
    foreach ($line in $dump) {
        $i = $line.IndexOf('=')
        if ($i -gt 0 -and $line[0] -ne '=') { Set-Item -Path "Env:$($line.Substring(0, $i))" -Value $line.Substring($i + 1) }
    }
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if (-not $cl) { Fail 'vcvars 이후에도 cl.exe 가 PATH 에 없습니다' }
    $script:MsvcCompiler = Fwd $cl.Source
    Write-Host "  cl: $($cl.Source)"
}

function Enable-Venv {
    foreach ($v in '.venv-build', '.venv') {
        $s = Join-Path $Root "$v\Scripts"
        if ((Test-Path "$s\cmake.exe") -and (Test-Path "$s\ninja.exe")) {
            Banner "venv 활성화: $v"
            $env:VIRTUAL_ENV = Join-Path $Root $v      # Activate.ps1 과 같은 효과 (실행 정책 영향 없음)
            $env:PATH = "$s;$env:PATH"
            Write-Host "  cmake: $((Get-Command cmake.exe).Source)   ninja: $((Get-Command ninja.exe).Source)"
            return
        }
    }
    Fail '.venv-build / .venv 에 cmake.exe + ninja.exe 가 없습니다 (pip install cmake ninja)'
}

function Find-Cuda {
    $root = if ($env:CUDA_ROOT) { $env:CUDA_ROOT } elseif ($env:CUDA_PATH) { $env:CUDA_PATH } else {
        $d = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*' -Directory -ErrorAction SilentlyContinue |
            Sort-Object { [version]($_.Name.TrimStart('v')) } | Select-Object -Last 1
        if ($d) { $d.FullName }
    }
    if (-not $root -or -not (Test-Path "$root\bin\nvcc.exe")) { Fail 'nvcc.exe 를 찾지 못했습니다. $env:CUDA_ROOT 로 지정하세요.' }
    $script:Nvcc = Fwd "$root\bin\nvcc.exe"
    $script:CudaArch = $env:CUDA_ARCH
    if (-not $script:CudaArch) {
        $caps = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null |
            ForEach-Object { ($_ -replace '[\s.]', '') } | Where-Object { $_ } | Sort-Object -Unique
        $script:CudaArch = if ($caps) { $caps -join ';' } else { 'native' }
    }
    $env:PATH = "$root\bin;$root\bin\x64;$env:PATH"
    Write-Host "  nvcc: $script:Nvcc   CUDA_ARCH: $script:CudaArch   JOBS: $Jobs"
}

function Find-CompilerMismatch($bpath, $compilers) {
    $cache = Join-Path $bpath 'CMakeCache.txt'
    if (-not (Test-Path -LiteralPath $cache -PathType Leaf)) { return }
    foreach ($line in Get-Content -LiteralPath $cache -ErrorAction Stop) {
        if ($line -match '^(CMAKE_(?:C|CXX|CUDA)_COMPILER):[^=]+=(.*)$') {
            $key = $Matches[1]
            $cached = Fwd $Matches[2]
            if ($compilers.Contains($key) -and $cached -ne $compilers[$key]) {
                return "${key}: $cached -> $($compilers[$key])"
            }
        }
    }
}

# Step <이름> <소스> <빌드폴더> <타깃> <산출물> <배포 대상> <version|help> <cmake -D 인자...>
function Step($name, $src, $bdir, $target, $artifact, $dest, $check, [string[]]$defs) {
    $log = Join-Path $LogDir "$Stamp-$name.log"
    $bpath = Join-Path $Root $bdir
    Banner "[$name] 구성 (cmake configure)  ->  $log"
    $compilers = [ordered]@{
        CMAKE_C_COMPILER = $script:MsvcCompiler
        CMAKE_CXX_COMPILER = $script:MsvcCompiler
    }
    if ($defs -contains "-DCMAKE_CUDA_COMPILER=$script:Nvcc") {
        $compilers['CMAKE_CUDA_COMPILER'] = $script:Nvcc
    }
    $mismatch = Find-CompilerMismatch $bpath $compilers
    if ($mismatch) {
        # Keep the old cache and objects at their original paths. Reuse the new tree on later updates.
        $identity = ($compilers.Values -join "`n").ToLowerInvariant()
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            $digest = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($identity))
        } finally {
            $sha.Dispose()
        }
        $hash = [BitConverter]::ToString($digest, 0, 6).Replace('-', '').ToLowerInvariant()
        $bdir = "$bdir-msvc-$hash"
        $bpath = Join-Path $Root $bdir
        "  CMake 캐시의 컴파일러 변경 감지 ($mismatch)" | Tee-Object -FilePath $log | Out-Host
        $conflict = Find-CompilerMismatch $bpath $compilers
        if ($conflict) { Fail "대체 빌드 폴더의 컴파일러가 일치하지 않습니다: $bpath ($conflict)" }
    }
    "  빌드 폴더: $bpath" | Tee-Object -FilePath $log -Append | Out-Host
    $exe = Join-Path $bpath $artifact
    $cfg = @('-S', (Fwd $src), '-B', (Fwd $bpath), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_C_COMPILER=$script:MsvcCompiler", "-DCMAKE_CXX_COMPILER=$script:MsvcCompiler") + $defs
    & cmake @cfg 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
    $ok = ($LASTEXITCODE -eq 0)
    if ($ok) {
        "`n==== [$name] 빌드 (target $target) ====" | Tee-Object -FilePath $log -Append | Out-Host
        & cmake --build (Fwd $bpath) --target $target -j $Jobs 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
        $ok = ($LASTEXITCODE -eq 0)
    }
    if (-not $ok) { $Results.Add("[실패] ${name}: 빌드 실패 (로그: $log)"); return $false }
    if (-not (Test-Path $exe)) { $Results.Add("[실패] ${name}: 산출물 없음 ($exe)"); return $false }

    # 스모크 체크: strata / strata-vision 은 --version, strata-poly 는 --help (종료 코드 2 가 정상)
    $flag = if ($check -eq 'help') { '--help' } else { '--version' }
    $out = & $exe $flag 2>&1 | Out-String
    $rc = $LASTEXITCODE
    $good = if ($check -eq 'help') { $out.Trim() -and $rc -le 2 } else { $rc -eq 0 }
    if (-not $good) { $Results.Add("[실패] ${name}: 스모크 체크 실패 ($exe)"); Write-Host $out; return $false }
    Write-Host "  스모크 체크 OK: $(($out -split "`r?`n" | Select-Object -First 1))"

    if (-not $NoDeploy) {
        $target_path = Join-Path $Root $dest
        New-Item -ItemType Directory -Force -Path (Split-Path $target_path) | Out-Null
        try {
            if (Test-Path $target_path) { Copy-Item $target_path "$target_path.previous" -Force -ErrorAction Stop }
            Copy-Item $exe $target_path -Force -ErrorAction Stop
        } catch {
            $Results.Add("[실패] ${name}: 배포 실패 - $dest 를 사용 중인 서버/프로세스를 종료하세요 (빌드 산출물: $exe)")
            return $false
        }
        # BUILD.json: strata launch(setup.py)가 쓰는 것과 같은 스탬프 (src/vision_src 해시, version, archs ...)
        $what = if ($name -like '*Vision') { 'vision' } else { 'engine' }
        $kind = if ($name -like '*Poly*') { 'poly' } else { 'strata' }
        if ($target -eq 'strata-hy4') { $kind = 'hy4' }
        if ($target -eq 'strata-inkling') { $kind = 'inkling2' }
        $vis = if ($VisionCuda -eq 'ON') { 'gpu' } else { 'cpu' }
        & python (Join-Path $Root 'update-build-json.py') $what --dir (Split-Path $dest) --kind $kind --arch $script:CudaArch --nvcc $script:Nvcc --vision $vis | Out-Host
        if ($LASTEXITCODE -ne 0) {
            $Results.Add("[실패] ${name}: BUILD.json 갱신 실패 ($(Split-Path $dest)\BUILD.json) - 바이너리는 배포됨")
            return $false
        }
        $Results.Add("[성공] ${name}: $dest 갱신 + BUILD.json (빌드: $bdir\$artifact)")
    } else {
        $Results.Add("[성공] ${name}: 빌드 완료 (배포 생략) $bdir\$artifact")
    }
    return $true
}

# ---- 실행 --------------------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Lock = Join-Path $LogDir '.lock'
if (-not [IO.Directory]::CreateDirectory($Lock).Exists) { Fail 'lock 생성 실패' }
# 동시 실행 방지: lock 폴더 안의 pid 파일로 확인
$pidFile = Join-Path $Lock 'pid'
if (Test-Path $pidFile) {
    $old = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($old -and (Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue)) {
        Write-Host "[오류] 다른 update-engines 가 실행 중입니다 (PID $old)." -ForegroundColor Red
        exit 1
    }
}
Set-Content $pidFile $PID

$rc = 0
try {
    if ($Pull) {
        Banner 'git pull --ff-only'
        & git pull --ff-only
        if ($LASTEXITCODE -ne 0) { Fail 'git pull 실패 (병합이 필요하거나 로컬 변경이 충돌). 직접 정리한 뒤 다시 실행하세요.' }
    }
    Write-Host "소스: $(& git rev-parse --short HEAD) ($(& git rev-parse --abbrev-ref HEAD))  단계: $($Steps -join ' ')"

    Import-Msvc
    Enable-Venv
    Banner 'CUDA 감지'; Find-Cuda

    if ($Steps -contains 'poly' -and -not (Test-Path (Join-Path $LlamaDir 'ggml\src\ggml-cuda\template-instances\mmq-instance-q4_0.cu'))) {
        Fail "Poly 용 llama.cpp 가 올바르지 않습니다: $LlamaDir (`$env:LLAMA_DIR 로 지정)"
    }
    if ($Steps -contains 'poly-vision' -and -not (Test-Path (Join-Path $VisionLlamaDir 'CMakeLists.txt'))) {
        Fail "Poly-Vision 용 llama.cpp 가 없습니다: $VisionLlamaDir (`$env:VISION_LLAMA_DIR 로 지정)"
    }

    $cudaDefs = @("-DCMAKE_CUDA_ARCHITECTURES=$script:CudaArch", "-DCMAKE_CUDA_COMPILER=$script:Nvcc")
    $visionDefs = @("-DSTRATA_VISION_CUDA=$VisionCuda", '-DSTRATA_PORTABLE=OFF')
    if ($VisionCuda -eq 'ON') { $visionDefs += $cudaDefs }

    # 한 단계가 실패하면 뒤 단계는 실행하지 않는다 (순차, 실패 시 중단)
    $done = $true
    if ($done -and $Steps -contains 'strata') {
        $done = Step 'Strata' (Join-Path $Root 'engine') 'terminal/build' 'strata' 'strata.exe' 'terminal\engine\strata.exe' 'version' (@('-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_BUILD_TESTS=OFF') + $cudaDefs)
    }
    if ($done -and $Steps -contains 'vision') {
        $done = Step 'Strata-Vision' $VisionSrc 'terminal/build-vision' 'strata-vision' 'bin\strata-vision.exe' 'terminal\engine\strata-vision.exe' 'version' $visionDefs
    }
    if ($done -and $Steps -contains 'poly') {
        $done = Step 'Strata-Poly' (Join-Path $Root 'engine') 'terminal/build-poly' 'strata-poly' 'strata-poly.exe' 'terminal\engine-poly\strata-poly.exe' 'help' (@('-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_BUILD_TESTS=OFF') + $cudaDefs + @('-DSTRATA_BUILD_POLY=ON', "-DSTRATA_GGML_DIR=$(Fwd $LlamaDir)"))
    }
    if ($done -and $Steps -contains 'poly-vision') {
        $done = Step 'Strata-Poly-Vision' $VisionSrc 'terminal/build-vision-poly' 'strata-vision' 'bin\strata-vision.exe' 'terminal\engine-poly\strata-vision.exe' 'version' ($visionDefs + @('-DSTRATA_VISION_POLY=ON', "-DLLAMA_DIR=$(Fwd $VisionLlamaDir)"))
    }
    if ($done -and $Steps -contains 'hy4') {
        $done = Step 'HY-4' (Join-Path $Root 'engine') 'terminal/build-hy4' 'strata-hy4' 'strata-hy4.exe' 'terminal\engine-hy4\strata-hy4.exe' 'version' (@('-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_BUILD_TESTS=OFF', '-DSTRATA_BUILD_HY4=ON', '-DSTRATA_BUILD_INKLING=OFF') + $cudaDefs)
    }
    if ($done -and $Steps -contains 'inkling') {
        $done = Step 'Inkling' (Join-Path $Root 'engine') 'terminal/build-inkling' 'strata-inkling' 'src\inkling2\strata-inkling.exe' 'terminal\engine-inkling2\strata-inkling.exe' 'version' (@('-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_BUILD_TESTS=OFF', '-DSTRATA_BUILD_INKLING=ON', '-DSTRATA_BUILD_INKLING2=ON') + $cudaDefs)
    }
    if ($done) { Banner '모든 빌드 완료' } else { Write-Host "`n빌드가 중단되었습니다. 위 로그를 확인하세요." -ForegroundColor Red; $rc = 1 }
} catch {
    Write-Host "[오류] $($_.Exception.Message)" -ForegroundColor Red
    $rc = 1
} finally {
    Remove-Item $Lock -Recurse -Force -ErrorAction SilentlyContinue
    Banner '요약'
    foreach ($r in $Results) { Write-Host "  $r" }
    Write-Host "  로그 폴더: $LogDir"
}
exit $rc
