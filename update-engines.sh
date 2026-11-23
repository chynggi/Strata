#!/usr/bin/env bash
# update-engines.sh - 원클릭 엔진 갱신 빌드 (Windows / Git Bash)
#
# 순차 빌드 (병렬 아님):  1) Strata  2) Strata-Vision  3) Strata-Poly  4) Strata-Poly-Vision
#   - MSVC(vcvars64) 환경을 자동으로 불러온다.
#   - .venv-build (없으면 .venv)를 자동으로 활성화한다 (cmake / ninja 사용).
#   - 빌드 로그는 화면에 그대로 출력되고, build-logs/ 에도 저장된다.
#   - 성공한 빌드는 --version/--help 로 확인한 뒤 terminal/engine/ , terminal/engine-poly/ 로 (strata launch 가 만드는 run config 의 경로) 복사한다
#     (기존 파일은 <이름>.previous 로 보관). --no-deploy 로 복사를 건너뛸 수 있다.
#   - 배포 후 BUILD.json 도 strata launch(terminal/setup.py)와 같은 방식으로 갱신한다 (update-build-json.py).
#
# 사용법:
#   ./update-engines.sh                      # 4개 전부, 현재 작업 트리의 소스로 빌드
#   ./update-engines.sh --pull               # 먼저 git pull --ff-only 로 최신 소스를 받는다
#   ./update-engines.sh strata poly          # 일부만 (strata | vision | poly | poly-vision), 항상 위 순서로 실행
#   ./update-engines.sh --no-deploy          # 빌드만 하고 engine/ 은 건드리지 않는다
#
# 환경 변수(선택):
#   VCVARS=<vcvars64.bat 경로>   CUDA_ROOT=<CUDA 툴킷 루트>   CUDA_ARCH="120" (기본: nvidia-smi 자동 감지, 실패 시 native)
#   JOBS=<병렬 컴파일 수, 기본 nproc>   VISION_CUDA=ON (Strata-Vision / Poly-Vision 을 GPU 용으로 빌드, 기본 OFF=CPU)
#   LLAMA_DIR=<Poly 용 llama.cpp>   VISION_LLAMA_DIR=<Poly-Vision 용 llama.cpp>

set -uo pipefail
export MSYS2_ARG_CONV_EXCL='*'      # CMake 인자의 Windows 경로가 MSYS 에 의해 변환되지 않게 한다

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
LOGDIR="$ROOT/build-logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
JOBS="${JOBS:-$(nproc)}"
VISION_CUDA="${VISION_CUDA:-OFF}"
LLAMA_DIR="${LLAMA_DIR:-$ROOT/third_party/llama.cpp}"
VISION_LLAMA_DIR="${VISION_LLAMA_DIR:-$ROOT/third_party/llama.cpp-69f201a}"
VISION_SRC="$ROOT/terminal/tools/vision"     # 0.2.16 이후 레이아웃: 루트에는 tools/vision 이 없다

DO_PULL=0; DEPLOY=1; SELECTED=()
for a in "$@"; do
  case "$a" in
    --pull) DO_PULL=1 ;;
    --no-deploy) DEPLOY=0 ;;
    strata|vision|poly|poly-vision) SELECTED+=("$a") ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "알 수 없는 인자: $a  (--help 참고)"; exit 2 ;;
  esac
done
[ ${#SELECTED[@]} -eq 0 ] && SELECTED=(strata vision poly poly-vision)
want() { local s; for s in "${SELECTED[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

banner() { printf '\n\033[1;36m==== %s ====\033[0m\n' "$*"; }
die()    { printf '\033[1;31m[오류] %s\033[0m\n' "$*" >&2; exit 1; }

# ---- 동시 실행 방지 (순차 실행 보장) ----------------------------------------------------------
mkdir -p "$LOGDIR"
LOCK="$LOGDIR/.lock"
mkdir "$LOCK" 2>/dev/null || die "다른 update-engines.sh 가 실행 중입니다 ($LOCK). 아니라면 그 폴더를 삭제하세요."

declare -a RESULTS=()
summary() {
  rmdir "$LOCK" 2>/dev/null
  banner "요약"
  local r; for r in "${RESULTS[@]:-}"; do [ -n "$r" ] && echo "  $r"; done
  echo "  로그 폴더: $LOGDIR"
}
trap summary EXIT

# ---- 1. MSVC 환경 활성화 -----------------------------------------------------------------------
find_vcvars() {
  [ -n "${VCVARS:-}" ] && { echo "$VCVARS"; return; }
  local vsw="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe" p
  if [ -x "$vsw" ]; then
    p="$("$vsw" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | tr -d '\r' | head -1)"
    [ -n "$p" ] && [ -f "$(cygpath -u "$p")/VC/Auxiliary/Build/vcvars64.bat" ] && { echo "$(cygpath -u "$p")/VC/Auxiliary/Build/vcvars64.bat"; return; }
  fi
  ls -d "/c/Program Files (x86)/Microsoft Visual Studio/"*/*/VC/Auxiliary/Build/vcvars64.bat \
        "/c/Program Files/Microsoft Visual Studio/"*/*/VC/Auxiliary/Build/vcvars64.bat 2>/dev/null | sort -V | tail -1
}

import_msvc() {
  local vc; vc="$(find_vcvars)"
  [ -n "$vc" ] && [ -f "$vc" ] || die "vcvars64.bat 를 찾지 못했습니다. VCVARS=<경로> 로 지정하세요."
  banner "MSVC 환경 활성화: $vc"
  local tmpbat; tmpbat="$(mktemp --suffix=.bat)"
  printf '@echo off\r\ncall "%s" >nul 2>&1\r\nif errorlevel 1 exit /b 1\r\nset\r\n' "$(cygpath -w "$vc")" > "$tmpbat"
  local dump; dump="$(cmd /c "$(cygpath -w "$tmpbat")" | tr -d '\r')" || { rm -f "$tmpbat"; die "vcvars64.bat 실행 실패"; }
  rm -f "$tmpbat"
  local line k v newpath=""
  while IFS= read -r line; do
    k="${line%%=*}"; v="${line#*=}"
    case "$k" in
      Path|PATH) newpath="$(cygpath -up "$v")" ;;
      INCLUDE|LIB|LIBPATH|EXTERNAL_INCLUDE|VC*|VS*|VSCMD_*|WindowsS*|WindowsLibPath|UCRT*|UniversalCRT*|DevEnvDir|Framework*|NETFX*|Platform|CommandPromptType)
        [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$k=$v" ;;
    esac
  done <<< "$dump"
  [ -n "$newpath" ] && export PATH="$newpath:$PATH"      # MSVC 가 Git 의 link.exe 보다 앞에 오도록
  command -v cl >/dev/null || die "vcvars 이후에도 cl.exe 가 PATH 에 없습니다"
  echo "  cl: $(command -v cl)"
}

# ---- 2. venv 활성화 ----------------------------------------------------------------------------
activate_venv() {
  local v
  for v in .venv-build .venv; do
    if [ -f "$ROOT/$v/Scripts/activate" ] && [ -x "$ROOT/$v/Scripts/cmake.exe" ] && [ -x "$ROOT/$v/Scripts/ninja.exe" ]; then
      banner "venv 활성화: $v"
      # shellcheck disable=SC1091
      source "$ROOT/$v/Scripts/activate"
      echo "  cmake: $(command -v cmake)   ninja: $(command -v ninja)"
      return
    fi
  done
  die ".venv-build / .venv 에 cmake.exe + ninja.exe 가 없습니다 (pip install cmake ninja)"
}

# ---- 3. CUDA / GPU 감지 ------------------------------------------------------------------------
detect_cuda() {
  local root="${CUDA_ROOT:-${CUDA_PATH:-}}"
  [ -z "$root" ] && root="$(ls -d "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/"v* 2>/dev/null | sort -V | tail -1)"
  [ -n "$root" ] && root="$(cygpath -u "$root")"
  [ -x "$root/bin/nvcc.exe" ] || die "nvcc.exe 를 찾지 못했습니다. CUDA_ROOT=<툴킷 경로> 로 지정하세요."
  NVCC="$(cygpath -m "$root/bin/nvcc.exe")"
  if [ -z "${CUDA_ARCH:-}" ]; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '\r .' | sort -u | paste -sd';')"
    [ -z "$CUDA_ARCH" ] && CUDA_ARCH="native"
  fi
  echo "  nvcc: $NVCC   CUDA_ARCH: $CUDA_ARCH   JOBS: $JOBS"
}

# ---- 빌드 헬퍼 ---------------------------------------------------------------------------------
# step_build <이름> <소스> <빌드폴더> <타깃> <산출물(빌드폴더 기준)> <배포 대상> <확인 모드> <cmake -D 인자...>
step_build() {
  local name="$1" src="$2" bdir="$3" target="$4" artifact="$5" dest="$6" check="$7"; shift 7
  local log="$LOGDIR/$STAMP-$name.log" exe="$ROOT/$bdir/$artifact"
  banner "[$name] 구성 (cmake configure)  →  $log"
  {
    cmake -S "$(cygpath -m "$src")" -B "$(cygpath -m "$ROOT/$bdir")" -G Ninja -DCMAKE_BUILD_TYPE=Release "$@" \
      && { printf '\n==== [%s] 빌드 (target %s) ====\n' "$name" "$target"; \
           cmake --build "$(cygpath -m "$ROOT/$bdir")" --target "$target" -j "$JOBS"; }
  } 2>&1 | tee "$log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then RESULTS+=("✗ $name: 빌드 실패 (로그: $log)"); return 1; fi
  [ -f "$exe" ] || { RESULTS+=("✗ $name: 산출물 없음 ($exe)"); return 1; }

  # 스모크 체크: strata / strata-vision 은 --version, strata-poly 는 --help (종료 코드 2 가 정상)
  local out rc
  if [ "$check" = "help" ]; then out="$("$exe" --help 2>&1)"; rc=$?; [ -n "$out" ] && [ $rc -le 2 ] || rc=1; [ $rc -le 2 ] && rc=0
  else out="$("$exe" --version 2>&1)"; rc=$?; fi
  if [ $rc -ne 0 ]; then RESULTS+=("✗ $name: 스모크 체크 실패 ($exe)"); echo "$out"; return 1; fi
  echo "  스모크 체크 OK: $(echo "$out" | head -1)"

  if [ "$DEPLOY" -eq 1 ]; then
    local target_path="$ROOT/$dest"
    mkdir -p "$(dirname "$target_path")"
    [ -f "$target_path" ] && cp -f "$target_path" "$target_path.previous"
    if ! cp -f "$exe" "$target_path"; then
      RESULTS+=("✗ $name: 배포 실패 — $dest 를 사용 중인 서버/프로세스를 종료하세요 (빌드 산출물: $exe)"); return 1
    fi
    # BUILD.json: strata launch(setup.py)가 쓰는 것과 같은 스탬프 (src/vision_src 해시, version, archs ...)
    local what=engine kind=strata vis=cpu
    [[ "$name" == *Vision ]] && what=vision; [[ "$name" == *Poly* ]] && kind=poly; [ "$VISION_CUDA" = "ON" ] && vis=gpu
    if ! python "$ROOT/update-build-json.py" "$what" --dir "$(dirname "$dest")" --kind "$kind" --arch "$CUDA_ARCH"          --nvcc "$NVCC" --vision "$vis"; then
      RESULTS+=("✗ $name: BUILD.json 갱신 실패 ($(dirname "$dest")/BUILD.json) — 바이너리는 배포됨"); return 1
    fi
    RESULTS+=("✓ $name: $dest 갱신 + BUILD.json (빌드: $bdir/$artifact)")
  else
    RESULTS+=("✓ $name: 빌드 완료 (배포 생략) $bdir/$artifact")
  fi
}

# ---- 실행 --------------------------------------------------------------------------------------
if [ "$DO_PULL" -eq 1 ]; then
  banner "git pull --ff-only"
  git pull --ff-only || die "git pull 실패 (병합이 필요하거나 로컬 변경이 충돌). 직접 정리한 뒤 다시 실행하세요."
fi
echo "소스: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))  단계: ${SELECTED[*]}"

import_msvc
activate_venv
banner "CUDA 감지"; detect_cuda

if want poly; then
  [ -f "$LLAMA_DIR/ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0.cu" ] \
    || die "Poly 용 llama.cpp 가 올바르지 않습니다: $LLAMA_DIR (LLAMA_DIR 로 지정)"
fi
if want poly-vision; then
  [ -f "$VISION_LLAMA_DIR/CMakeLists.txt" ] || die "Poly-Vision 용 llama.cpp 가 없습니다: $VISION_LLAMA_DIR (VISION_LLAMA_DIR 로 지정)"
fi

CUDA_DEFS=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH" "-DCMAKE_CUDA_COMPILER=$NVCC")
VISION_DEFS=("-DSTRATA_VISION_CUDA=$VISION_CUDA" "-DSTRATA_PORTABLE=OFF")
[ "$VISION_CUDA" = "ON" ] && VISION_DEFS+=("${CUDA_DEFS[@]}")

# 한 단계가 실패하면 뒤 단계는 실행하지 않는다 (순차, 실패 시 중단)
run_all() {
  want strata && { step_build Strata "$ROOT/engine" terminal/build strata strata.exe terminal/engine/strata.exe version \
      -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF "${CUDA_DEFS[@]}" || return 1; }
  want vision && { step_build Strata-Vision "$VISION_SRC" terminal/build-vision strata-vision bin/strata-vision.exe terminal/engine/strata-vision.exe version \
      "${VISION_DEFS[@]}" || return 1; }
  want poly && { step_build Strata-Poly "$ROOT/engine" terminal/build-poly strata-poly strata-poly.exe terminal/engine-poly/strata-poly.exe help \
      -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF "${CUDA_DEFS[@]}" \
      -DSTRATA_BUILD_POLY=ON "-DSTRATA_GGML_DIR=$(cygpath -m "$LLAMA_DIR")" || return 1; }
  want poly-vision && { step_build Strata-Poly-Vision "$VISION_SRC" terminal/build-vision-poly strata-vision bin/strata-vision.exe terminal/engine-poly/strata-vision.exe version \
      "${VISION_DEFS[@]}" -DSTRATA_VISION_POLY=ON "-DLLAMA_DIR=$(cygpath -m "$VISION_LLAMA_DIR")" || return 1; }
  return 0
}
run_all
rc=$?
[ $rc -eq 0 ] && banner "모든 빌드 완료" || printf '\n\033[1;31m빌드가 중단되었습니다. 위 로그를 확인하세요.\033[0m\n'
exit $rc
