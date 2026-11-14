#!/usr/bin/env bash
# DevLab: build CrispASR with the HIP backend for one gfx target, run the
# weight-free unit tier, then transcribe samples/jfk.wav with whisper tiny.en
# on the GPU. Runs inside a stock rocm/dev-ubuntu container, from the checkout.
# Results go to out/ (results.txt, ctest.xml, logs).
#
# Mount the checkout at /work/CrispASR, not /src: test_scnet_validation
# normalises paths by plain substring, so a repo at /src also rewrites the
# build tree's own src/ directory and the test fails on our layout.
#
# Usage: bash .github/devlab-rocm-test.sh <gfx>
set -uo pipefail
GFX=${1:?gfx target, e.g. gfx1201}
SRC=$(pwd)
OUT=$SRC/out
rm -rf "$OUT"; mkdir -p "$OUT"
: > "$OUT/results.txt"
result() { echo "$1=$2" >> "$OUT/results.txt"; }
export DEBIAN_FRONTEND=noninteractive

# ROCM_SDK_INDEX (optional): install the ROCm SDK from pip instead of using the
# image's /opt/rocm. For rocm-next, which ships no image or apt repo. Space-
# separated: the index, then any extra indexes.
ROCM_SDK_INDEX=${ROCM_SDK_INDEX:-}
# make: test-portable-cpu-config configures a nested project with the default
# "Unix Makefiles" generator (plain ubuntu has no make; the ROCm images do).
# The 7.2 "complete" image lacks the BLAS dev packages (as upstream's release.yml
# notes); the 7.14+ "full" images ship them and have no hipblas-dev in apt.
apt_rocm=""
[ -d /opt/rocm/lib/cmake/hipblas ] || apt_rocm="hipblas-dev rocblas-dev"
[ -n "$ROCM_SDK_INDEX" ] && apt_rocm="python3-venv"
apt-get update -qq && apt-get install -y -qq --no-install-recommends \
  cmake ninja-build make g++ git curl ca-certificates patchelf \
  python3 python3-numpy python3-pytest $apt_rocm > "$OUT/apt.log" 2>&1 \
  || { tail -n 20 "$OUT/apt.log"; result setup fail; exit 0; }
git config --global --add safe.directory '*'

if [ -n "$ROCM_SDK_INDEX" ]; then
  set -- $ROCM_SDK_INDEX
  idx="--index-url $1"; shift
  for x in "$@"; do idx="$idx --extra-index-url $x"; done
  python3 -m venv /opt/rocm-sdk \
    && PIP_CONFIG_FILE=/dev/null /opt/rocm-sdk/bin/pip install -q --pre $idx \
         "rocm[libraries,devel]" "rocm-sdk-device-$GFX" > "$OUT/sdk.log" 2>&1 \
    && /opt/rocm-sdk/bin/rocm-sdk init >> "$OUT/sdk.log" 2>&1 \
    || { tail -n 30 "$OUT/sdk.log"; result setup fail; exit 0; }
  ROCM_PATH=$(/opt/rocm-sdk/bin/rocm-sdk path --root)
  export ROCM_PATH PATH="$ROCM_PATH/bin:$PATH"
  /opt/rocm-sdk/bin/pip list 2>/dev/null | grep -i '^rocm' | tee "$OUT/sdk-version.txt"
else
  export ROCM_PATH=/opt/rocm PATH="/opt/rocm/bin:$PATH"
fi

rocminfo > "$OUT/rocminfo.txt" 2>&1 || true
grep -o 'gfx[0-9a-f]*' "$OUT/rocminfo.txt" | sort -u | tr '\n' ' ' > "$OUT/arch.txt" || true
{ hipconfig --version; [ -s "$OUT/sdk-version.txt" ] && echo " (pip $(awk '$1=="rocm"{print "rocm " $2}' "$OUT/sdk-version.txt"))"; } \
  > "$OUT/hip.txt" 2>/dev/null || true

# As upstream's release.yml: host code with g++, HIP through its own compiler.
export HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)"
if cmake -S $SRC -B $SRC/build -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DGGML_HIP=ON -DGGML_NATIVE=OFF \
     -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
     -DGPU_TARGETS="$GFX" -DAMDGPU_TARGETS="$GFX" \
     -DCRISPASR_BUILD_TESTS=ON \
     > "$OUT/configure.log" 2>&1 \
   && cmake --build $SRC/build -j "$(nproc)" > "$OUT/build.log" 2>&1; then
  result build pass
else
  result build fail
  tail -n 80 "$OUT/configure.log" "$OUT/build.log" 2>/dev/null
  exit 0
fi
tail -n 3 "$OUT/build.log"

# The weight-free unit tier, as upstream's ci.yml linux-unit job runs it.
# Exit code is recorded, not fatal: the judge step reads ctest.xml.
rc=0
ctest --test-dir $SRC/build -L unit -E live --timeout 300 --output-on-failure \
  --output-junit "$OUT/ctest.xml" > "$OUT/ctest.log" 2>&1 || rc=$?
echo "$rc" > "$OUT/ctest.rc"
tail -n 40 "$OUT/ctest.log"

# A real transcription on the GPU: whisper tiny.en on JFK's 11 s inaugural clip.
MODEL=$SRC/out/ggml-tiny.en.bin
if curl -fsSL --retry 5 -o "$MODEL" \
     https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin; then
  $SRC/build/bin/crispasr -m "$MODEL" -f $SRC/samples/jfk.wav > "$OUT/transcribe.log" 2>&1
  echo "exit $?" >> "$OUT/transcribe.log"
  cat "$OUT/transcribe.log"
  # The GPU backend must be the one used, not a silent CPU fallback.
  if grep -Eq 'using ROCm[0-9]+ backend' "$OUT/transcribe.log"; then
    result transcribe-gpu-backend pass
  else
    result transcribe-gpu-backend fail
  fi
  if grep -qi 'ask not what your country can do for you' "$OUT/transcribe.log"; then
    result transcribe-text pass
  else
    result transcribe-text fail
  fi
  rm -f "$MODEL"
else
  result transcribe-download fail
fi
