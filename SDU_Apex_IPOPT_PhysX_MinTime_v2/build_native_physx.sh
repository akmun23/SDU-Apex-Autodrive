#!/usr/bin/env bash
# Compile exactly the checked-out project's native PhysX source and archives.
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 /path/to/SDU-Apex-Autodrive [output-binary]" >&2
  exit 2
fi
REPO="$(cd "$1" && pwd)"
OUT="${2:-$PWD/native_physx_vehicle_step}"
ROOT="$REPO/external/physx-4.1-source/physx"
SHARED="$REPO/external/physx-4.1-source/pxshared"
LIB="$ROOT/bin/linux.clang/release"
SOURCES=( "$REPO/f1tenth_localization/src/physx_native_vehicle_step.cpp"
          "$REPO/f1tenth_localization/src/physx_glibc_compat.cpp" )
LIBS=(PhysXVehicle PhysXExtensions PhysXCooking PhysX PhysXCommon PhysXFoundation PhysXPvdSDK)
FLAGS=()
for name in "${LIBS[@]}"; do
  path="$LIB/lib${name}_static_64.a"
  if [ ! -f "$path" ]; then
    echo "Missing repo native-PhysX archive: $path" >&2;exit 1
  fi
  FLAGS+=("$path")
done
mkdir -p "$(dirname "$OUT")"
"${CXX:-g++}" -std=c++17 -O3 -DNDEBUG -DPX_PHYSX_STATIC_LIB \
  -I"$ROOT/include" -I"$SHARED/include" \
  -I"$ROOT/source/physxvehicle/src" \
  "${SOURCES[@]}" -Wl,--start-group "${FLAGS[@]}" -Wl,--end-group \
  -ldl -lpthread -o "$OUT"
echo "Native PhysX binary: $OUT"
