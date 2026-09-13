#!/bin/bash
# Run from the project root: bash scripts/with_ssd.sh <command> [arguments...]
# No global shell/Docker settings are changed. The internal VM is left intact.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/with_ssd.sh <command> [arguments...]" >&2
  exit 2
fi

ssd_volume="${HYDRA_SSD_VOLUME:-/Volumes/PortableSSD}"
if [[ "$ssd_volume" != /Volumes/* || "$ssd_volume" == *" "* ]]; then
  echo "Use an external volume under /Volumes with no spaces in its name." >&2
  exit 2
fi
# Refuse an absent/unmounted drive instead of creating directories on the Mac.
reported_mount="$(diskutil info -plist "$ssd_volume" | plutil -extract MountPoint raw -o - -)"
if [[ "$reported_mount" != "$ssd_volume" ]]; then
  echo "SSD is not mounted at $ssd_volume; no command was run." >&2
  exit 2
fi

ssd_root="$ssd_volume/hydra-swe"
umask 077
export COLIMA_HOME="$ssd_root/colima"
export COLIMA_CACHE_HOME="$ssd_root/cache/colima"
export COLIMA_PROFILE=hydra-swe-ssd
export UV_CACHE_DIR="$ssd_root/cache/uv"
export HF_HOME="$ssd_root/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$ssd_root/cache/xdg"
export TMPDIR="$ssd_root/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export DOCKER_HOST="unix://$COLIMA_HOME/$COLIMA_PROFILE/docker.sock"
export HYDRA_BENCH_OUTPUT="${HYDRA_BENCH_OUTPUT:-$ssd_root/runs/benchmark}"
unset DOCKER_CONTEXT LIMA_HOME

mkdir -p "$COLIMA_HOME" "$COLIMA_CACHE_HOME" "$UV_CACHE_DIR" \
  "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$ssd_root/runs"
exec "$@"
