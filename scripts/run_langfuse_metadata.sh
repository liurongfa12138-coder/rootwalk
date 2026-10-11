#!/usr/bin/env bash
set -euo pipefail

rootwalk_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOTWALK_LANGFUSE_CAPTURE_OVERRIDE=false \
  exec "${rootwalk_script_dir}/run_langfuse.sh" "$@"
