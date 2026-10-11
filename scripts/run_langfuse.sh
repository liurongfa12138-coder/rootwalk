#!/usr/bin/env bash
set -euo pipefail

rootwalk_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
rootwalk_langfuse_env="${ROOTWALK_LANGFUSE_ENV:-${rootwalk_repo_root}/.env.langfuse}"

if [[ ! -f "${rootwalk_langfuse_env}" ]]; then
  echo "Missing Langfuse config: ${rootwalk_langfuse_env}" >&2
  exit 2
fi

config_mode="$(stat -c '%a' "${rootwalk_langfuse_env}")"
if [[ "${config_mode}" != "600" ]]; then
  echo "Langfuse config must have mode 600: ${rootwalk_langfuse_env}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${rootwalk_langfuse_env}"
set +a

for required_name in LANGFUSE_BASE_URL LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "Missing ${required_name} in ${rootwalk_langfuse_env}" >&2
    exit 2
  fi
done

case "${LANGFUSE_BASE_URL}" in
  https://*) ;;
  *)
    echo "LANGFUSE_BASE_URL must use HTTPS" >&2
    exit 2
    ;;
esac

rootwalk_langfuse_auth="$({
  printf '%s' "${LANGFUSE_PUBLIC_KEY}:${LANGFUSE_SECRET_KEY}" | base64 | tr -d '\n'
})"
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Basic ${rootwalk_langfuse_auth},x-langfuse-ingestion-version=4"
unset rootwalk_langfuse_auth

capture_setting="${ROOTWALK_LANGFUSE_CAPTURE_OVERRIDE:-${ROOTWALK_OTEL_CAPTURE_CONTENT:-false}}"
case "${capture_setting,,}" in
  1 | true | yes | on) capture_flag="--otel-capture-content" ;;
  0 | false | no | off) capture_flag="--no-otel-capture-content" ;;
  *)
    echo "ROOTWALK_OTEL_CAPTURE_CONTENT must be true or false" >&2
    exit 2
    ;;
esac

cd "${rootwalk_repo_root}"
exec uv run rootwalk run \
  --otel \
  --otel-endpoint "${LANGFUSE_BASE_URL%/}/api/public/otel/v1/traces" \
  "${capture_flag}" \
  "$@"
