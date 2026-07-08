#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_DIR="${1:-}"
PACKAGE_DIR="${PROJECT_DIR}/outputs/exports"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PACKAGE_PATH="${PACKAGE_DIR}/pi05_policy_${TIMESTAMP}.tar.gz"

if [[ -z "${SOURCE_DIR}" ]]; then
  cat <<'EOF'
Usage:
  bash deploy/scripts/package_model.sh /abs/path/to/deploy_bundle
EOF
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  printf 'Model export directory not found: %s\n' "${SOURCE_DIR}" >&2
  exit 1
fi

mkdir -p "${PACKAGE_DIR}"
tar -C "${SOURCE_DIR}" -czf "${PACKAGE_PATH}" .
printf 'Packaged %s\n' "${PACKAGE_PATH}"
