#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"
mkdir -p "${UNIT_DIR}" /mnt/nas/logs/CamAngle_VLM/service
install -m 0644 "${SCRIPT_DIR}/camangle-vlm-app.service" "${UNIT_DIR}/camangle-vlm-app.service"
systemctl --user daemon-reload
systemctl --user enable --now camangle-vlm-app.service
systemctl --user status camangle-vlm-app.service --no-pager
