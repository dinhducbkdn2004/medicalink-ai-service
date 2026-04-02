#!/usr/bin/env bash
# Pull image medicalink-ai từ GHCR và chạy lại compose trên VM.
# Dùng trong GitHub Actions sau khi cd-docker push xong.
# Usage: ./scripts/deploy-ai-to-vm.sh <image_tag>
# Ví dụ: ./scripts/deploy-ai-to-vm.sh staging-abc123def456...
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <image_tag>" >&2
  exit 1
fi

IMAGE_TAG="$1"
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-}"
if [ -z "$GITHUB_REPOSITORY" ]; then
  echo "GITHUB_REPOSITORY is not set (e.g. owner/medicalink-ai-service)" >&2
  exit 1
fi
GITHUB_OWNER=$(echo "$GITHUB_REPOSITORY" | cut -d'/' -f1)
IMAGE="ghcr.io/${GITHUB_OWNER}/medicalink-ai:${IMAGE_TAG}"

if [ -z "${VM_HOST:-}" ] || [ -z "${VM_USER:-}" ] || [ -z "${VM_SSH_KEY:-}" ]; then
  echo "VM_HOST, VM_USER and VM_SSH_KEY must be set" >&2
  exit 1
fi

if [ -z "${GHCR_TOKEN:-}" ]; then
  echo "GHCR_TOKEN must be set (dùng GITHUB_TOKEN của workflow)" >&2
  exit 1
fi

PROJECT_DIR="/home/${VM_USER}/medicalink-ai-service"
OVERRIDE="docker-compose.cd.override.yml"

SSH_KEY_FILE=$(mktemp)
echo "$VM_SSH_KEY" > "$SSH_KEY_FILE"
chmod 600 "$SSH_KEY_FILE"
trap 'rm -f "$SSH_KEY_FILE"' EXIT

ssh_cmd() {
  ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${VM_USER}@${VM_HOST}" "$@"
}

echo "[deploy-ai] SSH ${VM_USER}@${VM_HOST}"
ssh_cmd "echo 'ok'"
ssh_cmd "cd $PROJECT_DIR && echo '$GHCR_TOKEN' | docker login ghcr.io -u '$GITHUB_OWNER' --password-stdin"
ssh_cmd "docker pull '$IMAGE'"

# Ghi override image (compose.integrated.yaml có build: . — override bắt buộc để dùng image GHCR)
ssh_cmd "cd $PROJECT_DIR && rm -f '$OVERRIDE'"
ssh_cmd "cd $PROJECT_DIR && printf '%s\n' 'services:' '  medicalink-ai:' '    image: $IMAGE' '    pull_policy: always' > '$OVERRIDE'"

if ssh_cmd "cd $PROJECT_DIR && docker compose -f compose.integrated.yaml -f $OVERRIDE up -d --force-recreate"; then
  echo "[deploy-ai] Done: $IMAGE"
  ssh_cmd "cd $PROJECT_DIR && docker compose -f compose.integrated.yaml -f $OVERRIDE ps"
else
  echo "[deploy-ai] docker compose failed" >&2
  exit 1
fi
