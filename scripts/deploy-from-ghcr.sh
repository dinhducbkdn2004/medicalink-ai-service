#!/usr/bin/env bash
# Deploy AI worker trên VM: login GHCR → pull image → compose.ghcr.yaml up
# Usage: GITHUB_REPOSITORY=owner/repo GITHUB_OWNER=owner VM_* GHCR_TOKEN=... ./scripts/deploy-from-ghcr.sh <image_tag>
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <image_tag>  (e.g. production-$(git rev-parse HEAD 2>/dev/null || echo sha))" >&2
  exit 1
fi

TAG="$1"
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-}"
GITHUB_OWNER="${GITHUB_OWNER:-}"
if [ -z "$GITHUB_OWNER" ] && [ -n "$GITHUB_REPOSITORY" ]; then
  GITHUB_OWNER=$(echo "$GITHUB_REPOSITORY" | cut -d'/' -f1)
fi
if [ -z "$GITHUB_OWNER" ]; then
  echo "Set GITHUB_OWNER or GITHUB_REPOSITORY" >&2
  exit 1
fi
GITHUB_OWNER=$(echo "$GITHUB_OWNER" | tr '[:upper:]' '[:lower:]')

if [ -z "${VM_HOST:-}" ] || [ -z "${VM_USER:-}" ] || [ -z "${VM_SSH_KEY:-}" ] || [ -z "${GHCR_TOKEN:-}" ]; then
  echo "VM_HOST, VM_USER, VM_SSH_KEY, GHCR_TOKEN required" >&2
  exit 1
fi

IMAGE="ghcr.io/${GITHUB_OWNER}/medicalink-ai:${TAG}"
AI_DIR="/home/${VM_USER}/medicalink-ai-service"

SSH_KEY_FILE=$(mktemp)
trap 'rm -f "$SSH_KEY_FILE"' EXIT
echo "$VM_SSH_KEY" > "$SSH_KEY_FILE"
chmod 600 "$SSH_KEY_FILE"

ssh_exec() {
  ssh -i "$SSH_KEY_FILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${VM_USER}@${VM_HOST}" "$@"
}

if [ "${VM_GIT_PULL:-}" = "true" ]; then
  echo "[deploy-ai] git pull medicalink-ai-service on VM..."
  if ssh_exec "cd $AI_DIR && git rev-parse --is-inside-work-tree >/dev/null 2>&1"; then
    ssh_exec "cd $AI_DIR && git pull --ff-only" || ssh_exec "cd $AI_DIR && git pull" || echo "[deploy-ai] warn: git pull failed"
  fi
fi

echo "[deploy-ai] docker login + pull $IMAGE"
ssh_exec "cd $AI_DIR && echo '$GHCR_TOKEN' | docker login ghcr.io -u '$GITHUB_OWNER' --password-stdin"
ssh_exec "cd $AI_DIR && export MEDICALINK_AI_IMAGE='$IMAGE' && docker compose -f compose.ghcr.yaml pull && docker compose -f compose.ghcr.yaml up -d"

echo "[deploy-ai] done."
