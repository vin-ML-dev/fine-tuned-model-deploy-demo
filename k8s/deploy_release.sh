#!/usr/bin/env bash
# =============================================================
# Phase 5 - Steps 21/22: pull-based deploy of a CI-built release
# into the local kind cluster, with rollback support.
#
# Deploy a release built by GitHub Actions:
#   bash k8s/deploy_release.sh <git-sha>
#
# Roll back to the previous version:
#   bash k8s/deploy_release.sh rollback
#
# Watch a rollout:
#   kubectl -n novabot rollout status deploy/novabot-api
# History:
#   kubectl -n novabot rollout history deploy/novabot-api
#
# Note: ghcr.io images from private repos need a pull secret; easiest
# for learning is to make the GHCR package public (Package settings ->
# Change visibility). The pull-secret variant is documented in README.
# =============================================================
set -e

OWNER="${GHCR_OWNER:?Set GHCR_OWNER=your-github-username (lowercase)}"

if [ "$1" == "rollback" ]; then
  echo ">> Rolling back to previous revision..."
  kubectl -n novabot rollout undo deploy/novabot-api
  kubectl -n novabot rollout status deploy/novabot-api
  exit 0
fi

SHA="${1:?Usage: bash k8s/deploy_release.sh <git-sha> | rollback}"
IMAGE="ghcr.io/${OWNER}/novabot-api:${SHA}"

echo ">> Deploying ${IMAGE}"
kubectl -n novabot set image deploy/novabot-api api="$IMAGE"
kubectl -n novabot rollout status deploy/novabot-api
echo ">> Deployed. Verify:"
echo "   kubectl -n novabot get pods"
echo "   python serving/test_api.py --base http://localhost:8080   (with port-forward running)"
