#!/usr/bin/env bash
# =============================================================
# Phase 5 - one-shot local Kubernetes setup with kind.
#
# What it does:
#   1. creates a kind cluster (Kubernetes-in-Docker)
#   2. installs metrics-server (needed by the HPA)
#   3. builds the API image and loads it into the cluster
#   4. creates the engine-credentials secret from serving/.env
#   5. applies all manifests
#
# Prereqs: docker, kind, kubectl installed. serving/.env filled in.
#   kind:    https://kind.sigs.k8s.io/docs/user/quick-start/#installation
#   kubectl: https://kubernetes.io/docs/tasks/tools/
#
# Usage (from repo root):
#   bash k8s/setup_kind.sh
#
# Afterwards:
#   kubectl -n novabot get pods -w
#   kubectl -n novabot port-forward svc/novabot-api 8080:80
#   curl -s localhost:8080/readyz
#   python serving/test_api.py --base http://localhost:8080
# =============================================================
set -e

CLUSTER_NAME="novabot"
IMAGE="novabot-api:local"

# ---- 1. Cluster ----
if ! kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
  echo ">> Creating kind cluster '${CLUSTER_NAME}'..."
  kind create cluster --name "$CLUSTER_NAME"
else
  echo ">> kind cluster '${CLUSTER_NAME}' already exists."
fi
kubectl cluster-info --context "kind-${CLUSTER_NAME}"

# ---- 2. metrics-server (for HPA) ----
echo ">> Installing metrics-server..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
# kind's kubelets use self-signed certs -> metrics-server needs this flag:
kubectl -n kube-system patch deployment metrics-server --type=json -p \
  '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' \
  || true

# ---- 3. Build + load the API image ----
echo ">> Building API image..."
docker build -f serving/Dockerfile.api -t "$IMAGE" .
echo ">> Loading image into kind..."
kind load docker-image "$IMAGE" --name "$CLUSTER_NAME"

# ---- 4. Namespace + secret from serving/.env ----
kubectl apply -f k8s/00-namespace.yaml
if [ -f serving/.env ]; then
  echo ">> Creating engine-credentials secret from serving/.env..."
  set -a; source serving/.env; set +a
  kubectl -n novabot create secret generic engine-credentials \
    --from-literal=ENGINE_BASE_URL="$ENGINE_BASE_URL" \
    --from-literal=ENGINE_API_KEY="$ENGINE_API_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -
else
  echo "WARNING: serving/.env not found - create the secret manually (see k8s/10-secret.example.yaml)"
fi

# ---- 5. Apply workload manifests ----
kubectl apply -f k8s/20-deployment.yaml
kubectl apply -f k8s/30-service.yaml
kubectl apply -f k8s/40-hpa.yaml

echo ""
echo ">> Done. Watch pods come up:"
echo "   kubectl -n novabot get pods -w"
echo ">> Then expose locally:"
echo "   kubectl -n novabot port-forward svc/novabot-api 8080:80"
echo "   curl -s localhost:8080/readyz"
echo "   python serving/test_api.py --base http://localhost:8080"
