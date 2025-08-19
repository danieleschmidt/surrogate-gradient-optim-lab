#!/bin/bash
set -e

echo "Starting deployment..."

# Build and push Docker image
docker build -t surrogate-optimizer:latest .
docker tag surrogate-optimizer:latest your-registry/surrogate-optimizer:latest
docker push your-registry/surrogate-optimizer:latest

# Deploy to Kubernetes
kubectl apply -f deployment/k8s/

# Wait for rollout to complete
kubectl rollout status deployment/surrogate-optimizer

# Run health checks
kubectl wait --for=condition=ready pod -l app=surrogate-optimizer

echo "Deployment complete!"
