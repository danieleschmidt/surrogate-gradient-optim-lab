#!/bin/bash
set -e

echo "Rolling back deployment..."

# Rollback to previous version
kubectl rollout undo deployment/surrogate-optimizer

# Wait for rollback to complete
kubectl rollout status deployment/surrogate-optimizer

echo "Rollback complete!"
