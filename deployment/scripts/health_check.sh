#!/bin/bash

# Health check script
ENDPOINT="${1:-http://localhost:8080/health}"

echo "Checking health of $ENDPOINT..."

response=$(curl -s -o /dev/null -w "%{http_code}" $ENDPOINT)

if [ $response -eq 200 ]; then
    echo "✅ Health check passed"
    exit 0
else
    echo "❌ Health check failed (HTTP $response)"
    exit 1
fi
