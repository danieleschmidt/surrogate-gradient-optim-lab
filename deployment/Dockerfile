# Production Dockerfile for Surrogate Optimization Service
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash surrogateuser
RUN chown -R surrogateuser:surrogateuser /app
USER surrogateuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from surrogate_optim import SurrogateOptimizer; print('Health check passed')"

# Production command
CMD ["python", "-m", "surrogate_optim.server", "--host", "0.0.0.0", "--port", "8080"]
