#!/bin/sh
# init-ollama.sh

# Exit on any error
set -e

# Define the Ollama host and model
OLLAMA_HOST=${OLLAMA_HOST:-"http://ollama:11434"}
OLLAMA_MODEL=${OLLAMA_MODEL:-"all-minilm:33m"}

echo "Waiting for Ollama to be ready at ${OLLAMA_HOST}..."

# Wait for the Ollama API to be available
# Use a loop with a timeout to avoid waiting forever
ATTEMPTS=0
MAX_ATTEMPTS=30
while ! curl -s -f "${OLLAMA_HOST}/api/tags" > /dev/null; do
  ATTEMPTS=$((ATTEMPTS + 1))
  if [ ${ATTEMPTS} -ge ${MAX_ATTEMPTS} ]; then
    echo "Ollama is not responding after ${MAX_ATTEMPTS} attempts. Exiting."
    exit 1
  fi
  echo "Ollama not ready, waiting 5 seconds..."
  sleep 5
done

echo "Ollama is ready."

# Check if the model is already available
echo "Checking if model '${OLLAMA_MODEL}' is available..."
if curl -s -f "${OLLAMA_HOST}/api/tags" | grep -q "\"name\": \"${OLLAMA_MODEL}\""; then
  echo "Model '${OLLAMA_MODEL}' already exists. Skipping pull."
else
  echo "Model '${OLLAMA_MODEL}' not found. Pulling model..."
  # Use curl to pull the model
  curl -X POST "${OLLAMA_HOST}/api/pull" -d "{\"name\": \"${OLLAMA_MODEL}\"}"
  echo "Model pull command sent. Waiting for model to be ready..."

  # Wait for the model to be available
  ATTEMPTS=0
  MAX_ATTEMPTS=60 # Increased attempts for model download
  while ! curl -s -f "${OLLAMA_HOST}/api/tags" | grep -q "\"name\": \"${OLLAMA_MODEL}\""; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ ${ATTEMPTS} -ge ${MAX_ATTEMPTS} ]; then
      echo "Model '${OLLAMA_MODEL}' not ready after ${MAX_ATTEMPTS} attempts. Exiting."
      exit 1
    fi
    echo "Model '${OLLAMA_MODEL}' not yet available, waiting 5 seconds..."
    sleep 5
  done
  echo "Model '${OLLAMA_MODEL}' is ready."
fi

echo "Ollama initialization complete."
