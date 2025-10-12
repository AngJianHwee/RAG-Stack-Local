# Use the official Ollama image as a base
FROM ollama/ollama:latest

# Expose the default Ollama port
EXPOSE 11434

# Command to pull the all-minilm:33m model
# This will run when the container starts for the first time
RUN ollama pull all-minilm:33m

# Command to serve Ollama
CMD ["ollama", "serve"]
