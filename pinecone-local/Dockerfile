# Use the official Pinecone Local image as a base
FROM ghcr.io/pinecone-io/pinecone-local:latest

# Expose the default Pinecone Local ports
# The user specified 5081-6000, so we'll expose that range
EXPOSE 5081-6000

# Set environment variables as specified by the user
ENV PORT=5081
ENV PINECONE_HOST=localhost
