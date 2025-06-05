# Base image with Python 3.12
FROM python:3.12

# Set working directory
WORKDIR /workspace

# Install Git and Node.js (via NodeSource)
RUN apt-get update && apt-get install -y git curl gnupg ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Git inside the container
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

CMD ["bash"]