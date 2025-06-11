# Base image with Python 3.12
FROM python:3.12

# Set working directory
WORKDIR /workspace

# Install Git inside the container
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY . . 

RUN pip install -r requirements.txt

#ENTRYPOINT [ "/bin/bash", "-l", "-c" ]

#CMD ["bash"]
