FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Install the package
RUN pip3 install -e .

# Expose Flask API port
EXPOSE 5000

# Set environment variables
ENV PYTHONPATH=/app/src
ENV CUDA_VISIBLE_DEVICES=0

# Run the Flask API
CMD ["python3", "api/app.py"]