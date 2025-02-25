# Use Python 3.12 with Debian Bullseye (full support for apt)
FROM python:3.12-bullseye

# Set working directory inside the container
WORKDIR /app

# Use a faster mirror and reduce unnecessary package installations
RUN echo "deb http://deb.debian.org/debian bullseye main" > /etc/apt/sources.list && \
    apt-get update -o Acquire::Retries=3 && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to use Docker's cache efficiently
COPY requirements.txt .

# Install Python dependencies (cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Start FastAPI server with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
