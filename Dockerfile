FROM python:3.10-slim

# Install some basic dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements before cache install
COPY requirements.txt ./

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy code into container
COPY . .

# Expose port for Flask
EXPOSE 5000

# Start service
CMD ["python","main.py"]
