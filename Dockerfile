# Use a minimal Python 3.12 base image
FROM python:3.12-slim

# Install system dependencies required by OpenCV, pytesseract, pdf2image, etc.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose a port (informational, Railway assigns dynamically)
EXPOSE 8080

# Start the app with Gunicorn and bind to the dynamic port provided by Railway
CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:8080"]
