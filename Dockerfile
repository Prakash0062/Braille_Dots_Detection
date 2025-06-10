# Use Python 3.10 as YOLOv8 and Ultralytics support it well
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements first for better cache usage
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose Flask default port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
