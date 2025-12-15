# 1. Base Image: Start with a lightweight Python 3.12 version
FROM python:3.12-slim

# 2. Set Working Directory: Create a folder inside the container
WORKDIR /app

# 3. Install System Dependencies (Needed for some PDF/Math libraries)
# We install these just in case your report generation needs them
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Requirements: Move requirements.txt first (for caching speed)
COPY app/requirements.txt .

# 5. Install Python Libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Project Files: Move all your code into the container
COPY . .

# 7. Expose Port: Open the port Streamlit runs on
EXPOSE 8501

# 8. Run Command: What happens when the container starts?
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]