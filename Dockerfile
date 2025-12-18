# 1. Base Image
FROM python:3.12-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Requirements
COPY app/requirements.txt .

# 5. Install Python Libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Project Files
COPY . .

# --- CRITICAL MISSING PART FOR HUGGING FACE ---
# 7. Create a non-root user (Required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 8. Expose Port 7860
EXPOSE 7860

# 9. Run Command
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]