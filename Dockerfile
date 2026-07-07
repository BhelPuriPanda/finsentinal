FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create model artifacts directory
RUN mkdir -p models/artifacts

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHERUSAGESTATISTICS=false

EXPOSE 8080

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0"]