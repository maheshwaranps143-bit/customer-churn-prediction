# Customer Churn Prediction System - Docker image
# Builds an image that trains the model (if needed) and serves the FastAPI app.
#
# Build:  docker build -t churn-api .
# Run:    docker run -p 8000:8000 churn-api
#
# To run the Streamlit dashboard instead, override the CMD:
#   docker run -p 8501:8501 churn-api streamlit run src/dashboard.py --server.address 0.0.0.0

FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by some ML libraries (e.g. LightGBM-style builds, matplotlib fonts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model at build time so the image is ready to serve immediately.
# (For a real production setup you'd usually train offline and COPY in the
# artifacts instead - this keeps the demo self-contained.)
RUN python data/generate_sample_data.py && python src/train_model.py

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
