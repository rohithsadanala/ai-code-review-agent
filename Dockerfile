FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then the app
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY eval_data/ ./eval_data/
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

ENV CRA_HOST=0.0.0.0 CRA_PORT=8000 CRA_DATA_DIR=/app/data
EXPOSE 8000

CMD ["cra", "serve", "--host", "0.0.0.0", "--port", "8000"]
