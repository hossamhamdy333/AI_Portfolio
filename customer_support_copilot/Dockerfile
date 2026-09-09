# CPU-only image, deployed on Azure Container Apps. Runs a quantized GGUF
# model via llama.cpp instead of raw transformers -- that's what makes
# CPU-only inference fast enough to finish within Azure's request timeout.
# See src/app.py and notebooks/gguf_conversion.ipynb for details.
FROM python:3.10-slim-bookworm

WORKDIR /app

# System deps needed to build llama-cpp-python and sentence-transformers
# wheels, plus Microsoft's ODBC Driver 18 for Azure SQL (pyodbc alone
# isn't enough -- it needs this system-level driver installed too).
# Uses the modern signed-by keyring approach since apt-key is removed on
# current Debian base images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    curl \
    gnupg \
    unixodbc \
    unixodbc-dev \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -sSL https://packages.microsoft.com/config/debian/12/prod.list \
       | sed 's#deb #deb [signed-by=/usr/share/keyrings/microsoft-prod.gpg] #' \
       > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
