FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

<<<<<<< HEAD
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
=======
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
