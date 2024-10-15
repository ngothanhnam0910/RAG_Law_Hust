FROM python:3.10-slim

# Define label
LABEL maintainer="NamNT"

WORKDIR /app

COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt

# install necessary library
RUN apt-get update -y
RUN pip install --no-cache-dir -r requirements.txt

# Expose cổng để truy cập từ bên ngoài
EXPOSE 5001

# Chạy FastAPI khi container được khởi động
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]
