FROM python:3.12.11-slim-bookworm

WORKDIR /app

COPY . /app

RUN pip install uv

RUN uv pip install

CMD ["python3", "app.py"]