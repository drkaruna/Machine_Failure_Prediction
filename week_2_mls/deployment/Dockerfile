# Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim

WORKDIR /app

COPY week_2_mls/deployment/requirements.txt .

RUN pip install -r requirements.txt

COPY week_2_mls/deployment/ .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
