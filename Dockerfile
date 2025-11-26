# Use official Python image (not slim to avoid TF issues)
FROM python:3.11

# Don’t write .pyc files + flush output immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory in container
WORKDIR /code

# Install Python dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4

# Copy the rest of the app (code + models)
COPY . /code

# Cloud Run expects the app to listen on $PORT, default 8080
ENV PORT=8080

# Start FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
