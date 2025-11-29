# Click-a-bait? — Clickbait Detection App

Clickbait is a lightweight NLP application that evaluates news headlines for clickbait likelihood and emotional tone. It uses a trained LSTM model served through a FastAPI backend, with a simple browser-based frontend.

Frontend (Netlify): https://clickabaitproject.netlify.app/

The backend on Google Cloud Run is currently stopped, so predictions will not work until it is restarted.

### Features
- Detects clickbait vs non-clickbait
- Predicts emotional tone (neutral / sensational)
- Displays confidence scores
- Clean, minimal web interface
- Entire system is containerized and cloud-deployable

### Tech Stack

### Backend:
1. Python
2. FastAPI
3. TensorFlow (LSTM model)
4. NLTK preprocessing
5. Docker
6. Google Cloud Run + Artifact Registry

### Frontend:
1. HTML
2. CSS
3. JavaScript
4. Netlify for static hosting
5. Node modules

### Steps to Run Backend Locally:

a. Create a virtual environment:
    python -m venv .venv
    .venv\Scripts\activate

b. Install:
    pip install -r requirements.txt

c. Run:
    uvicorn main:app --reload

Backend is available at: http://127.0.0.1:8000/docs


### Steps to Run Frontend Locally:

a. Open frontend/index.html in a browser or run a simple server: 
    cd frontend
    python -m http.server 8001

b. Then open: http://localhost:8001


### Run Backend Using Docker

a. Build image:
    docker build -t clickbait-backend .

b. Run Container:
    docker run -p 8080:8080 clickbait-backend


### Deployment Summary: 

a. Backend container is pushed to Google Artifact Registry

b. Served via Google Cloud Run

c. Frontend is deployed on Netlify

d. Frontend calls the Google Cloud Run /predict endpoint



#### Note: Cloud Run is currently stopped, so the public site cannot fetch predictions.



Authors:
Dnyaneshwari Rakshe
Jayan Agarwal





