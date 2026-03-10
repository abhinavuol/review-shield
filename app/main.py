from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.predictor import ReviewPredictor
from app.schemas import ReviewRequest, ReviewResponse


app = FastAPI(title="Fake Review Detector API")

templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parents[1] / "templates")
)

predictor = ReviewPredictor()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": None,
            "probability": None,
            "review_text": ""
        }
    )


@app.post("/predict", response_model=ReviewResponse)
def predict_api(payload: ReviewRequest):
    label, fake_probability = predictor.predict(payload.text)
    return ReviewResponse(label=label, fake_probability=round(fake_probability, 4))


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(request: Request, text: str = Form(...)):
    label, fake_probability = predictor.predict(text)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": label,
            "probability": round(fake_probability, 4),
            "review_text": text
        }
    )