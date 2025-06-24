from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import pickle

# Load model
with open("crop_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Mapping from class to crop name
crop_labels = {
    0: "pomegranate",
    1: "mango",
    2: "grapes",
    3: "mulberry",
    4: "ragi",
    5: "potato"
}

# FastAPI setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Input feature list
features_list = ["N", "P", "K", "pH", "EC", "S", "Cu", "Fe", "Mn", "Zn", "B"]

# Show form
@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "features": features_list, "result": None, "crop": None})

# Predict from form input
@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    N: float = Form(...),
    P: float = Form(...),
    K: float = Form(...),
    pH: float = Form(...),
    EC: float = Form(...),
    S: float = Form(...),
    Cu: float = Form(...),
    Fe: float = Form(...),
    Mn: float = Form(...),
    Zn: float = Form(...),
    B: float = Form(...)
):
    input_data = np.array([[N, P, K, pH, EC, S, Cu, Fe, Mn, Zn, B]])
    prediction = int(model.predict(input_data)[0])
    crop_name = crop_labels.get(prediction, "Unknown")
    return templates.TemplateResponse("form.html", {
        "request": request,
        "features": features_list,
        "result": prediction,
        "crop": crop_name
    })
#python -m uvicorn main:app --reload 