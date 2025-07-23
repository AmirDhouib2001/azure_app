from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("iris_model.pkl")

# Initialiser l'application FastAPI
app = FastAPI()

# Définir les dossiers pour le frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            sepal_length: float = Form(...),
            sepal_width: float = Form(...),
            petal_length: float = Form(...),
            petal_width: float = Form(...)):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    flower_name = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"][prediction]
    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": flower_name
    })
