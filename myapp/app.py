from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy   as np
from joblib import load
from sklearn.preprocessing import StandardScaler
import uvicorn


app = FastAPI()

# load templates

templates = Jinja2Templates(directory="templates")

# load model and scaler

model = load("pkl/ridge.pkl")
scaler = load("pkl/scaler.pkl")


@app.get("/",response_class=HTMLResponse)
async def read_index(request:Request):
    return templates.TemplateResponse("home.html", {"request": request})


# Route : prediction page (Get for form, Post for prediction)
@app.get("/predictdata",response_class=HTMLResponse)
async def show_form(request:Request):
    return templates.TemplateResponse("home.html", {"request":request})




@app.post("/predictdata",response_class=HTMLResponse,name="predict")
async def predict(
    request: Request,
    Temperature: float = Form(...),
    RH: float = Form(...),
    Ws: float = Form(...),
    Rain: float = Form(...),
    FFMC: float = Form(...),
    DMC: float = Form(...),
    ISI: float = Form(...),
    Classes: float = Form(...),
    Region: float = Form(...)
):
    
    # prepare input 
    input_features = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

    # Scale input
    scaled_input = scaler.transform(input_features)


    # predict
    prediction = model.predict(scaled_input)

    # return result

    return templates.TemplateResponse("home.html", {"request":request,'result':round(prediction[0],2)})




if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
