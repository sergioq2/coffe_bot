from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import uvicorn

app = FastAPI()

model = YOLO('runs/classify/train/weights/best.pt')

def prediction(image: Image.Image):
    results = model.predict(image)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    #higuest probability
    probability = max(probs)
    pred = names_dict[probs.index(max(probs))]
    return pred, probability

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        pred, probability = prediction(image)
        return JSONResponse(content={"prediction": pred, 
                                     "probability": probability})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
