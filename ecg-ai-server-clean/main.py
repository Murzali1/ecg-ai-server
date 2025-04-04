
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = tf.keras.models.load_model("model/model.h5")
with open("model/labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

class ImagePayload(BaseModel):
    image: str

def preprocess_image(base64_string):
    header, encoded = base64_string.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.post("/analyze-ecg")
async def analyze_ecg(payload: ImagePayload):
    image = preprocess_image(payload.image)
    prediction = model.predict(image)[0]
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return {
        "diagnosis": labels[class_index],
        "confidence": round(confidence, 4),
        "advice": get_advice(labels[class_index])
    }

def get_advice(diagnosis):
    if "AV" in diagnosis:
        return "Противопоказано введение метопролола"
    elif "инфаркт" in diagnosis.lower():
        return "Пациенту требуется неотложная помощь"
    return "Метопролол может быть применён"
