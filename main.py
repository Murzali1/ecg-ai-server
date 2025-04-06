from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

app = FastAPI()

# Заглушка модели (вместо настоящей)
class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.1, 0.3, 0.6]])  # Пример предсказания

# Заменить на torch.load('model/model.pt') при наличии настоящей модели
model = DummyModel()
model.eval()

# Заменить на загрузку из JSON при необходимости
labels = ["Норма", "AV-блокада", "Инфаркт миокарда"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/analyze-ecg")
async def analyze_ecg(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(input_tensor)
        predicted_class = predictions.argmax().item()
        confidence = torch.softmax(predictions, dim=1)[0][predicted_class].item()

    result = {
        "class_index": predicted_class,
        "class_name": labels[predicted_class],
        "confidence": round(confidence, 4)
    }

    return JSONResponse(content=result)
