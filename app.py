from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn

app = Flask(__name__)
CORS(app)

# ================= LOAD MODEL =================
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("brain_model.pth", map_location="cpu"))
model.eval()

classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= ROUTE =================
@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']   # ✅ matches React

    image = Image.open(file).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        confidence, predicted = torch.max(probabilities, 0)

    result = {
        "disease": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }

    return jsonify(result)

# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=True)
