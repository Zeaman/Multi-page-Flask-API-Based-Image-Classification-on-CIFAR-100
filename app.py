import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from collections import OrderedDict
import torchvision

app = Flask(__name__)

# Define a generic model architecture that supports b0, b1, and resnet50
class CustomModel(nn.Module):
    def __init__(self, model_name="b0", num_classes=100):
        super(CustomModel, self).__init__()
        if model_name == "b0":
            self.model = models.efficientnet_b0(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        elif model_name == "b1":
            self.model = models.efficientnet_b1(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        elif model_name == "resnet50":
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            raise ValueError("Invalid model selected!")
    
    def forward(self, x):
        return self.model(x)

# Function to load the selected model from saved weights
def load_model(model_name):
    model = CustomModel(model_name=model_name, num_classes=100)
    if model_name == "b0":
        model_path = "efficientnet_b0_cifar100.pth"
    elif model_name == "b1":
        model_path = "efficientnet_b1_cifar100_kag.pth"
    elif model_name == "resnet50":
        model_path = "resnet50_cifar100_40.pth"
    else:
        raise ValueError("Invalid model selected!")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # For EfficientNet models, adjust keys by adding the "model." prefix.
    if model_name in ["b0", "b1"]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = "model." + k  # Adjust key to match our model definition
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:  # For resnet50, load the state_dict directly
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model

# Function to return the appropriate image transformation for each model
def get_transform(model_name):
    if model_name == "b0":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif model_name == "b1":
        return transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_name == "resnet50":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Invalid model selected!")

# Load CIFAR-100 class names
cifar100_classes = torchvision.datasets.CIFAR100(root="./data", train=True, download=True).classes
class_names = cifar100_classes

# Ensure the uploads folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if file is provided
        if "file" not in request.files:
            return render_template("index.html", message="No file part", image_path=None)
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No selected file", image_path=None)
        # Save image with a unique filename to avoid overwriting
        filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        counter = 1
        while os.path.exists(save_path):
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(UPLOAD_FOLDER, f"{name}_{counter}{ext}")
            counter += 1
        file.save(save_path)
        return render_template("index.html", image_path=save_path)
    return render_template("index.html", image_path=None)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the uploaded image path and selected model from the form data
    image_path = request.form.get("image_path")
    selected_model = request.form.get("model")
    
    if not image_path:
        return render_template("index.html", message="Please upload an image first", image_path=None)
    
    if selected_model not in ["b0", "b1", "resnet50"]:
        return render_template("index.html", message="Invalid model selected", image_path=image_path)
    
    # Load the appropriate model and transformation
    model = load_model(selected_model)
    transform = get_transform(selected_model)
    
    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax(dim=1).item()  # Get predicted index
        class_name = class_names[predicted_class]         # Map index to class name
    
    return render_template("index.html", image_path=image_path, prediction=f"Predicted Class: {class_name}")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
