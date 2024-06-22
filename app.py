import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Define the image transformation for the input image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the pre-trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=False)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 15)  # Update to 16 classes
model.load_state_dict(torch.load('plantvillage_mobilenetv2.pth', map_location=device))
model = model.to(device)
model.eval()

class_names = ['Pepper Bell - Bacterial spot', 'Pepper Bell - Healthy', 'Potato - Early Blight', 'Potato - Late Blight', 'Potato - healthy', 'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Healthy', 'Tomato - Late Blight', 'Tomato - Leaf Mold', 'Tomato - Mosaic Virus', 'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites', 'Tomato - Target Spot', 'Tomato - YellowLeaf Curl Virus']  # Add the missing class

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict(image_path):
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds[0]]
        return predicted_class

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict(file_path)
            return render_template('result.html', prediction=prediction, image_url=url_for('static', filename='uploads/' + filename))
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
