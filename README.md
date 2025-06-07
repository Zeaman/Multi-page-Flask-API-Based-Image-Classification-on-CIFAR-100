# Multi-page-Flask-API-Based-Image-Classification-on-CIFAR-100-
Multi page and multi models flask-Based Image Classification with EfficientNet on CIFAR-100
# Image Classifier System

This project is a web-based image classification system that uses pre-trained deep learning models to categorize images into one of 100 classes from the CIFAR-100 dataset.  It provides a user-friendly interface for uploading images and selecting different models for prediction.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This image classifier system allows users to upload an image, choose from a selection of pre-trained models (EfficientNet-B0, EfficientNet-B1, and ResNet50), and receive a prediction of the image's category. The system is built using Flask for the backend and HTML/CSS for the frontend.

## Features

- Image upload and classification.
- Support for multiple deep learning models (EfficientNet-B0, EfficientNet-B1, ResNet50).
- User-friendly web interface.
- Display of training logs and accuracy plots.
- Information about the models and the dataset used.

## Technologies Used

- Python
- Flask
- TensorFlow/Keras (or PyTorch, depending on your implementation)
- HTML
- CSS

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/Zeaman/Multi-page-Flask-API-Based-Image-Classification-on-CIFAR-100.git
    ```

2.  Navigate to the project directory:

    ```bash
    cd Multi-page-Flask-API-Based-Image-Classification-on-CIFAR-100
    ```

3.  Create a virtual environment (recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    (Make sure you create a `requirements.txt` file listing all your project's dependencies.  You can generate it using `pip freeze > requirements.txt` after installing the packages.)

## Usage

1.  Run the Flask application:

    ```bash
    python app.py  # Or whatever your main Python file is named
    ```

2.  Open your web browser and go to `http://127.0.0.1:5000/` (or the address shown in your terminal).

3.  Follow the instructions on the website to upload an image and select a model for classification.

## Model Training

The models used in this system were trained on the CIFAR-100 dataset.  Details about the training process are provided on the "About" page of the application, including the number of epochs used, batch sizes, optimizer, and loss function.  Training logs and accuracy plots are also available.

## Dataset

The system uses the CIFAR-100 dataset, which contains 100 classes with 600 images per class (500 for training, 100 for testing). Images are resized before being fed to the models.  More information about the dataset can be found on the official CIFAR-100 website: <https://www.cs.toronto.edu/~kriz/cifar.html>

## How It Works

1.  **Upload an Image:** Users can upload an image from their local machine.
2.  **Choose a Model:** Users can select one of the available pre-trained models.
3.  **Click "Predict":** The system processes the image using the chosen model and returns the predicted class label.

## File Structure 
In Local machine

Folder name / 

├── app.py          # Main Flask application file

├── templates/      # HTML templates

│   ├── index.html

│   ├── about.html

│   └── contact.html

├── static/         # Static files (CSS, images)

│   ├── style.css

│   ├── logo.png

│   ├── about_bg.jpg

│   ├── contact_bg.jpg

│   └── ... (other images)

└── requirements.txt  # Project dependencies

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License

Copyright (c) 2025 Amanuel Mihiret

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact

Amanuel Mihiret (MSc. in Mechatronics Engineering)
zeaman44@gmail.com,
amanmih@dtu.edu.et

