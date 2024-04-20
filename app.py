import gradio as gr
import numpy as np
import cv2
import requests
from tensorflow.keras.models import load_model

# Load the trained models
model1 = load_model('./model1.h5')
model2 = load_model('./model2.h5')
model3 = load_model('./model3.h5')
model4 = load_model('./model4.h5')
model5 = load_model('./model5.h5')
# Define image sizes
img_size1 = 150
img_size2 = 224

# Preprocess functions
def load_and_preprocess_image(img, img_size):
    # Convert Gradio image (PIL Image) to numpy array
    img = np.array(img)
    # Convert RGB to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize image to the required input size
    img = cv2.resize(img, (img_size, img_size))
    # Reshape image for model input
    img = img.reshape(-1, img_size, img_size, 1)
    # Normalize image
    img = img / 255.0
    return img

# Prediction functions for each model
def model1_prediction(img):
    img = load_and_preprocess_image(img, img_size1)
    prediction = model1.predict(img)[0][0]
    return {'PNEUMONIA': prediction, 'NORMAL': 1-prediction}

def model2_prediction(img):
    img = load_and_preprocess_image(img, img_size2)
    prediction = model2.predict(img)[0][0]
    return {'PNEUMONIA': 1-prediction, 'NORMAL': prediction}


def load_and_preprocess_image3(img):
    img = np.array(img)
    img = cv2.resize(img, (img_size2, img_size2))
    img = img.reshape(-1, img_size2, img_size2, 3)
    img = img / 255.0
    return img

def model3_prediction(img):
    # Preprocess the image
    img = load_and_preprocess_image3(img)
    # Perform prediction
    prediction = model3.predict(img)[0]  # Assuming your model outputs a single probability
    print(prediction)
    # Return the prediction
    return {'PNEUMONIA': prediction[1], 'NORMAL': prediction[0]}  # Invert the prediction for Gradio


def load_and_preprocess_image4(img):
    img = np.array(img)
    img = cv2.resize(img, (img_size2, img_size2))
    img = img.reshape(-1, img_size2, img_size2, 3)
    img = img / 255.0
    return img

def model4_prediction(img):
    # Preprocess the image
    img = load_and_preprocess_image4(img)
    # Perform prediction
    prediction = model4.predict(img)[0]  # Assuming your model outputs a single probability
    # Return the prediction
    return {'PNEUMONIA': prediction[1], 'NORMAL': prediction[0]}  # Invert the prediction for Gradio


def load_and_preprocess_image5(img):
    img = np.array(img)
    img = cv2.resize(img, (img_size2, img_size2))
    img = img.reshape(-1, img_size2, img_size2, 3)
    img = img / 255.0
    return img

def model5_prediction(img):
    # Preprocess the image
    img = load_and_preprocess_image5(img)
    # Perform prediction
    prediction = model5.predict(img)[0]  # Assuming your model outputs a single probability
    print(prediction)
    # Return the prediction
    return {'PNEUMONIA': prediction[1], 'NORMAL': prediction[0]}  # Invert the prediction for Gradio

# Main function to select model based on input
def pneumonia_detection(model_choice, img):
    if model_choice == "Model-CNN":
        return model1_prediction(img)
    if model_choice == "Model-Inceptionv3":
        return model2_prediction(img)
    if model_choice == "Model-Resnet50":
       return model3_prediction(img)
    if model_choice == "Model-VGG16":
       return model4_prediction(img)
    if model_choice == "Model-VGG19":
       return model5_prediction(img)

# Launch Gradio interface
demo_model = gr.Interface(
    fn=pneumonia_detection,
    inputs=[gr.Dropdown(["Model-CNN", "Model-Inceptionv3" , "Model-Resnet50","Model-VGG16","Model-VGG19"], label="Choose Model"), "image"],  # model choice dropdown, image
    outputs="label",  # label output
    title="Pneumonia Detection",
    description="Choose a model for pneumonia detection: ",
)

# Launch the interface
if __name__ == "__main__":
    demo_model.launch(share=True)
