import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import requests



# URL of the haarcascade xml file
url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

# Download the file
response = requests.get(url)
with open("haarcascade_frontalface_default.xml", "wb") as file:
    file.write(response.content)

# Load the face cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the model
model = tf.keras.models.load_model("modelv1.h5")

# Function to preprocess images
def preprocess_image(img_path):
    # Load the image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        raise ValueError("No faces detected in the image")

    # Crop the first detected face
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]

    # Resize the face to 48x48
    face_resized = cv2.resize(face, (48, 48))
    
    # Expand dimensions to match the expected input shape
    face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel dimension
    face_resized = np.expand_dims(face_resized, axis=0)   # Add batch dimension
    
    # Normalize the image
    face_resized = face_resized / 255.0
    
    return face_resized

# Function to predict the class of an image
def predict_image(model, img_path):
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    return predictions

# Example usage
if __name__ == "__main__":
    # Path to the image
    img_path = r"C:\Users\prana\OneDrive\Desktop\projectD\image-asset.jpeg"
    # Get predictions
    try:
        predictions = predict_image(model, img_path)
        print('Predictions:', predictions)
        # Optionally, get the class with the highest probability
        predicted_class = np.argmax(predictions, axis=1)
        print('Predicted class:', predicted_class)
        label=['angry','disgust','fear','happy','neutral','sad','surprise']
        print(label[predicted_class[0]])
    except ValueError as e:
        print(e)



