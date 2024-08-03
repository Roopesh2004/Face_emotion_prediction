import cv2
import tensorflow as tf
import numpy as np
from collections import Counter
from tkinter import Tk, Label, StringVar, Frame, BOTH
from tkinter import ttk
from PIL import Image, ImageTk
import time
import threading
import pandas as pd

# Load the face cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the model
model = tf.keras.models.load_model('new_model.h5')  # Ensure you use the re-saved model


# Function to preprocess images
def preprocess_image(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print("cv2 error:", e)
        return None, None

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # If no faces are detected, return None
    if len(faces) == 0:
        return None, None

    # Crop the first detected face
    (x, y, w, h) = faces[0]
    face = gray[y:y + h, x:x + w]

    # Resize the face to 48x48
    face_resized = cv2.resize(face, (48, 48))

    # Expand dimensions to match the expected input shape
    face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel dimension
    face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension

    # Normalize the image
    face_resized = face_resized / 255.0

    return face_resized, (x, y, w, h)


# Function to predict the class of an image
def predict_image(model, img):
    img, face_coords = preprocess_image(img)
    if img is None:
        return None, None
    predictions = model.predict(img)
    return predictions, face_coords


# Load music data
Music_Player = pd.read_csv("data_moods.csv")
Music_Player = Music_Player[['name', 'artist', 'mood', 'popularity']]


# Function to recommend songs based on predicted class
def Recommend_Songs(pred_class):
    Play = pd.DataFrame()  # Initialize an empty DataFrame
    if pred_class == 'disgust':
        Play = Music_Player[Music_Player['mood'] == 'Sad']
    elif pred_class in ['happy', 'sad']:
        Play = Music_Player[Music_Player['mood'] == 'Happy']
    elif pred_class in ['fear', 'angry']:
        Play = Music_Player[Music_Player['mood'] == 'Calm']
    elif pred_class in ['surprise', 'neutral']:
        Play = Music_Player[Music_Player['mood'] == 'Energetic']

    if not Play.empty:
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
    return Play


# Initialize Tkinter window
root = Tk()
root.title("Live Emotion Recognition")

# Create a frame for better layout management
main_frame = Frame(root, bg="#f0f0f0")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Create a label to display the video feed
video_label = Label(main_frame, bg="#f0f0f0", relief="groove")
video_label.grid(row=0, column=0, padx=10, pady=10, rowspan=3)

# Create a frame for the decision and emotion bar
decision_frame = Frame(main_frame, bg="#f0f0f0", relief="groove", bd=2)
decision_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Create a label to display the decision
decision_label = StringVar()
decision_label.set("Decision will appear here...")
decision_display = Label(decision_frame, textvariable=decision_label, font=("Helvetica", 16), bg="#f0f0f0",
                         wraplength=300)
decision_display.pack(pady=10)

# Create a progress bar to display the emotion
emotion_bar = ttk.Progressbar(decision_frame, orient="horizontal", length=200, mode="determinate", maximum=10)
emotion_bar.pack(pady=10)


# Function to update the progress bar color
def set_progressbar_color(widget, color):
    style = ttk.Style()
    style.theme_use('default')
    style.configure(widget + ".Horizontal.TProgressbar", troughcolor='white', background=color)
    widget_name = widget + ".Horizontal.TProgressbar"
    return widget_name


# Create a frame for the song recommendations
song_frame = Frame(main_frame, bg="#f0f0f0", relief="groove", bd=2)
song_frame.grid(row=1, column=1, padx=10, pady=10, sticky="n")

# Create a label to display recommended songs
song_label = Label(song_frame, text="Recommended Songs:", font=("Helvetica", 14), bg="#f0f0f0")
song_label.pack(pady=10)

# Create a Treeview to display recommended songs
columns = ("name", "artist", "mood", "popularity")
tree = ttk.Treeview(song_frame, columns=columns, show="headings")
tree.pack(pady=10)

# Define column headings
for col in columns:
    tree.heading(col, text=col.capitalize())
    tree.column(col, width=100)

# Create a frame for the breathing exercise
exercise_frame = Frame(main_frame, bg="#f0f0f0", relief="groove", bd=2)
exercise_frame.grid(row=2, column=1, padx=10, pady=10, sticky="n")

# Create a label to display breathing exercise instructions
exercise_label = Label(exercise_frame,
                       text="Breathing Exercise:\n1. Inhale slowly for 4 seconds.\n2. Hold your breath for 7 seconds.\n3. Exhale slowly for 8 seconds.\n4. Repeat for a few minutes.",
                       font=("Helvetica", 14), bg="#f0f0f0", wraplength=200)
exercise_label.pack(pady=10)
exercise_frame.grid_remove()  # Hide the exercise frame initially


# Capture image from webcam and predict
def live_emotion_recognition():
    cap = cv2.VideoCapture(0)  # Open the webcam (0 is the default ID for built-in webcam)

    if not cap.isOpened():
        decision_label.set("Error: Could not open webcam")
        return

    label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    emotion_counter = Counter()
    start_time = time.time()

    def update_frame():
        nonlocal start_time, emotion_counter
        ret, frame = cap.read()
        if not ret:
            decision_label.set("Error: Failed to capture image")
            root.after(10, update_frame)
            return

        # Process and predict the captured frame
        predictions, face_coords = predict_image(model, frame)
        if predictions is not None:
            predicted_class = np.argmax(predictions, axis=1)
            emotion = label[predicted_class[0]]
            emotion_counter[emotion] += 1

            # Draw rectangle around the face and put the emotion label
            (x, y, w, h) = face_coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Convert the frame to ImageTk format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        video_label.configure(image=img)
        video_label.image = img

        # Check if 10 seconds have passed
        elapsed_time = time.time() - start_time
        if elapsed_time >= 10:
            make_decision()
            start_time = time.time()
            emotion_counter = Counter()

        # Schedule the next frame update
        root.after(10, update_frame)

    def make_decision():
        if emotion_counter:
            most_common_emotion, count = emotion_counter.most_common(1)[0]
            if most_common_emotion in ['happy', 'neutral']:
                decision = "Driver is happy or neutral. Everything is normal."
                color = "green"
                value = 10
                exercise_frame.grid_remove()  # Hide the exercise frame
            elif most_common_emotion in ['sad', 'fear', 'disgust']:
                decision = "Implementing calming measures..."
                color = "yellow"
                value = 0
                exercise_frame.grid_remove()  # Hide the exercise frame
            elif most_common_emotion in ['angry', 'surprise']:
                decision = "Driver is surprised or angry. Stay alert."
                color = "red"
                value = 3
                exercise_frame.grid()  # Show the exercise frame
            else:
                decision = "Unknown emotional state. Proceed with caution."
                color = "yellow"
                value = 0
                exercise_frame.grid_remove()  # Hide the exercise frame

            decision_label.set(f"Most common emotion: {most_common_emotion} ({count} times). Decision: {decision}")
            emotion_bar['value'] = value
            style_name = set_progressbar_color("TProgressbar", color)
            emotion_bar.config(style=style_name)

            # Recommend songs
            songs = Recommend_Songs(most_common_emotion)
            for row in tree.get_children():
                tree.delete(row)
            for _, song in songs.iterrows():
                tree.insert("", "end", values=(song["name"], song["artist"], song["mood"], song["popularity"]))

    # Start updating frames
    root.after(10, update_frame)


# Start the live emotion recognition in a separate thread
thread = threading.Thread(target=live_emotion_recognition)
thread.daemon = True
thread.start()

# Run the Tkinter main loopj
root.mainloop()
