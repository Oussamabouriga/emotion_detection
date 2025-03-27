import numpy as np
import cv2
from tensorflow.keras.models import load_model
import json

def detect_emotion(frame, model):
    """
    Detects faces in the frame and predicts the emotion for each detected face.
    Annotates the frame with a rectangle and the emotion label.
    """
    # Convert frame for processing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}
    
    # For each detected face, predict the emotion
    for (x, y, w, h) in faces:
        # Crop and resize face region for the model
        roi_color = image_rgb[y:y+h, x:x+w]
        resized_img = cv2.resize(roi_color, (48, 48))
        resized_img = np.expand_dims(resized_img, axis=0)
        resized_img = resized_img / 255.0
        
        prediction = model.predict(resized_img)
        max_index = np.argmax(prediction)
        emotion = emotion_dict[max_index]
        
        # Annotate the frame with a rectangle and emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # (Optional) Create a JSON representation of the emotion
        emotion_json = json.dumps({'Emotion': emotion})
        print(emotion_json)
    
    return frame

def main():
    # Load the pre-trained model once
    model = load_model('model.h5')
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        
        # Process frame to detect and annotate emotion
        processed_frame = detect_emotion(frame, model)
        
        # Display the resulting frame
        cv2.imshow('Emotion Detection', processed_frame)
        
        # Press 'q' to exit the video loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
