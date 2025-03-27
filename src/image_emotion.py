import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import json

def detect_emotion(image_path):

    model = load_model('model.h5')

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image_rgb[y:y+h, x:x+w]
        
        resized_img = cv2.resize(roi_color, (48, 48))
        resized_img = np.expand_dims(resized_img, axis=0)   
        resized_img = resized_img / 255.0
        
        prediction = model.predict(resized_img)
        max_index = np.argmax(prediction)
        emotion = emotion_dict[max_index]
        
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image_rgb, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    emotion_dict = {'Emotion': emotion}
    emotion_json = json.dumps(emotion_dict)

    return emotion_json


def main():
    image_path = '../assets/images/nFace.jpg'
    emotion_json = detect_emotion(image_path)
    print(emotion_json)

if __name__ == "__main__":
    main()
