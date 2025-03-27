import asyncio
import websockets
import json
import os
import base64
import io
import face_recognition
from image_emotion import detect_emotion
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


firstPicture = face_recognition.load_image_file("../assets/images/face1.jpg")
firstPicture_encoding = face_recognition.face_encodings(firstPicture)[0]

async def websocket_handler(websocket, path):
    try:
        if path == "/save_image":
            async for message in websocket:
                response = await save_image(message)
                await websocket.send(json.dumps(response))
        elif path == "/recognize_face":
            async for message in websocket:
                response = recognize_face(message)
                await websocket.send(json.dumps(response))
        else:
            await websocket.send(json.dumps({"status": False, "message": "Invalid path"}))
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")

async def save_image(message):
    try:
        data = json.loads(message)
        image_data = base64.b64decode(data["image"])
        emotion = data.get("emotion", "neutral")
        image_number = data.get("imageNumber", 1)
        done = data.get("done", False)

        dir_path = f"user/{emotion}"
        os.makedirs(dir_path, exist_ok=True)
        image_path = os.path.join(dir_path, f"image_{image_number}.jpg")

        with open(image_path, "wb") as f:
            f.write(image_data)
        
        if done:
            print("Training model")
            model = create_model()
            train_generator, validation_generator = get_data_generators('user', 'data/test', 64)
            history = train_model(model, train_generator, validation_generator, 28709, 7178, 64, 50)
            model.save('userModel.h5')
            #plot_model_history(history)
        
        return {"status": True, "message": "Image saved successfully", "image_path": image_path}

    except Exception as e:
        return {"status": False, "message": str(e)}

def recognize_face(message):
    try:
        secondPicture = face_recognition.load_image_file(io.BytesIO(message))
        secondPicture_encodings = face_recognition.face_encodings(secondPicture)
        if len(secondPicture_encodings) > 0:
            secondPicture_encoding = secondPicture_encodings[0]
        else:
            return {"status": True, "message": "No Face Detected", "data": 0}
        results = face_recognition.compare_faces([firstPicture_encoding], secondPicture_encoding)
        
        with open("temp.jpg", "wb") as f:
            f.write(message)

        emotion_json = detect_emotion("temp.jpg")

        if results[0] == True:
            return {"status": True, "message": "Recognition successful", "data": 2, "emotion": emotion_json}
        else:
            return {"status": True, "message": "Recognition unsuccessful", "data": 1, "emotion": emotion_json}

    except Exception as e:
        return {"status": False, "message": str(e)}


def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 3)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])
    return model

def get_data_generators(train_dir, val_dir, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(48,48), batch_size=batch_size, color_mode="rgb", class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            val_dir, target_size=(48,48), batch_size=batch_size, color_mode="rgb", class_mode='categorical')

    return train_generator, validation_generator

def train_model(model, train_generator, validation_generator, num_train, num_val, batch_size, num_epoch):
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
        ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size,
        callbacks=callbacks
    )
    return history

def plot_model_history(history, plot_path='plot.png'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(history.history['accuracy'], label='train accuracy')
    axs[0].plot(history.history['val_accuracy'], label='validation accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='best')
    axs[1].plot(history.history['loss'], label='train loss')
    axs[1].plot(history.history['val_loss'], label='validation loss')
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='best')
    plt.savefig(plot_path)
    plt.show()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    print("Server started running...")
    loop.run_until_complete(
        websockets.serve(websocket_handler, "0.0.0.0", 8765)
    )
    loop.run_forever()
