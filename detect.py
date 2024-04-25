import tensorflow as tf

# Define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the TensorFlow model
model_path = r'emotion_model.h5'
model = tf.keras.models.load_model(model_path)


import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Resize the frame
    frame = cv2.resize(frame, (1280, 720))
    
    # Check if the frame was successfully read
    if not ret:
        print("Error reading frame")
        break
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in num_faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y-70), (x+w, y+h+18), (255,255, 102), 4)
        
        # Extract the region of interest (ROI) for the face
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotion for the ROI
        emotion_prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        
        # Display the predicted emotion as text
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (254, 84, 102), 2, cv2.LINE_AA)

    # Display the frame with bounding boxes and predicted emotions
    cv2.imshow('Emotion Detection', frame)
    
    # Check for key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()