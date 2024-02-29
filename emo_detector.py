
import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the saved model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Load the image
frame_img = r"C:\Users\cheru\OneDrive\Desktop\michael-dam-mEZ3PoFGs_k-unsplash.jpg"
frame = cv2.imread(frame_img)

# Calculate the scale factor to resize the image while maintaining aspect ratio
desired_height = 720
scale_factor = desired_height / frame.shape[0]

# Resize the image with the calculated scale factor
frame_resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

# Find faces in the resized image and perform emotion detection
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in num_faces:
    cv2.rectangle(frame_resized, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
    roi_gray_frame = gray_frame[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

    # Predict emotions
    emotion_prediction = emotion_model.predict(cropped_img)
    maxindex = int(np.argmax(emotion_prediction))
    cv2.putText(frame_resized, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Display the resized image with detected emotions
cv2.imshow('Emotion Detection', frame_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
