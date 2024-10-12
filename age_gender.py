import cv2
import numpy as np
from tensorflow.keras.models import load_model #type: ignore

def faceBox(faceNet, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox = []   
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence >= 0.6:
            box = detection[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, x1, y1) = box.astype("int")
            bbox.append((x, y, x1, y1))
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), int(round(frame.shape[0]/150)), 8)
    return frame, bbox

# Paths to model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

# Load face detection model
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Load your TensorFlow model
model = load_model('age_gender_detection.h5')

# Gender and age dictionaries (customize based on your model's output)
gender_dict = {0: 'Male', 1: 'Female'}

# Start video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Detect faces and get bounding boxes
    frame, bboxs = faceBox(faceNet, frame)

    # Display number of faces detected on the video
    num_faces = len(bboxs)
    cv2.putText(frame, f"Faces in frame: {num_faces}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    if not bboxs:
        print("No face Detected, Checking next frame")
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Process each detected face
    for bbox in bboxs:
        face = frame[max(0, bbox[1]):min(bbox[3], frame.shape[0]), max(0, bbox[0]):min(bbox[2], frame.shape[1])]
        
        # Preprocess the face for the TensorFlow model
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face_resized = cv2.resize(face_gray, (128, 128))  # Resize to 128x128
        face_resized = face_resized.reshape(1, 128, 128, 1)  # Reshape to match model's input shape
        face_resized = face_resized / 255.0  # Normalize pixel values
        
        # Make predictions with the TensorFlow model
        pred = model.predict(face_resized)
        
        # Assuming gender prediction is a softmax with two classes (Male/Female), use np.argmax to find the class
        pred_gender = gender_dict[np.argmax(pred[0])]  # Gender is predicted in the first output
        
        # For age prediction, it's usually a scalar, so just access it directly
        pred_age = (pred[1][0])  # Age is predicted in the second output
        
        # Display the predicted gender and age
        cv2.putText(frame, f"{pred_gender}, {pred_age} years", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the frame with face boxes and additional information
    cv2.imshow('Video', frame)
    
    # Check if the 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
