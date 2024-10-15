import cv2
import requests
import numpy as np
import json
from datetime import datetime
import base64

API_ENDPOINT = "http://34.172.8.251/process_face"

# Start video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get face locations
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_locations = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5)

    # Display number of faces detected on the video
    num_faces = len(face_locations)
    cv2.putText(frame, f"Faces in frame: {num_faces}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in face_locations:
        # Extract face image
        face_image = rgb_frame[y:y+h, x:x+w]
        
        # Encode image to send over HTTP
        _, img_encoded = cv2.imencode('.jpg', face_image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Prepare data for API request
        data = {
            'image': img_base64,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send POST request to server
        try:
            response = requests.post(API_ENDPOINT, json=data)
            
            # Process server response
            if response.status_code == 200:
                result = response.json()
                if result.get('is_unique', False):
                    print(f"New unique face detected: {result['filename']}")
            else:
                print(f"Error: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending request: {e}")
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()