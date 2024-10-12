import cv2
import os
import time
import json
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, storage, firestore
import face_recognition

# Initialize Firebase admin SDK
cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'baylinkimagerecognition.appspot.com'
})
bucket = storage.bucket()
db = firestore.client()

# Create a directory to store the captured face images
output_dir = "captured_faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# JSON file to store unique face embeddings
faces_data_file = "faces_data.json"

# JSON file to store uploaded face images
uploaded_faces_file = "uploaded_faces.json"

# Create the JSON files if they don't exist
if not os.path.exists(faces_data_file):
    with open(faces_data_file, 'w') as f:
        json.dump([], f)

if not os.path.exists(uploaded_faces_file):
    with open(uploaded_faces_file, 'w') as f:
        json.dump([], f)

# Function to load data from JSON
def load_json(file):
    """Load data from a JSON file."""
    with open(file, 'r') as f:
        return json.load(f)

# Function to save data to JSON
def save_json(file, data):
    """Save data to a JSON file."""
    with open(file, 'w') as f:
        json.dump(data, f)

# Load the initial unique faces data from the JSON file
unique_faces_data = load_json(faces_data_file)

# Load the initial uploaded faces data from the JSON file
uploaded_faces_data = load_json(uploaded_faces_file)

# Function to calculate Euclidean distance between face embeddings
def compare_faces(known_face_encodings, face_encoding, threshold=0.6):
    """Compare a face encoding with known face encodings using Euclidean distance."""
    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    return np.any(distances <= threshold)

def upload_to_firebase(directory):
    """Upload all saved face images and encodings to Firebase storage and Firestore."""
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") and filename not in uploaded_faces_data:
            file_path = os.path.join(directory, filename)

            # Upload image to Firebase Storage
            blob = bucket.blob(f"faces/{filename}")
            blob.upload_from_filename(file_path)
            print(f"Uploaded {filename} to Firebase Storage.")

            # Get the public URL of the uploaded image
            image_url = blob.public_url

            # Get the corresponding face encoding
            face_encoding = next((face["face_encoding"] for face in unique_faces_data if face["filename"] == filename), None)
            if face_encoding is not None:
                # Upload face encoding and image URL to Firestore
                doc_ref = db.collection('faces').document(filename)  # Create a document with the filename as ID
                doc_ref.set({
                    'encoding': face_encoding,
                    'url': image_url,  # Store the image URL
                    'timestamp': datetime.now().isoformat()
                })
                print(f"Uploaded encoding and URL for {filename} to Firestore.")

            # Update uploaded faces list
            uploaded_faces_data.append(filename)
            save_json(uploaded_faces_file, uploaded_faces_data)

# Start video capture
video = cv2.VideoCapture(0)

# Track time for Firebase upload
start_time = time.time()

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to RGB (face_recognition uses RGB instead of BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get face locations
    face_locations = face_recognition.face_locations(rgb_frame)

    # Display number of faces detected on the video
    num_faces = len(face_locations)
    cv2.putText(frame, f"Faces in frame: {num_faces}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    if not face_locations:
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Get face encodings (embeddings) for the detected faces
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Process each detected face
    for face_location, face_encoding in zip(face_locations, face_encodings):
        # Check if face encoding is unique
        known_face_encodings = [np.array(face_data["face_encoding"]) for face_data in unique_faces_data]
        
        if not compare_faces(known_face_encodings, face_encoding):
            # If the face is unique, add it to the list of unique faces
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = f"face_{timestamp}.jpg"
            face_data = {
                "face_encoding": face_encoding.tolist(),  # Store face encoding (128-dimensional vector)
                "timestamp": datetime.now().isoformat(),
                "filename":face_filename  # Store timestamp
            }
            unique_faces_data.append(face_data)

            # Save the face data to the JSON file
            save_json(faces_data_file, unique_faces_data)

            # Save the unique face as an image
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            face_filename = os.path.join(output_dir, f"face_{timestamp}.jpg")
            cv2.imwrite(face_filename, face_image)
            print(f"Saved new face: {face_filename}")

        # Display detected faces with bounding boxes
        cv2.imshow('Video', frame)
    
    # Check if 30 minutes have passed to upload to Firebase
    if time.time() - start_time >= 10:  # 1800 seconds = 30 minutes
        upload_to_firebase(output_dir)
        start_time = time.time()  # Reset the timer
    
    # Check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
