import cv2
import json
import os
import numpy as np

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to load the stored face encodings and names from the JSON file
def load_encodings(file_path='face_encodings.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)  # Load face encodings from the JSON file
        return data
    else:
        print(f"No file named '{file_path}' found.")
        return {}


# Function to calculate the Euclidean distance between two face encodings
def compare_faces(known_encoding, test_encoding):
    return np.linalg.norm(np.array(known_encoding) - np.array(test_encoding))


# Function to recognize the face from the webcam input
def recognize_face(known_faces, threshold=3000.0):
    # Start the webcam video capture
    video_capture = cv2.VideoCapture(0)

    print("Position your face in front of the camera...")

    while True:
        # Capture a single frame of video
        ret, frame = video_capture.read()

        # Convert to grayscale for better detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using Haarcascade
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If a face is detected
        if len(faces) > 0:
            # Use the first detected face
            (x, y, w, h) = faces[0]

            # Extract the region of interest (ROI) for the face
            face_roi = gray_frame[y:y+h, x:x+w]

            # Resize the face to a standard size (100x100) - matching the format used in input_data.py
            face_resized = cv2.resize(face_roi, (100, 100))

            # Flatten the 2D face matrix into a 1D array to match the format in JSON
            face_encoding = face_resized.flatten()

            # Initialize variables to keep track of the best match
            best_match_name = "Unknown"
            best_match_distance = float('inf')

            # Compare this face with known faces
            for name, known_encoding in known_faces.items():
                distance = compare_faces(known_encoding, face_encoding)
                if distance < best_match_distance:
                    best_match_distance = distance
                    best_match_name = name

            # Check if the best match is below the threshold
            

            # Display the frame with a bounding box and name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, best_match_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    # Load the stored face encodings
    known_faces = load_encodings()

    if known_faces:
        # Start the face recognition process
        recognize_face(known_faces)
    else:
        print("No known faces to compare.")


if __name__ == "__main__":
    main()
