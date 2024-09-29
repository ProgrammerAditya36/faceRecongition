import cv2
import json
import os
import numpy as np

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to capture the face and return face encoding (flattened grayscale values)
def capture_face():
    # Start the webcam video capture
    video_capture = cv2.VideoCapture(0)

    print("Please position your face in front of the camera...")

    while True:
        # Capture a single frame of video
        ret, frame = video_capture.read()

        # Convert to grayscale for better detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using Haarcascade
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If at least one face is detected
        if len(faces) > 0:
            # Just use the first detected face
            (x, y, w, h) = faces[0]

            # Extract the region of interest (ROI) for the face
            face_roi = gray_frame[y:y+h, x:x+w]

            # Resize the face to a standard size (e.g., 100x100)
            face_resized = cv2.resize(face_roi, (100, 100))

            # Flatten the 2D face matrix into a 1D array to save as encoding
            face_encoding = face_resized.flatten()

            print("Face detected and captured successfully.")
            
            # Show the detected face for feedback
            cv2.imshow('Detected Face', face_resized)

            # Release the webcam and close windows
            video_capture.release()
            cv2.destroyAllWindows()

            return face_encoding

        # Display the video feed with OpenCV to show real-time feedback
        cv2.imshow('Video', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows if the user quits
    video_capture.release()
    cv2.destroyAllWindows()

    return None


# Function to save the face encoding to a JSON file
def save_face_encoding(name, face_encoding, file_path='face_encodings.json'):
    # Check if the JSON file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)  # Load existing data
    else:
        data = {}

    # Add the new name and face encoding to the dictionary
    data[name] = face_encoding.tolist()

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Face encoding for {name} has been saved to {file_path}.")


# Main program to capture user input and face encoding
def main():
    # Ask the user for their name
    name = input("Enter your name: ")

    # Capture the face encoding
    face_encoding = capture_face()

    if face_encoding is not None:
        # Save the face encoding to a JSON file
        save_face_encoding(name, face_encoding)
    else:
        print("Face not captured. Please try again.")


if __name__ == "__main__":
    main()
