import cv2
import numpy as np

def faceBox(faceNet, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net = faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox = []   
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            box = detection[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, x1, y1) = box.astype("int")
            bbox.append((x, y, x1, y1))
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), int(round(frame.shape[0]/150)), 8)
    return frame, bbox

# Paths to model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Age and gender lists
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

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
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Predict gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        
        # Predict age
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
        
        # Display gender and age on the frame
        cv2.putText(frame, f"{gender}, {age}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the frame with face boxes and additional information
    cv2.imshow('Video', frame)
    
    # Check if the 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
