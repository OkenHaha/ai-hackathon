import cv2
import numpy as np

# Label mappings
age_map = {
    '0-2': 0, '3-9': 1, '10-19': 2, '20-29': 3,
    '30-39': 4, '40-49': 5, '50-59': 6, '60-69': 7,
    'more than 70': 8
}
gender_map = {"Male": 0, "Female": 1}
race_map = {
    "East Asian": 0,
    "Indian": 1,
    "Southeast Asian": 2
}


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


net = cv2.dnn.readNet('model.onnx')

# Preprocessing parameters matching training pipeline
input_size = (224, 224)
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype(np.float32)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype(np.float32)

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    frame_resized = cv2.resize(frame, input_size)
    frame_normalized = (frame_resized.astype(np.float32) / 255.0 - mean) / std
    frame_transposed = frame_normalized.transpose(2, 0, 1).reshape(1, 3, *input_size)
    return frame_transposed

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

def detect_and_classify(frame):
    """Detect faces and classify race"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess face
        blob = preprocess_frame(face_roi)
        
        # Model inference
        net.setInput(blob)
        outputs = net.forward(['age_out', 'gender_out', 'race_out'])
        
        # Extract logits
        age_logits = outputs[0][0]
        gender_logits = outputs[1][0]
        race_logits = outputs[2][0]
        
        # Compute softmax probabilities
        age_probs = softmax(age_logits)
        gender_probs = softmax(gender_logits)
        race_probs = softmax(race_logits)
        
        # Invert maps for label lookup
        inverse_age_map = {v: k for k, v in age_map.items()}
        inverse_gender_map = {v: k for k, v in gender_map.items()}
        inverse_race_map = {v: k for k, v in race_map.items()}
        
        # Get predicted labels
        age_pred = inverse_age_map[np.argmax(age_probs)]
        gender_pred = inverse_gender_map[np.argmax(gender_probs)]
        race_pred = inverse_race_map[np.argmax(race_probs)]
        
        # Remap race to locality
        #locality = "non-local" if race_pred == "Indian" else "local"
        
        # Draw bounding box and labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Race: {race_pred}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    
    return frame

# Start real-time video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = detect_and_classify(frame)
        
        cv2.imshow('Real-time Classification', processed_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()