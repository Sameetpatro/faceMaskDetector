import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
import datetime

# Load the pre-trained model
try:
    mymodel = load_model('mask_detector_model.h5')
    print("Model loaded successfully!")
except:
    print("Error: Could not load mask_detector_model.h5")
    print("Make sure the file exists in the current directory")
    exit()

# Load Haar Cascade for face detection
# Try multiple possible file names
cascade_files = ['face_detector.xml', 'haarcascade_frontalface_default.xml']
face_cascade = None

for cascade_file in cascade_files:
    try:
        face_cascade = cv2.CascadeClassifier(cascade_file)
        if not face_cascade.empty():
            print(f"Loaded face detector: {cascade_file}")
            break
    except:
        continue

# If custom cascade not found, try OpenCV's built-in
if face_cascade is None or face_cascade.empty():
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("Loaded built-in face detector")
    except:
        print("Error: Could not load face detector")
        exit()

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Starting face mask detection...")
print("Press 'q' to quit")

while cap.isOpened():
    ret, img = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = img[y:y+h, x:x+w]
        
        # Save temporarily and load for prediction
        cv2.imwrite('temp.jpg', face_img)
        test_image = image.load_img('temp.jpg', target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize
        
        # Predict
        pred = mymodel.predict(test_image, verbose=0)[0][0]
        
        # Draw rectangle and label
        if pred > 0.5:  # No Mask
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(img, 'NO MASK', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:  # Mask
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img, 'MASK', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Add timestamp
    datet = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cv2.putText(img, datet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show frame
    cv2.imshow('Face Mask Detection', img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Detection stopped")