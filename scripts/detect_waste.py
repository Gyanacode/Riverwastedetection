import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/river_waste_model.h5')

# Start video capture from webcam or camera module
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame for prediction (resize and normalize)
    img = cv2.resize(frame, (224, 224))
    img_array = np.array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Display prediction result on frame (you may want to map class index to actual labels)
    cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Show the output frame
    cv2.imshow('River Waste Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()