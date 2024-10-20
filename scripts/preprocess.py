import os
import cv2

# Define paths
dataset_path = 'data/training'  # Adjust as necessary
processed_path = 'data/processed'

# Create directory for processed images if it doesn't exist
if not os.path.exists(processed_path):
    os.makedirs(processed_path)

# Preprocessing function: Resize and normalize images
def preprocess_images():
    for subdir, _, files in os.walk(dataset_path):
        for file in files:
            try:
                img_path = os.path.join(subdir, file)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (224, 224))  # Resize to match model input
                image = image / 255.0  # Normalize image

                # Save preprocessed image
                processed_img_path = os.path.join(processed_path, file)
                cv2.imwrite(processed_img_path, image * 255)  # Multiply back for saving
            except Exception as e:
                print(f'Error processing {file}:', e)

# Call the function to preprocess images
preprocess_images()