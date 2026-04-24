from ultralytics import YOLO
import easyocr
import cv2
import re

# Load model
model = YOLO("runs/detect/train4/weights/best.pt")

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=True)

# Function to clean plate text
def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

# Input image
image_path = "images/car.jpg"

img = cv2.imread(image_path)

# Run detection
results = model(img)[0]

found = False

for box in results.boxes.data.tolist():
    x1, y1, x2, y2, score, cls = box

    if score > 0.5:
        found = True
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Crop plate
        plate = img[y1:y2, x1:x2]

        # OCR
        ocr_result = reader.readtext(plate)

        if len(ocr_result) > 0:
            text = clean_text(ocr_result[0][1])
            print("Detected Plate Number:", text)
        else:
            print("No text detected")

if not found:
    print("No plate detected")