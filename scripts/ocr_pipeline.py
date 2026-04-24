from ultralytics import YOLO
import easyocr
import cv2

# Load model
model = YOLO("runs/detect/train4/weights/best.pt")

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=True)

# Load image
image_path = "dataset/test/images"  # you can also use single image
cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = box

        if score > 0.5:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Crop plate
            plate = frame[y1:y2, x1:x2]

            # OCR
            text = reader.readtext(plate)

            detected_text = ""
            if len(text) > 0:
                detected_text = text[0][1]

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, detected_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("ANPR", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()