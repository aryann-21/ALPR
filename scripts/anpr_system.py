from ultralytics import YOLO
import easyocr
import cv2
import re
import os
from collections import Counter

# =========================
# LOAD MODEL + OCR
# =========================

model = YOLO("runs/detect/train4/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# =========================
# OCR MEMORY FOR VIDEO VOTING
# =========================

ocr_votes = Counter()

# =========================
# CLEAN TEXT
# =========================

def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

# =========================
# OCR CHARACTER CORRECTIONS
# =========================

# Indian format:
# AA00AA0000

# Position wise correction:
# letters section -> convert numbers to letters
# numbers section -> convert letters to numbers


def correct_plate(text):

    text = list(text)

    # Common OCR mistakes
    num_to_char = {
        '0': 'O',
        '1': 'I',
        '5': 'S',
        '8': 'B'
    }

    char_to_num = {
        'O': '0',
        'I': '1',
        'Z': '2',
        'S': '5',
        'B': '8'
    }

    # State code letters
    for i in [0, 1]:
        if i < len(text) and text[i] in num_to_char:
            text[i] = num_to_char[text[i]]

    # District numbers
    for i in [2, 3]:
        if i < len(text) and text[i] in char_to_num:
            text[i] = char_to_num[text[i]]

    # Series letters
    for i in [4, 5]:
        if i < len(text) and text[i] in num_to_char:
            text[i] = num_to_char[text[i]]

    # Last 4 digits
    for i in range(6, len(text)):
        if i < len(text) and text[i] in char_to_num:
            text[i] = char_to_num[text[i]]

    return ''.join(text)

# =========================
# VALIDATE INDIAN PLATE
# =========================


def is_valid_plate(text):

    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'

    return re.match(pattern, text) is not None

# =========================
# PREPROCESS PLATE IMAGE
# =========================


def preprocess_plate(plate):

    # Resize
    plate = cv2.resize(plate, None, fx=2, fy=2)

    # Grayscale
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Denoise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold
    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return thresh

# =========================
# PROCESS FRAME
# =========================


def process_frame(frame, detected_plates):

    results = model(frame)[0]

    for box in results.boxes.data.tolist():

        x1, y1, x2, y2, score, cls = box

        if score > 0.5:

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Crop plate
            plate = frame[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            # =========================
            # PREPROCESS BEFORE OCR
            # =========================

            processed_plate = preprocess_plate(plate)

            # OCR
            ocr_result = reader.readtext(processed_plate)

            if len(ocr_result) > 0:

                text = clean_text(ocr_result[0][1])

                # =========================
                # OCR CORRECTIONS
                # =========================

                text = correct_plate(text)

                # =========================
                # VALIDATION
                # =========================

                if is_valid_plate(text):

                    # =========================
                    # FRAME VOTING
                    # =========================

                    ocr_votes[text] += 1

                    detected_plates.add(text)

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Draw text
                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

    return frame

# =========================
# IMAGE MODE
# =========================


def process_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Image not found!")
        return

    detected_plates = set()

    output = process_frame(img, detected_plates)

    print("Detected Plates:")

    if len(detected_plates) == 0:
        print("No valid plates detected")

    else:
        for plate in sorted(detected_plates):
            print(plate)

    # Save output image
    os.makedirs("output", exist_ok=True)

    output_path = "output/result_image.jpg"

    cv2.imwrite(output_path, output)

    print(f"Output image saved to: {output_path}")

# =========================
# VIDEO MODE
# =========================


def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video")
        return

    detected_plates = set()

    frame_count = 0

    last_output_frame = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Process every 3rd frame
        if frame_count % 3 != 0:
            continue

        output = process_frame(frame, detected_plates)

        last_output_frame = output

    cap.release()

     # =========================
    # FINAL OUTPUT USING VOTING
    # =========================

    print("Final Detected Vehicle Numbers:")

    final_results = []

    for plate, count in ocr_votes.items():

        # Keep only plates seen multiple times
        if count >= 2:
            final_results.append((plate, count))

    if len(final_results) == 0:
        print("No valid plates detected")

    else:

        final_results.sort(key=lambda x: x[1], reverse=True)

        for i, (plate, count) in enumerate(final_results, start=1):
            print(f"{i}. {plate}  (Detected {count} times)")

    # =========================
    # SAVE RESULTS
    # =========================

    os.makedirs("output", exist_ok=True)

    with open("output/detected_plates.txt", "w") as f:

        for plate, count in final_results:
            f.write(f"{plate} | Count: {count}\n")

    if last_output_frame is not None:

        cv2.imwrite(
            "output/video_result.jpg",
            last_output_frame
        )

    print("Results saved to output/detected_plates.txt")
    print("Output frame saved to output/video_result.jpg")

# =========================
# MAIN MENU
# =========================

print("===== ANPR SYSTEM =====")
print("1. Process Image")
print("2. Process Video")

choice = input("Enter choice (1 or 2): ")

if choice == '1':

    image_path = input("Enter image path: ")

    process_image(image_path)

elif choice == '2':

    video_path = input("Enter video path: ")

    process_video(video_path)

else:
    print("Invalid choice")