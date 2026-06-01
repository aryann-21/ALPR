from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import easyocr
import cv2
import re
import os
from collections import Counter
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# =========================
# LOAD MODEL + OCR
# =========================

model = YOLO("runs/detect/train4/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# =========================
# CLEAN TEXT
# =========================


def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

# =========================
# CORRECT OCR MISTAKES
# =========================


def correct_plate(text):

    text = list(text)

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

    for i in [0, 1]:
        if i < len(text) and text[i] in num_to_char:
            text[i] = num_to_char[text[i]]

    for i in [2, 3]:
        if i < len(text) and text[i] in char_to_num:
            text[i] = char_to_num[text[i]]

    for i in [4, 5]:
        if i < len(text) and text[i] in num_to_char:
            text[i] = num_to_char[text[i]]

    for i in range(6, len(text)):
        if i < len(text) and text[i] in char_to_num:
            text[i] = char_to_num[text[i]]

    return ''.join(text)

# =========================
# VALIDATE INDIAN PLATES
# =========================


def is_valid_plate(text):

    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'

    return re.match(pattern, text) is not None

# =========================
# PREPROCESS PLATE
# =========================


def preprocess_plate(plate):

    plate = cv2.resize(plate, None, fx=2, fy=2)

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

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

def process_frame(frame, detected_plates, ocr_votes):

    results = model(frame)[0]

    for box in results.boxes.data.tolist():

        x1, y1, x2, y2, score, cls = box

        if score > 0.5:

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            plate = frame[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            processed_plate = preprocess_plate(plate)

            ocr_result = reader.readtext(processed_plate)

            if len(ocr_result) > 0:

                text = clean_text(ocr_result[0][1])

                text = correct_plate(text)

                if is_valid_plate(text):

                    detected_plates.add(text)
                    ocr_votes[text] += 1

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

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
# IMAGE PROCESSING
# =========================


def process_image(filepath):

    detected_plates = set()
    ocr_votes = Counter()

    img = cv2.imread(filepath)

    output = process_frame(img, detected_plates, ocr_votes)

    result_filename = f'result_{uuid.uuid4().hex}.jpg'
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    cv2.imwrite(result_path, output)

    return sorted(detected_plates), result_path

# =========================
# VIDEO PROCESSING
# =========================

def process_video(filepath):

    cap = cv2.VideoCapture(filepath)

    detected_plates = set()
    ocr_votes = Counter()

    frame_count = 0

    last_frame = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 3 != 0:
            continue

        output = process_frame(frame, detected_plates, ocr_votes)

        last_frame = output

    cap.release()

    final_results = []

    for plate, count in ocr_votes.items():

        if count >= 2:
            final_results.append((plate, count))

    final_results.sort(key=lambda x: x[1], reverse=True)

    result_filename = f'video_{uuid.uuid4().hex}.jpg'
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    if last_frame is not None:
        cv2.imwrite(result_path, last_frame)

    plates = [plate for plate, count in final_results]

    return plates, result_path

# =========================
# MAIN ROUTE
# =========================


@app.route('/', methods=['GET', 'POST'])
def index():

    plates = []
    result_image = None

    if request.method == 'POST':

        file = request.files['file']

        if file:

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)

            file.save(filepath)

            ext = file.filename.split('.')[-1].lower()

            if ext in ['jpg', 'jpeg', 'png']:

                plates, result_image = process_image(filepath)

            elif ext in ['mp4', 'avi', 'mov']:

                plates, result_image = process_video(filepath)

    return render_template(
        'index.html',
        plates=plates,
        result_image=result_image
    )

# =========================
# RUN APP
# =========================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
