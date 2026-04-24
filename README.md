# Indian License Plate Detection + OCR

Basic steps for teammates to run this project.

## 1. Clone Repo

```bash
git clone <your-repo-url>
cd MAJOR
```

## 2. Create and Activate Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Download Required Files Locally

- Dataset folders: `dataset/train`, `dataset/valid`, `dataset/test`
- Model weight file: `runs/detect/train4/weights/best.pt`

Dataset link: https://app.roboflow.com/aryans-workspace-inqhr/indian-license-plate-knte7-l6agk/1

Use project metadata in `dataset/data.yaml` and `dataset/README.roboflow.txt`.

## 5. Run

### Single image OCR

```bash
python scripts/image_ocr.py
```

### Webcam OCR

```bash
python scripts/ocr_pipeline.py
```

Press `Esc` to exit webcam window.
