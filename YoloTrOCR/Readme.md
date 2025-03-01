# Document OCR Pipeline

A comprehensive document OCR pipeline that combines document layout analysis, text detection, text recognition, and text correction to extract text from documents containing both printed and handwritten text.

## Features

- PDF to image conversion
- Document layout analysis using DocLayout YOLO
- Text detection for both printed and handwritten text
- Text recognition using PaddleOCR for printed text and TrOCR for handwritten text
- Automatic classification of text as printed or handwritten using Token granularity and confidence score
- Minimum use of TrOCR for low compute time and resources 
- Completely secure if integrated even with proprietary software/LLMs
- Text correction using language tools
- Modular and extensible architecture

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/document-ocr-pipeline.git
cd YoloTrOCR
```

2. Install dependencies:
```bash
chmod +x ./install.sh
./install.sh
```



## Usage

```bash
!python main.py --input survey_form_4.pdf --output txt 


- `--input`: Path to input PDF or image file (required)
- `--output`: Path to output text folder (default: results)
- `--model_path`: Path to DocLayout YOLO model (default: ./model_doclayout/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt)
- `--dpi`: DPI for PDF conversion (default: 300)
- `--confidence`: Confidence threshold for text detection (default: 0.5)
- `--clean`: Clean temp directories before processing
- `--visualize`: Visualize detected regions



```

## Project Structure

```bash
document-ocr-pipeline/
├── main.py               # Main script
├── setup.py              # **Setup script for packaging**
├── processors/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── layout_processor.py
│   ├── text_processor.py
│   ├── text_recognition.py
│   └── correction_processor.py
├── utils/
│   ├── __init__.py
│   └── file_utils.py
├── images/
│   ├── original/
│   ├── resized/
│   └── extract_pdf/
├── txt/
|   |_txt.1
|   |_txt.2
|   |_txt.3.....
└── models
|    |_bestline.pt
|__ models_doclayout


```

## Example

```bash
# Process a PDF document
!python main.py --input survey_form_4.pdf --output txt --model_path ./model_doclayout/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt --clean

```

## Requirements

- Python 3.7+
- CUDA compatible GPU (recommended for faster processing)
- See requirements.txt and install.sh for required packages
