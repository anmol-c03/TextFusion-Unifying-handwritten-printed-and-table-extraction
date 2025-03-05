# Document OCR Pipeline

A comprehensive document OCR pipeline that  extacts everything from the given document independent of layout and structure. It combines document layout analysis, text detection, text recognition, and text correction to extract text as well as tables from documents containing both printed and handwritten text.
This project implements modular approach such that it can be used with minimum compute time and resources.

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

## System Pipeline

![Layout Independent Extraction Pipeline](https://github.com/anmol-c03/Structured_handwritten_data_extraction/blob/main/images/project_pipeline/system_pipeline.png)


# Results

<!-- 1. Survey Form without Tables:
![NO Tables](https://github.com/anmol-c03/Structured_handwritten_data_extraction/blob/main/results/survey_results.png)

2. Survey Form with Table:
![With  Tables](https://github.com/anmol-c03/Structured_handwritten_data_extraction/blob/main/results/surveyform_withtables.png)

3. Table Only
![Marksheet](https://github.com/anmol-c03/Structured_handwritten_data_extraction/blob/main/results/table_as_csv.png) -->
<p align="center">
  <img src="https://github.com/anmol-c03/Structured_handwritten_data_extraction/blob/main/results/images_wo_table/survey_results.png" width="45%" alt="NO Tables">
  <img src="https://github.com/anmol-c03/Structured_handwritten_data_extraction/blob/main/results/images_with_table/surveyform_withtables.png" width="45%" alt="With Tables">
  <br>  <!-- Line break to move next two images below -->
  <img src="https://github.com/anmol-c03/Structured_handwritten_data_extraction/blob/main/results/Table_only/table_as_csv.png" width="45%" alt="Marksheet">
  <img src="https://github.com/anmol-c03/Structured_handwritten_data_extraction/blob/main/results/Table_only/download_csv.jpg" width="45%" alt="Fourth Image">
</p>



## Installation

1. Clone the repository:
```bash
git clone https://github.com/anmol-c03/Structured_handwritten_data_extraction.git
cd YoloTrOCR
```

2. Install dependencies:
```bash
chmod +x ./install.sh
./install.sh
```



## Usage

```bash
!python main.py --input /path/to/your/pdf --output txt 

### options

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
Structured_handwritten_data_extraction/                      # Project root
├── images/  
│   ├── original/
│   ├── resized/
│   └── extract_pdf/                 # Folder for images
├── model_doclayout/                 # Model-related files for layout processor
├── models/                          # Folder for storing YOLO model for text detection
|    |_bestline.pt                         
├── processors/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── layout_processor.py
│   ├── text_processor.py
│   ├── text_recognition.py
│   └── correction_processor.py                      # Processing scripts
├── Table_extraction/                 # Table extraction module
│   ├── images/                       # Subfolder for table images
│   ├── __init__.py                   # Init file
│   ├── cell_coordinates.py           # Table cell detection script
│   ├── crop_table.py                 # Table cropping script
│   ├── main.py                       # Main script for table extraction
│   ├── ocr.py                        # OCR processing script
│   └── preprocess.py                 # Preprocessing script
├── txt/                              # Folder for extracted text files
│   ├── txt.1                         
│   ├── txt.2                         
│   ├── txt.3                         
├── utils/
│   ├── __init__.py
│   └── file_utils.py                          # Utility functions
├── .gitignore                        # Git ignore file
├── install.sh                        # Installation script
├── main_for_api.py                   # Main script for API integration
├── main.py                           # Main script
├── Readme.md                         # Project documentation
└── requirements.txt                   # Dependencies list


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


## Contributors

For backend , please refer to the link  
Backend Service ([Backend Link](https://github.com/ashim-karki/StructuredHandwrittenDataExtraction-Backend))
