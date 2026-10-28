# Virtual Try-On Application

## Description

Virtual Try-On is a web application that allows users to upload a person image and a garment image, generate a virtual try-on result, download it, and save it to a personal wardrobe.

The application uses:

- HTML, CSS and JavaScript for the frontend;
- Python and Flask for the backend;
- SQLite for data storage;
- CP-VTON+ for image generation.

## Project Structure

```text
VirtualTryOn/
├── backend/
│   ├── app.py
│   ├── test_app.py
│   ├── virtual_tryon.db
│   ├── uploads/
│   └── output/
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── assets/
├── cp-vton-plus/
├── pyproject.toml
├── README.md
└── RUN_GUIDE.md
```

## Requirements

- Python 3.11;
- modern web browser;
- CP-VTON+ model files.

## Installation

Open a terminal in the project root:

```bash
cd VirtualTryOn
```

Create a virtual environment.

### macOS / Linux

```bash
python3 -m venv backend/venv
source backend/venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv backend\venv
backend\venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv backend\venv
backend\venv\Scripts\activate.bat
```

Install all dependencies from `pyproject.toml`:

```bash
python -m pip install ".[all]"
```

## Running the Application

### Backend — macOS / Linux

```bash
cd VirtualTryOn
source backend/venv/bin/activate
python3 backend/app.py
```

### Backend — Windows PowerShell

```powershell
cd VirtualTryOn
backend\venv\Scripts\Activate.ps1
python backend\app.py
```

### Backend — Windows Command Prompt

```cmd
cd VirtualTryOn
backend\venv\Scripts\activate.bat
python backend\app.py
```

The backend runs at:

```text
http://127.0.0.1:5051
```

### Frontend — macOS / Linux

Open a second terminal:

```bash
cd VirtualTryOn/frontend
python3 -m http.server 5500 --bind 127.0.0.1
```

### Frontend — Windows

```powershell
cd VirtualTryOn\frontend
python -m http.server 5500 --bind 127.0.0.1
```

Open the application in a browser:

```text
http://127.0.0.1:5500/index.html
```

## How to Use

1. Upload a person image.
2. Upload a garment image.
3. Optionally enter a store link.
4. Click `Generate`.
5. Download the result or log in and save it to the wardrobe.

## Tests

Run the tests from the project root.

### macOS / Linux

```bash
source backend/venv/bin/activate
python -m pytest backend/test_app.py -v
```

### Windows PowerShell

```powershell
backend\venv\Scripts\Activate.ps1
python -m pytest backend\test_app.py -v
```


## Stop the Servers

Press:

```text
Ctrl + C
```


## Limitations

- the application currently runs locally;
- CPU processing can be slow;
- result quality depends on the input images.

## Author

Zinaida Esir  
Semester project, 2026