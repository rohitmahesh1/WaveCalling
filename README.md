# WaveCalling

A simple toolkit to generate kymograph analyses from CSV or image files, extract wave traces, optionally upload results to Google Sheets, and save summary plots & data locally.

---

## 📋 Table of Contents

1. [What It Does](#what-it-does)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Setting Up WolframScript](#setting-up-wolframscript)  
5. [Configuration (Google Sheets)](#configuration-google-sheets)  
6. [Usage](#usage)  
7. [Troubleshooting](#troubleshooting)  

---

## What It Does

- **From CSV**: Converts a CSV of intensity values into a heatmap, runs KymoButler to extract wave traces, fits sinusoids, and outputs:
  - A summary CSV of wave characteristics  
  - Histograms of wave lengths & widths  
  - Individual sine‐regression plots  
- **From Image**: Preprocesses images via WolframScript, then detects & annotates colored lines, extracts traces, and outputs:
  - Annotated image  
  - A summary CSV of line characteristics  

Optional Google Sheets upload if you supply your own `client_secret.json`.

---

## Prerequisites

- **Python 3.8+** (Windows or macOS)  
- **WolframScript** (to run `.wls` scripts)  
- A **Google account** & a [service OAuth client](https://console.developers.google.com/) if you want Sheets integration

---

## Installation

### 1. Clone or download this repo

```bash
# Open Terminal (macOS) or PowerShell/Command Prompt (Windows) and run:
git clone https://github.com/rohitmahesh1/WaveCalling.git
cd WaveCalling
```

### 2. Create & activate a Python virtual environment

<details>
<summary>macOS / Linux</summary>

```bash
python3 -m venv waves
source waves/bin/activate
```

</details>

<details>
<summary>Windows (PowerShell)</summary>

```powershell
python -m venv waves
.\waves\Scripts\activate
```

</details>

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Setting Up WolframScript

1. **Download** the Wolfram Engine (free for development) from  
   https://www.wolfram.com/engine/  
2. **Install** and **log in** with your Wolfram ID.  
3. Ensure `wolframscript` is on your **PATH**:
   - **macOS**: usually `which wolframscript` → `/usr/local/bin/wolframscript`  
   - **Windows**: run `wolframscript` in PowerShell; if not found, add its install directory (e.g., `C:\Program Files\Wolfram Research\WolframEngine\12.3\`) to your PATH.

---

## Configuration (Google Sheets)

1. In the Google Cloud Console, create an **OAuth 2.0 Client ID** (Desktop App).  
2. Download the JSON (named e.g. `client_secret.json`) and place it in this project’s root folder.  
3. When you run with `--push_to_drive=True`, the script will look for `client_secret*.json` in the current directory.

---

## Usage

From your activated environment, use:

```bash
# Basic CSV processing, save everything locally:
python run.py path/to/data.csv

# Basic image processing:
python run.py path/to/image.png

# Include min trace length (default 30):
python run.py data.csv --min_trace_length=50

# Upload results to Google Sheets:
python run.py data.csv --push_to_drive=True
# (ensure your client_secret*.json is in this folder)

# Combined example:
python run.py heatmap.csv \
    --min_trace_length=40 \
    --push_to_drive=True
```

After running, you’ll find:

- **For CSV**:
  - `heatmap.png` (visual heatmap)  
  - `kymobutler_output/*.npy` (raw trace arrays)  
  - `*_wave_data.csv` (wave summary)  
  - `*_length_distribution.png`, `*_width_distribution.png`  

- **For Image**:
  - `proc/<image>_processed_2.png`  
  - `proc/sine_regression/*.png` (per‐line regression)  
  - `proc/sine_regression/<image>_annotated.png`  
  - `proc/<image>_line_data.csv`

---

## Troubleshooting

- **“command not found: wolframscript”**  
  Ensure WolframScript is installed and on your system PATH.
- **Google OAuth browser window won’t open**  
  Make sure you have a default browser set, or run on a machine with GUI.
- **Large file errors pushing to GitHub**  
  Ensure you’ve added `models/` and `waves/` to `.gitignore` and not committed them.