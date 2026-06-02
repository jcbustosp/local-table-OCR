# Table OCR

A modular, CPU-optimized Python pipeline for table structural parsing and text extraction built on top of **PaddleOCR (PP-Structure)**.

---

## Getting Started

This project uses [uv](https://github.com/astral-sh/uv), an extremely fast Python package and project manager.

### Pre-requisites

Make sure you have `uv` installed on your machine. If you don't have it yet, install it via:

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

Installation & Environment Sync
Clone the repository, navigate into the project root, and execute the following command to automatically create a virtual environment (.venv) and install all locked dependencies:

```bash
uv sync
```

Note: uv sync will automatically handle setup requirements for core components including Pydantic, PyYAML, OpenCV, and PaddleOCR.

## Configuration (config.yaml)

The system relies strictly on a local configuration file to define engine capabilities and output destinations. Below is the standard layout:

```yaml
ocr:
  language: "en"     # Language model dictionary (e.g., 'en', 'fr')
  export_format: "csv"     # Format of output (either 'csv', 'excel' or 'both')

paths:
  output_dir: "./output_result"  # Folder where output will be saved
```

## Features & Valid Formats

The orchestrator dynamically evaluates your input and applies safety checks before running the neural network layers:

- Supported Extensions: Only files ending in `.png`, `.jpg`, or `.jpeg` (case-insensitive) are accepted.

- Smart Routing: It automatically detects whether your target argument is a single file or a directory directory.

- Unified Logging: The pipeline runs with full internal logging framework (`DEBUG` level) activated to track processing queues and layout analysis steps in real-time.

### Examples

To run the pipeline, execute `src.main` via `uv run` by passing the configuration path followed by your target file or directory.

#### Example 1: Processing a Single Image File

If you feed it an individual file, the engine validates the extension and processes it immediately.

```bash
uv run python -m src.main config.yaml dataset/invoice_table.png
```

#### Example 2: Batch Processing an Entire Folder

If you feed it a folder path, the pipeline scans the directory, automatically skips incompatible files (like text files), sorts the remaining images alphabetically, and loops through them sequentially.

```bash
uv run python -m src.main config.yaml dataset/scanned_documents/
```

## Hugging Face Authentication

This project utilizes **PP-StructureV3**, which streams state-of-the-art layout and text parsing models natively from Hugging Face. To prevent connection timeouts and lift download speed restrictions, you must provide a free Hugging Face User Access Token.

### How to get your Hugging Face Token

1. **Create an Account:** Go to [huggingface.co](https://huggingface.co/) and sign up for a free account (if you don't have one already).
2. **Navigate to Settings:** Click on your profile picture in the top right-hand corner of the page and select **Settings**.
3. **Access Tokens Tab:** In the left sidebar, click on **Access Tokens** (or navigate directly to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
4. **Create New Token:** - Click the **New token** button.
   - Give your token a recognizable name (e.g., `local-ocr`).
   - Set the **Type** role to **Read** (this project only needs read access to pull down public models).
5. **Copy the Token:** Click **Generate**, then click the copy icon next to your new token string. It will begin with `hf_...`.

### Local Environment Setup

1. In the root directory of this project, create a file named `.env`:

  ```bash
  touch .env
  ```

2. Open the .env file and paste your copied token alongside the built-in high-speed download optimization flags:

```env
HF_TOKEN="your_copied_hf_token_here"
HF_HUB_READ_TIMEOUT=300
HF_HUB_DISABLE_XET=1
```

⚠️ The .env file is automatically ignored by Git inside .gitignore. Never commit your real Hugging Face token to public version control systems.
