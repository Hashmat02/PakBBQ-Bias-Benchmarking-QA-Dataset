# PakBBQ Bias Benchmarking QA Dataset



## Environment Setup
To download the relevant packages, please run

```bash
pip install -r requirements.txt
```

## Evaluation Setup for OpenAI/DeekSeek Models

| Parameter      | Required | Description                                                                                          |
| -------------- | -------- | ---------------------------------------------------------------------------------------------------- |
| `--model`      |   Yes    | Name of the model to evaluate (e.g., `gpt-4`, `gpt-4.1-mini`, `gpt-4.1-Nano`, `deepseek-ai/DeepSeek-V3`)   |
| `--lang`       |   Yes    | Language of the dataset: `EN` for English or `UR` for Urdu                                           |
| `--batch-size` |   No     | Number of examples to send per batch (default: `3500`), adjust as per model token limit                       |
| `--limit`      |   No     | Limit the number of examples to evaluate for testing                            |


1. Run the following command in the CLI.
    ```bash
    cd GPT_and_DeepSeek_eval.py

    python3 streamline_batch_pipeline.py \
      --model <Model_Name> \
      --lang <EN/UR> \
      --batch-size <3500> 
    ```

2. To test a subset of data first.
    ```bash
    cd GPT_and_DeepSeek_eval.py

    python3 streamline_batch_pipeline.py \
      --model <Model_Name> \
      --lang <EN/UR> \
      --batch-size <3500> \
      --limit <5>
    ```

Final results will be generated in a ModelName_Lang_Results.csv file, which can then be used to calculate any relevant metrics.

# Evaluation with Google Gemini

This repository contains scripts to evaluate biases in large language models (LLMs) using the PakBBQ dataset with Google’s Gemini model. The pipeline processes English and Urdu question-answering data, submits batch jobs to Gemini, extracts responses, and evaluates results.

## Overview

The project automates:
- Generating JSONL input files for Gemini from PakBBQ data.
- Submitting batch jobs to Gemini via Google Cloud.
- Extracting model responses (English: A, B, C; Urdu: ا, ب, پ).
- Mapping responses to ground truth and saving evaluation results as CSV.

## Prerequisites

- Python 3.8+
- Google Cloud account with API key
- Installed dependencies: `google-genai`, `pandas`, `python-dotenv`
- PakBBQ dataset (English and Urdu JSONL files)


## Usage

The pipeline consists of four scripts, each with CLI arguments for customization.

1. **Create JSONL Input**:
   ```bash
   python create_jsonl.py --data-folder ../PakBBQ/data --language english --output-file gemini_input.jsonl
   ```
   For Urdu:
   ```bash
   python create_jsonl.py --data-folder ../PakBBQ/data_Urdu --language urdu
   ```

2. **Submit Batch Job**:
   ```bash
   python submit_jobs.py --key-path /path/to/client_secret.json --input-gcs-uri gs://your-bucket/input.jsonl --output-gcs-uri gs://your-bucket/output/
   ```

3. **Extract Answers**:
   ```bash
   python extract_answers.py --results-file responses_gemini-2.0-flash-lite.jsonl --language english
   ```
   For Urdu:
   ```bash
   python extract_answers.py --language urdu
   ```

4. **Evaluate Results**:
   ```bash
   python mapping.py --data-folder ../PakBBQ/data --model-answers-file model_answers.json --output-file results.csv
   ```

## Files

- `create_jsonl.py`: Generates JSONL input for Gemini from PakBBQ data (English or Urdu).
- `submit_jobs.py`: Submits batch jobs to Gemini via Google Cloud.
- `extract_answers.py`: Extracts model responses and maps Urdu answers to English (A, B, C).
- `mapping.py`: Matches responses to ground truth and saves evaluation results as CSV.

## Notes

- Ensure your Google Cloud project and GCS bucket are configured correctly.
- Urdu processing requires UTF-8 encoding support.
- Use `--help` with any script for CLI argument details (e.g., `python create_jsonl.py --help`).

## Responsible Use:

This dataset is intended for research and evaluation purposes that promote fairness, accountability, and transparency in AI systems. Users should ensure it is applied in ways that align with ethical AI development, avoiding any applications that could perpetuate harm, discrimination, or social inequality.
    

    




 
