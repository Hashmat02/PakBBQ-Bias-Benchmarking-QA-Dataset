# PakBBQ---QA-Bias-Benchmarking-Dataset





## Environment Setup
To download the relevant packages, please run

pip install -r requirements.txt

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
      --limit 5 
    ```


## Responsible Use:

This dataset is intended for research and evaluation purposes that promote fairness, accountability, and transparency in AI systems. Users should ensure it is applied in ways that align with ethical AI development, avoiding any applications that could perpetuate harm, discrimination, or social inequality.
    

    




 
