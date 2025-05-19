from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from dotenv import load_dotenv
import time
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Submit a batch job to the Gemini model.")
    parser.add_argument(
        "--key-path",
        default="client_secret.json",
        help="Path to the Google Cloud API key JSON file."
    )
    parser.add_argument(
        "--region",
        default="us-central1",
        help="Google Cloud region (e.g., us-central1)."
    )
    parser.add_argument(
        "--project",
        default="gen-lang-client-0339774390",
        help="Google Cloud project ID."
    )
    parser.add_argument(
        "--input-gcs-uri",
        default="gs://pakbbq-batch-data/gemini_input.jsonl",
        help="GCS URI for input JSONL file."
    )
    parser.add_argument(
        "--output-gcs-uri",
        default="gs://pakbbq-batch-data/",
        help="GCS URI for output directory."
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash-lite-001",
        help="Model name for batch job."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load environment variables
    load_dotenv()

    # Validate and set API key path
    key_path = os.path.abspath(args.key_path)
    if not os.path.exists(key_path):
        print(f"Error: API key file not found at {key_path}")
        exit(1)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

    # Initialize client
    client = genai.Client(vertexai=True, location=args.region, project=args.project)

    # Submit batch job
    job = client.batches.create(
        model=args.model,
        src=args.input_gcs_uri,
        config=CreateBatchJobConfig(dest=args.output_gcs_uri)
    )

    print(f"Submitted job: {job.name}")

    # Define completed job states
    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
    }

    # Poll job status
    while job.state not in completed_states:
        time.sleep(30)
        job = client.batches.get(name=job.name)
        print(f"Job state: {job.state}")

    print(f"Job finished with state: {job.state}")
    print(f"Output available at: {args.output_gcs_uri}")