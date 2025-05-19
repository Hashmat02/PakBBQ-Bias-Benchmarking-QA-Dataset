import json
import os
import glob
import time
import pandas as pd
import logging
import argparse
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model answers against JSONL data files.")
    parser.add_argument(
        "--data-folder",
        default="C:\\Users\\Arham Mirza\\Documents\\GitHub\\Sproj-LLM-Biases\\PakBBQ\\data copy",
        help="Directory containing JSONL data files."
    )
    parser.add_argument(
        "--model-answers-file",
        default="model_answers.json",
        help="JSON file containing model answers."
    )
    parser.add_argument(
        "--output-file",
        default="gemini-2.0-flash-lite-results.csv",
        help="Output CSV file for evaluation results."
    )
    parser.add_argument(
        "--progress-file",
        default="progress.json",
        help="File to track processing progress."
    )
    return parser.parse_args()

# Function to load saved progress
def load_progress(progress_file: str) -> Dict:
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse progress file {progress_file}. Starting from beginning.")
    return {}

# Function to save progress
def save_progress(progress_file: str, file_name: str, line_idx: int) -> None:
    with open(progress_file, 'w') as f:
        json.dump({"last_file": file_name, "last_line": line_idx}, f)

# Load model answers from a file
def load_model_answers(model_answers_file: str) -> Dict[str, str]:
    """
    Loads model answers from a file with the format:
    {"0": "A", "1": "A", "2": "B", ...}
    """
    try:
        with open(model_answers_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading model answers: {e}")
        return {}

# Main evaluation function
def evaluate(data_folder: str, output_file: str, progress_file: str, model_answers: Dict[str, str]):
    """
    Process JSONL files and match them with model answers.
    """
    # Track processed examples
    processed_count = 0
    
    # Load progress if resuming
    progress = load_progress(progress_file)
    resume_from_file = progress.get("last_file")
    resume_from_line = progress.get("last_line", -1)
    resume = resume_from_file is None
    
    # Define output columns
    columns = [
        "source_file", "line_index", "example_id", "question_index", "type",
        "context", "question", "ans0", "ans1", "ans2",
        "ans0_stereotype", "ans1_stereotype", "ans2_stereotype",
        "model_choice", "correct_choice",
        "category", "subcategory", "context_condition",
        "polarity", "stereotyped_groups", "correct", "version",
    ]
    
    # Create output file if it doesn't exist
    if not os.path.exists(output_file):
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)
    
    # Process each JSONL file
    for file_path in sorted(glob.glob(os.path.join(data_folder, "*.jsonl"))):
        file_name = os.path.basename(file_path)
        
        # Handle resuming from a specific file
        if not resume:
            if file_name == resume_from_file:
                resume = True
            else:
                continue
        
        logger.info(f"Processing file: {file_name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                # Skip already processed lines when resuming
                if file_name == resume_from_file and idx <= resume_from_line:
                    continue
                
                if line.strip() == "":
                    logger.debug(f"Skipping empty line in {file_name}")
                    continue
                
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON at line {idx} in {file_name}: {e}")
                    continue
                
                # Extract example ID
                example_id = str(example.get("id"))
                
                # Check if we have a model answer for this example
                if example_id not in model_answers:
                    logger.warning(f"No model answer found for example ID {example_id}")
                    continue
                
                # Get model choice
                model_choice = model_answers[example_id]
                
                # Extract data from example
                context = example.get("context", "")
                question = example.get("question", "")
                ans0 = example.get("ans0", "")
                ans1 = example.get("ans1", "")
                ans2 = example.get("ans2", "")
                answer_info = example.get("answer_info", {})
                ans0_stereotype = answer_info.get("ans0", ["", ""])[1]
                ans1_stereotype = answer_info.get("ans1", ["", ""])[1]
                ans2_stereotype = answer_info.get("ans2", ["", ""])[1]
                
                # Map letter choice to index
                letter_to_idx = {"A": 0, "B": 1, "C": 2}
                model_idx = letter_to_idx.get(model_choice)
                
                # Get correct index from example
                correct_idx = int(example.get("label", -1))
                
                # Check if model choice matches correct choice
                is_correct = (model_idx == correct_idx)
                
                # Get additional metadata
                additional_metadata = example.get("additional_metadata", {})
                stereotyped_groups = additional_metadata.get("stereotyped_groups", [])
                
                # Create result dictionary
                result = {
                    "source_file": file_name,
                    "line_index": idx,
                    "example_id": example_id,
                    "question_index": example.get("question_index", ""),
                    "type": example.get("type", ""),
                    "context": context,
                    "question": question,
                    "ans0": ans0,
                    "ans1": ans1,
                    "ans2": ans2,
                    "ans0_stereotype": ans0_stereotype,
                    "ans1_stereotype": ans1_stereotype,
                    "ans2_stereotype": ans2_stereotype,
                    "model_choice": model_choice,
                    "correct_choice": correct_idx,
                    "category": example.get("category", ""),
                    "subcategory": additional_metadata.get("subcategory", ""),
                    "context_condition": example.get("context_condition", ""),
                    "polarity": example.get("question_polarity", ""),
                    "stereotyped_groups": ", ".join(stereotyped_groups),
                    "correct": is_correct,
                    "version": additional_metadata.get("version", "")
                }
                
                # Save result to output file
                pd.DataFrame([result]).to_csv(output_file, mode='a', header=False, index=False)
                processed_count += 1
                
                # Save progress
                save_progress(progress_file, file_name, idx)
                
                # Optional sleep to avoid overwhelming systems
                time.sleep(0.01)  # Minimal sleep, adjust as needed
                
                # Logging progress
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} examples")
        
        logger.info(f"Completed file: {file_name}")
    
    logger.info(f"Evaluation complete. Processed {processed_count} examples. Results saved to {output_file}")

# Main entry point
if __name__ == "__main__":
    args = parse_args()
    
    # Resolve absolute path for data folder
    data_folder = os.path.abspath(args.data_folder)
    
    # Validate model answers file
    if not os.path.exists(args.model_answers_file):
        logger.error(f"Model answers file not found: {args.model_answers_file}")
        exit(1)
    
    # Load model answers
    model_answers = load_model_answers(args.model_answers_file)
    if not model_answers:
        logger.error("Failed to load model answers. Exiting.")
        exit(1)
    
    logger.info(f"Loaded {len(model_answers)} model answers")
    
    # Run evaluation
    evaluate(data_folder, args.output_file, args.progress_file, model_answers)