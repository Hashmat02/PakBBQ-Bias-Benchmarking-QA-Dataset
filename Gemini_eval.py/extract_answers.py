import re
import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract model answers from Gemini response JSONL file.")
    parser.add_argument(
        "--results-file",
        default="responses_gemini-2.0-flash-lite.jsonl",
        help="Path to the JSONL file containing model responses."
    )
    parser.add_argument(
        "--output-file",
        default="model_answers.json",
        help="Path to the output JSON file for extracted answers."
    )
    parser.add_argument(
        "--language",
        choices=["english", "urdu"],
        default="english",
        help="Language of the model responses (english or urdu)."
    )
    return parser.parse_args()

def extract_answers(results_path, language):
    """Yield (numeric_id, answer_letter) from response JSONL file."""
    # Urdu-to-English answer mapping
    urdu_to_english = {
        "ا": "A",
        "ب": "B",
        "پ": "C"
    }

    # Regular expression for valid answer letters
    valid_letters = r"[ABC]" if language == "english" else r"[ابپ]"

    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue

            # Get custom_id and extract numeric_id
            cid = rec.get("id", "")
            numeric_id = cid.split(":")[-1] if ":" in cid else cid

            # Get the assistant's answer
            try:
                text = rec["response"]["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError):
                print(f"Warning: No valid response text for ID {numeric_id}")
                continue

            # Extract the answer letter
            match = re.search(valid_letters, text)
            letter = match.group(0) if match else None

            if letter:
                # Map Urdu to English if necessary
                if language == "urdu":
                    letter = urdu_to_english.get(letter, letter)
                yield numeric_id, letter
            else:
                print(f"Warning: No valid answer letter found for ID {numeric_id}")

def main():
    args = parse_args()

    if not os.path.exists(args.results_file):
        print(f"❌ File not found: {args.results_file}")
        return

    print(f"🔍 Parsing: {args.results_file}")

    answers = {}
    for numeric_id, letter in extract_answers(args.results_file, args.language):
        answers[numeric_id] = letter

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(answers)} answers to {args.output_file}")

if __name__ == "__main__":
    main()