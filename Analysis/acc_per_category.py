import pandas as pd
import os

def extract_language_and_model(filepath):
    filename = os.path.basename(filepath)
    parts = filename.replace('_results.csv', '').split('_', 1)
    return parts if len(parts) == 2 else (None, None)

def compute_accuracy(df, group_cols):
    return (
        df.groupby(group_cols)
        .apply(lambda g: (g['correct'] == True).mean())
        .reset_index(name='accuracy')
    )

def evaluate_all_models(reference_csv_path, model_csv_paths):
    try:
        reference_df = pd.read_csv(reference_csv_path)
    except Exception as e:
        print(f"❌ Failed to load reference CSV: {e}")
        return

    category_rows = []

    for model_csv in model_csv_paths:
        language, model_name = extract_language_and_model(model_csv)
        if not language or not model_name:
            print(f"⚠️ Skipping {model_csv}: couldn't extract language/model.")
            continue

        try:
            df = pd.read_csv(model_csv)

            # Full (overall) accuracy per category
            overall_acc = compute_accuracy(df, ['category']).rename(columns={'accuracy': 'overall_accuracy'})

            # disambig-only
            disambig_acc = compute_accuracy(df[df['context_condition'] == 'disambig'], ['category']).rename(columns={'accuracy': 'disambig_accuracy'})

            # ambig-only
            ambig_acc = compute_accuracy(df[df['context_condition'] == 'ambig'], ['category']).rename(columns={'accuracy': 'ambig_accuracy'})

            # Merge all three
            merged = overall_acc.merge(disambig_acc, on='category', how='outer') \
                                .merge(ambig_acc, on='category', how='outer')

            merged['language'] = language
            merged['model'] = model_name

            # Round to 2 decimal places
            for col in ['overall_accuracy', 'disambig_accuracy', 'ambig_accuracy']:
                if col in merged:
                    merged[col] = (merged[col] * 100).round(2)

            category_rows.extend(merged.to_dict(orient='records'))
            print(f"✅ Processed {language}_{model_name}")

        except Exception as e:
            print(f"⚠️ Error processing {model_csv}: {e}")
            continue

    # Save final result
    df_out = pd.DataFrame(category_rows)
    df_out = df_out[['language', 'model', 'category', 'overall_accuracy', 'disambig_accuracy', 'ambig_accuracy']]
    df_out.to_csv("category_accuracy.csv", index=False)
    print("📁 Results saved to category_accuracy.csv")

if __name__ == '__main__':
    results_folder = 'results'
    try:
        files = [os.path.join(results_folder, f) for f in os.listdir(results_folder) if f.endswith('.csv')]

        # Find reference CSV (assuming it follows the same naming pattern)
        reference_csv = None
        for f in files:
            if 'gemini-2.0-flash-lite_results' in f:
                reference_csv = f
                break

        if not reference_csv:
            print("❌ Could not find reference CSV file")
            exit(1)

        model_csvs = files

        evaluate_all_models(reference_csv, model_csvs)
    except Exception as e:
        print(f"❌ Unexpected error in main: {e}")
