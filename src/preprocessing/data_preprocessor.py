import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.cleaning import apply_all_cleaning, prepare_text_for_transformers
from src.utils.logger import log_info, log_step, log_success

RAW_PATH = "data/raw/original_news_data.csv"
OUTPUT_DIR = "data/processed"
REMOVE_OUTLIERS = True

if __name__ == "__main__":
    log_info(f"Loading and cleaning dataset from: {RAW_PATH}")
    df = apply_all_cleaning(RAW_PATH, remove_outliers=REMOVE_OUTLIERS)

    log_step("Saving cleaned CSV files...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df[["text_min", "label"]].to_csv(f"{OUTPUT_DIR}/text_clean_min.csv", index=False)
    log_step("Mininal text cleaning CSV file saved...")

    df[["text_agg", "label"]].to_csv(f"{OUTPUT_DIR}/text_clean_agg.csv", index=False)
    log_step("Aggressive text cleaning CSV file saved...")

    df[["text_min", "title_length", "content_length", "platform_encoded", "label"]].to_csv(
        f"{OUTPUT_DIR}/with_features.csv", index=False
    )

    # Save raw text for transformers (no aggressive cleaning)
    log_step("Preparing and saving dataset for transformer models...")
    df_transformers = prepare_text_for_transformers(RAW_PATH)
    df_transformers.to_csv(f"{OUTPUT_DIR}/text_for_transformers.csv", index=False)
    log_step("Transformers model text cleaning CSV file saved...")

    log_success("Data preprocessing complete.")
