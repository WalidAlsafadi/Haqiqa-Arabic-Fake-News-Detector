#!/usr/bin/env python3
"""
Palestine Fake News Detection - Main Entry Point

Clean, professional interface for the entire ML pipeline.
Uses the cleaned and refactored modules for optimal performance.
"""

import argparse
import sys
import os
from pathlib import Path


def _show_config():
    """Display current configuration settings"""
    from src.config.settings import (
        UCAS_NLP_COURSE_DATASET_PATH, ARABIC_NEWS_VERIFICATION_DATASET_PATH, 
        RANDOM_STATE, CV_FOLDS, TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF,
        ARABERT_MODEL_NAME, ARABERT_EPOCHS, ARABERT_BATCH_SIZE
    )
    
    print("CURRENT CONFIGURATION")
    print("-" * 30)
    print(f"UCAS NLP Course Dataset: {UCAS_NLP_COURSE_DATASET_PATH}")
    print(f"Arabic News Verification Dataset: {ARABIC_NEWS_VERIFICATION_DATASET_PATH}")
    print(f"Random State: {RANDOM_STATE}")
    print(f"CV Folds: {CV_FOLDS}")
    print(f"TF-IDF Max Features: {TFIDF_MAX_FEATURES}")
    print(f"TF-IDF Min DF: {TFIDF_MIN_DF}")
    print(f"TF-IDF Max DF: {TFIDF_MAX_DF}")
    print()
    print("AraBERT Configuration:")
    print(f"  Model: {ARABERT_MODEL_NAME}")
    print(f"  Epochs: {ARABERT_EPOCHS}")
    print(f"  Batch Size: {ARABERT_BATCH_SIZE}")


def _show_pipeline_info():
    """Display pipeline overview and structure"""
    # First show dataset information
    from src.config.settings import print_dataset_info
    print_dataset_info()
    
    print("PIPELINE OVERVIEW")
    print("-" * 30)
    print("1. Data Preparation:")
    print("   - Load raw datasets (UCAS NLP Course 2025 + Arabic News Verification)")
    print("   - Apply 3 text cleaning approaches (minimal, aggressive, transformers)")
    print("   - Create consistent train/validation/test splits (60/20/20)")
    print("   - Save 3 separate CSV files for flexibility:")
    print("     • minimal_cleaned.csv (traditional ML)")
    print("     • aggressive_cleaned.csv (robust ML)")
    print("     • transformers_cleaned.csv (AraBERT fine-tuning)")
    print()
    print("2. Model Selection:")
    print("   - Compare 5 models (XGBoost, Random Forest, Logistic Regression, SVM, Naive Bayes)")
    print("   - Use 5-fold cross-validation on training data for robust comparison")
    print("   - Evaluate models on validation data")
    print("   - Test minimal & aggressive datasets")
    print("   - Test set remains untouched")
    print()
    print("3. Hyperparameter Tuning:")
    print("   - Tune best performing model using validation set")
    print("   - Use GridSearchCV with proper parameter grids")
    print("   - Train final model on train+validation combined")
    print()
    print("4. Final Evaluation:")
    print("   - Evaluate tuned model on held-out test set")
    print("   - Generate comprehensive reports and visualizations")
    print()
    print("5. AraBERT Fine-tuning (Optional):")
    print("   - Use transformers_cleaned.csv independently")
    print("   - Compatible with Hugging Face workflows")


def _run_data_preparation():
    """Run data preparation phase"""
    from src.preprocessing.data_preparation import prepare_modern_dataset, create_separate_datasets
    from src.utils.data_splits import DataSplitter
    from src.config.settings import (
        UCAS_NLP_COURSE_DATASET_PATH, ARABIC_NEWS_VERIFICATION_DATASET_PATH,
        MINIMAL_DATASET_PATH, AGGRESSIVE_DATASET_PATH, TRANSFORMERS_DATASET_PATH,
        PROCESSED_DATA_DIR, ensure_dir_exists
    )

    print("\nSTARTING DATA PREPARATION")
    print("=" * 50)

    # Load and clean data using new approach (UCAS + Arabic News Verification)
    df = prepare_modern_dataset(
        ucas_path=UCAS_NLP_COURSE_DATASET_PATH,
        kaggle_path=ARABIC_NEWS_VERIFICATION_DATASET_PATH
    )
    print(f"Data prepared: {len(df)} samples")

    # Create data splits (for consistent indices across all datasets)
    print("Creating train/validation/test splits (60/20/20)")
    splitter = DataSplitter()
    splitter.create_splits(df)

    # Save separate CSV files for flexibility and framework compatibility
    print("Saving separate datasets for different approaches")
    ensure_dir_exists(PROCESSED_DATA_DIR)

    minimal_df, aggressive_df, transformers_df = create_separate_datasets(df)
    minimal_df.to_csv(MINIMAL_DATASET_PATH, index=False)
    print(f"  - Minimal dataset: {len(minimal_df)} samples → {MINIMAL_DATASET_PATH}")

    aggressive_df.to_csv(AGGRESSIVE_DATASET_PATH, index=False)
    print(f"  - Aggressive dataset: {len(aggressive_df)} samples → {AGGRESSIVE_DATASET_PATH}")

    transformers_df.to_csv(TRANSFORMERS_DATASET_PATH, index=False)
    print(f"  - Transformers dataset: {len(transformers_df)} samples → {TRANSFORMERS_DATASET_PATH}")

    print("Data preparation completed successfully")
    print("All datasets use consistent train/val/test splits via saved indices")
    return df, splitter


def _run_model_selection(splitter):
    """Run model selection phase with cross-validation"""
    from src.ml_algorithms.model_selection import compare_models
    from src.config.settings import MODEL_SELECTION_OUTPUT_DIR, ensure_dir_exists
    
    # Create output directory
    ensure_dir_exists(MODEL_SELECTION_OUTPUT_DIR)
    
    # Compare models using cross-validation
    results = compare_models(splitter, dataset_names=['minimal', 'aggressive'], 
                           output_dir=MODEL_SELECTION_OUTPUT_DIR)
    
    if not results:
        print("Model selection failed")
        return None
    
    print("Model selection completed successfully")
    return results


def _run_hyperparameter_tuning(splitter):
    """Run hyperparameter tuning phase"""
    from src.ml_algorithms.hyperparameter_tuning import tune_best_model
    
    # Tune the best model
    result = tune_best_model(splitter)
    
    if not result:
        print("Hyperparameter tuning failed")
        return None
    
    print("Hyperparameter tuning completed successfully")
    return result
    return result


def _run_final_evaluation(splitter):
    """Run final evaluation phase"""
    from src.ml_algorithms.model_evaluation import evaluate_final_model
    
    # Evaluate on test set
    results = evaluate_final_model(splitter)
    
    if not results:
        print("Final evaluation failed")
        return None
    
    # Display key results
    print("Final Test Results:")
    print(f"  Accuracy: {results['test_accuracy']:.4f}")
    print(f"  F1 (weighted): {results['test_f1_weighted']:.4f}")
    print(f"  Fake News F1: {results['fake_news_f1']:.4f}")
    print(f"  Real News F1: {results['real_news_f1']:.4f}")
    
    print("Final evaluation completed successfully")
    return results


def _run_complete_pipeline():
    """Run the complete ML pipeline"""

    try:
        # Phase 1: Data Preparation
        df, splitter = _run_data_preparation()
        
        # Phase 2: Model Selection
        model_results = _run_model_selection(splitter)
        if not model_results:
            return False
        
        # Phase 3: Hyperparameter Tuning
        tuning_result = _run_hyperparameter_tuning(splitter)
        if not tuning_result:
            return False
        
        # Phase 4: Final Evaluation
        final_results = _run_final_evaluation(splitter)
        if not final_results:
            return False
        
        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("Check outputs/ directory for detailed results")
        return True
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return False


def main():
    """Main entry point with clean argument parsing"""
    parser = argparse.ArgumentParser(
        description="Palestine Fake News Detection  Machine Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Run complete pipeline
  python main.py --data-prep            # Prepare data and create splits
  python main.py --model-selection      # Compare models using CV
  python main.py --tuning               # Tune best model parameters
  python main.py --evaluation           # Evaluate on test set
  python main.py --config               # Show configuration
  python main.py --info                 # Show pipeline information
        """
    )
    
    # Phase commands
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data-prep', action='store_true',
                       help='Run data preparation phase only')
    group.add_argument('--model-selection', action='store_true',
                       help='Run model selection phase only')
    group.add_argument('--tuning', action='store_true',
                       help='Run hyperparameter tuning phase only')
    group.add_argument('--evaluation', action='store_true',
                       help='Run final evaluation phase only')
    group.add_argument('--config', action='store_true',
                       help='Show configuration settings')
    group.add_argument('--info', action='store_true',
                       help='Show pipeline information')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Palestine Fake News Detection ML Pipeline")
    print("=" * 60)
    
    try:
        # Show configuration
        if args.config:
            _show_config()
            return
        
        # Show pipeline info
        if args.info:
            _show_pipeline_info()
            return
        
        # Run individual phases
        if args.data_prep:
            print("Running data preparation phase")
            df, splitter = _run_data_preparation()
            print("Data preparation completed")
            return
        
        if args.model_selection:
            print("Running model selection phase")
            # Load existing splits
            from src.utils.data_splits import DataSplitter
            
            # Check if splits exist
            splitter = DataSplitter.load_splits()
            if splitter is None:
                print("No existing splits found. Running data preparation first...")
                df, splitter = _run_data_preparation()
            
            _run_model_selection(splitter)
            return
        
        if args.tuning:
            print("Running hyperparameter tuning phase")
            # Load existing splits
            from src.utils.data_splits import DataSplitter
            
            splitter = DataSplitter.load_splits()
            if splitter is None:
                print("No data splits found. Please run data preparation first.")
                return
            
            _run_hyperparameter_tuning(splitter)
            return
        
        if args.evaluation:
            print("Running final evaluation phase")
            # Load existing splits
            from src.utils.data_splits import DataSplitter
            
            splitter = DataSplitter.load_splits()
            if splitter is None:
                print("No data splits found. Please run data preparation first.")
                return
            
            _run_final_evaluation(splitter)
            return
        
        # Default: Run complete pipeline
        print("Running complete pipeline...")
        success = _run_complete_pipeline()
        
        if success:
            print("Pipeline completed successfully!")
        else:
            print("Pipeline failed. Check outputs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
