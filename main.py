#!/usr/bin/env python3
"""
Palestine Fake News Detection - Main Entry Point

Simple, clean interface for the entire ML pipeline.
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Palestine Fake News Detection ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                      # Run complete pipeline
  python main.py --data-prep                # Run data preparation only
  python main.py --model-selection          # Run model selection only  
  python main.py --tuning                   # Run hyperparameter tuning only
  python main.py --evaluation               # Run final evaluation only
  python main.py --tuning --force-retune    # Force fresh grid search (ignore cache)
  python main.py --config                   # Show configuration
  python main.py --summary                  # Show pipeline info
        """
    )
    
    # Main commands
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (recommended)')
    parser.add_argument('--data-prep', action='store_true',
                       help='Run data preparation phase only')
    parser.add_argument('--model-selection', action='store_true',
                       help='Run model selection phase only')
    parser.add_argument('--tuning', action='store_true',
                       help='Run hyperparameter tuning phase only')
    parser.add_argument('--evaluation', action='store_true',
                       help='Run final evaluation phase only')
    parser.add_argument('--config', action='store_true',
                       help='Show configuration settings')
    parser.add_argument('--summary', action='store_true',
                       help='Show pipeline summary')
    
    # Tuning options
    parser.add_argument('--force-retune', action='store_true',
                       help='Force fresh hyperparameter tuning (ignore cached parameters)')
    
    # Future features
    parser.add_argument('--predict', type=str,
                       help='Make single prediction on text (coming soon)')
    
    args = parser.parse_args()
    
    # Ensure at least one command is provided
    if not any([args.all, args.data_prep, args.model_selection, args.tuning, args.evaluation, args.config, args.summary, args.predict]):
        parser.print_help()
        return
    
    print("=" * 60)
    print("Palestine Fake News Detection ML Pipeline")
    print("=" * 60)
    
    try:
        # Configuration display
        if args.config:
            from src.config import settings
            print("CONFIGURATION")
            print("-" * 20)
            print(f"Datasets: {settings.DATASETS_TO_USE}")
            print(f"CV Folds: {settings.CV_FOLDS}")
            print(f"Max Features: {settings.TFIDF_MAX_FEATURES}")
            return
            
        # Pipeline summary
        if args.summary:
            print("PIPELINE SUMMARY")
            print("=" * 20)
            print("Datasets: ['text_minimal', 'text_aggressive', 'text_transformers']")
            print("CV Folds: 5")
            print("Max Features: 5000")
            print("Text Cleaning: Minimal + Aggressive + Transformers")
            print("Model Selection: 5 models on minimal & aggressive datasets")
            print("Hyperparameter Tuning: Best 2 models with GridSearch")
            print("Model Evaluation: Deep analysis with plots and reports")
            return
            
        # Complete pipeline
        if args.all:
            # Import everything we need
            import pandas as pd
            from src.preprocessing.text_cleaner import prepare_data
            from src.utils.data_splits import create_data_splits
            from src.models.model_selection import compare_models_on_datasets
            from src.models.hyperparameter_tuning_proper import tune_best_models_proper
            from src.models.model_evaluation import evaluate_final_model_properly
            
            # Data Loading Stage
            print("=" * 60)
            print("DATA LOADING")
            print("=" * 60)
            print("Loading data...")
            df = pd.read_csv("data/raw/original_news_data.csv")
            print(f"Loaded {len(df)} articles")
            print()
            
            # Data Preprocessing Stage
            print("=" * 60)
            print("DATA PREPROCESSING")
            print("=" * 60)
            print("Cleaning data...")
            clean_df = prepare_data(df)
            print(f"After cleaning: {len(clean_df)} articles")
            print()
            
            # Save separate datasets for minimal, aggressive, and transformers cleaning
            minimal_df = clean_df[['text_minimal', 'label']].copy()
            minimal_df = minimal_df.rename(columns={'text_minimal': 'text'})
            aggressive_df = clean_df[['text_aggressive', 'label']].copy()
            aggressive_df = aggressive_df.rename(columns={'text_aggressive': 'text'})
            transformers_df = clean_df[['text_transformers', 'label']].copy()
            transformers_df = transformers_df.rename(columns={'text_transformers': 'text'})
            
            minimal_df.to_csv("data/processed/minimal_cleaned.csv", index=False)
            aggressive_df.to_csv("data/processed/aggressive_cleaned.csv", index=False)
            transformers_df.to_csv("data/processed/transformers_cleaned.csv", index=False)
            
            print("Processed datasets saved:")
            print("  - Minimal cleaning: data/processed/minimal_cleaned.csv") 
            print("  - Aggressive cleaning: data/processed/aggressive_cleaned.csv")
            print("  - Transformers ready: data/processed/transformers_cleaned.csv")
            print()

            # Data Splitting Stage - CRITICAL FOR PROPER ML
            print("=" * 60)
            print("DATA SPLITTING")
            print("=" * 60)
            
            # Create consistent splits for both datasets
            text_minimal = clean_df['text_minimal']
            text_aggressive = clean_df['text_aggressive'] 
            y = clean_df['label']
            
            splits = create_data_splits(text_minimal, text_aggressive, y, save_splits=True)
            print()
            
            # Model Selection Phase
            results = compare_models_on_datasets(splits)
            
            if results is None:
                print("Model selection failed")
                return
            
            # Find best 2 models across all datasets
            sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_f1_mean'], reverse=True)
            best_models = [(result[1]['model_name'], result[1]['dataset']) for result in sorted_results[:2]]
            
            print("Best 2 models selected for hyperparameter tuning:")
            for model_name, dataset in best_models:
                f1_score = next(r[1]['cv_f1_mean'] for r in sorted_results if r[1]['model_name'] == model_name and r[1]['dataset'] == dataset)
                print(f"  - {model_name} on {dataset}: F1={f1_score:.4f}")
            print()
            
            # Hyperparameter Tuning Phase
            tuned_results = tune_best_models_proper(best_models, splits, force_retune=args.force_retune)
            
            if not tuned_results:
                print("Hyperparameter tuning failed")
                return
            
            # Find the absolute best model after tuning
            best_tuned = max(tuned_results.items(), key=lambda x: x[1]['best_score'])
            best_model_key = best_tuned[0]
            best_model_info = best_tuned[1]
            
            print(f"Best tuned model: {best_model_key} with Validation F1={best_model_info['best_score']:.4f}")
            print()
            
            # Final Model Evaluation on held-out test set
            print("=" * 60)
            print("FINAL EVALUATION")
            print("=" * 60)
            
            model_name = best_model_info['model_name']
            dataset_name = best_model_info['dataset']
            tuned_model = best_model_info['model']
            
            # Save the best model and vectorizer for Streamlit
            import pickle
            import os
            os.makedirs("models/trained", exist_ok=True)
            
            # Save the final model
            with open("models/trained/best_model.pkl", 'wb') as f:
                pickle.dump(tuned_model, f)
            
            # Save the corresponding vectorizer
            from src.models.model_selection import get_vectorizers_from_model_selection
            vectorizers = get_vectorizers_from_model_selection()
            best_vectorizer_key = f"{model_name}_{dataset_name}"
            if best_vectorizer_key in vectorizers:
                with open("models/trained/fitted_vectorizer.pkl", 'wb') as f:
                    pickle.dump(vectorizers[best_vectorizer_key], f)
                print(f"Best model and vectorizer saved: {best_vectorizer_key}")
            
            # Final evaluation on held-out test set
            final_test_results = evaluate_final_model_properly(
                tuned_model, 
                splits,
                dataset_name,
                model_name
            )
            
            if final_test_results:
                print("PIPELINE COMPLETED - SCIENTIFICALLY VALID RESULTS")
                print("=" * 60)
                print("SUMMARY FOR REPORTING:")
                print()
                print("Model Selection (train+validation data only):")
                print(f"  Best model: {model_name} on {dataset_name} dataset")
                print(f"  CV F1 (weighted): {sorted_results[0][1]['cv_f1_mean']:.4f}")
                print()
                print("Hyperparameter Tuning (validation data only):")
                print(f"  Best validation F1: {best_model_info['best_score']:.4f}")
                print()
                print("Final Test Results:")
                print(f"  Test F1 (weighted): {final_test_results['test_f1_weighted']:.4f}")
                print(f"  Test Accuracy: {final_test_results['test_accuracy']:.4f}")
                print(f"  Fake News F1: {final_test_results['fake_news_f1']:.4f}")
                print(f"  Real News F1: {final_test_results['real_news_f1']:.4f}")
                if final_test_results['test_auc']:
                    print(f"  Test AUC: {final_test_results['test_auc']:.4f}")
                print()
                print("Check outputs/final_evaluation/ for detailed results")
                print("Models saved in models/trained/ for deployment")
                print("=" * 60)
                
            return
            
        # Individual phase commands
        if args.data_prep:
            import pandas as pd
            from src.preprocessing.text_cleaner import prepare_data
            
            print("=" * 60)
            print("DATA PREPARATION PHASE")
            print("=" * 60)
            
            # Data Loading
            print("Loading data...")
            df = pd.read_csv("data/raw/original_news_data.csv")
            print(f"Loaded {len(df)} articles")
            print()
            
            # Data Preprocessing
            print("Cleaning data...")
            clean_df = prepare_data(df)
            print(f"After cleaning: {len(clean_df)} articles")
            print()
            
            # Save datasets
            minimal_df = clean_df[['text_minimal', 'label']].copy()
            minimal_df = minimal_df.rename(columns={'text_minimal': 'text'})
            aggressive_df = clean_df[['text_aggressive', 'label']].copy()
            aggressive_df = aggressive_df.rename(columns={'text_aggressive': 'text'})
            transformers_df = clean_df[['text_transformers', 'label']].copy()
            transformers_df = transformers_df.rename(columns={'text_transformers': 'text'})
            
            minimal_df.to_csv("data/processed/minimal_cleaned.csv", index=False)
            aggressive_df.to_csv("data/processed/aggressive_cleaned.csv", index=False)
            transformers_df.to_csv("data/processed/transformers_cleaned.csv", index=False)
            
            print("Processed datasets saved:")
            print("  - Minimal cleaning: data/processed/minimal_cleaned.csv") 
            print("  - Aggressive cleaning: data/processed/aggressive_cleaned.csv")
            print("  - Transformers ready: data/processed/transformers_cleaned.csv")
            print("=" * 60)
            return
            
        if args.model_selection:
            import pandas as pd
            import os
            from src.models.model_selection import compare_models_on_datasets
            from src.utils.data_splits import create_data_splits, load_data_splits
            
            print("=" * 60)
            print("MODEL SELECTION")
            print("=" * 60)
            
            # Check if cleaned data exists
            minimal_file = "data/processed/minimal_cleaned.csv"
            aggressive_file = "data/processed/aggressive_cleaned.csv"
            
            if not os.path.exists(minimal_file) or not os.path.exists(aggressive_file):
                print("ERROR: Cleaned data not found!")
                print("Please run data preparation first: python main.py --data-prep")
                return
            
            # Load cleaned data
            print("Loading cleaned data...")
            minimal_df = pd.read_csv(minimal_file)
            aggressive_df = pd.read_csv(aggressive_file)
            print(f"  - Minimal dataset: {len(minimal_df)} articles")
            print(f"  - Aggressive dataset: {len(aggressive_df)} articles")
            print()
            
            # Check if data splits exist, create if not
            splits = load_data_splits()
            if splits is None:
                print("Creating data splits...")
                text_minimal = minimal_df['text']
                text_aggressive = aggressive_df['text']
                y = minimal_df['label']
                
                splits = create_data_splits(text_minimal, text_aggressive, y, save_splits=True)
            else:
                print("Using existing data splits...")
            
            print()
            
            # Model comparison
            results = compare_models_on_datasets(splits)
            
            if results:
                sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_f1_mean'], reverse=True)
                print("Top 5 model results:")
                for i, (key, result) in enumerate(sorted_results[:5]):
                    print(f"  {i+1}. {result['model_name']} on {result['dataset']}: F1={result['cv_f1_mean']:.4f}")
                
                print()
                print("Best 2 models for hyperparameter tuning:")
                for i, (key, result) in enumerate(sorted_results[:2]):
                    print(f"  {i+1}. {result['model_name']} on {result['dataset']}: F1={result['cv_f1_mean']:.4f}")
            
            print("=" * 60)
            return
            
        if args.tuning:
            import pandas as pd
            import os
            from src.models.model_selection import load_model_selection_results
            from src.models.hyperparameter_tuning_proper import tune_best_models_proper
            from src.utils.data_splits import load_data_splits
            
            # Check if model selection results exist
            model_results = load_model_selection_results()
            if not model_results:
                print("ERROR: Model selection results not found!")
                print("Please run model selection first: python main.py --model-selection")
                return
            
            # Check if data splits exist
            splits = load_data_splits()
            if splits is None:
                print("ERROR: Data splits not found!")
                print("Please run model selection first: python main.py --model-selection")
                return
            
            # Get best 2 models from saved results
            sorted_results = sorted(model_results.items(), key=lambda x: x[1]['cv_f1_mean'], reverse=True)
            best_models = [(result[1]['model_name'], result[1]['dataset']) for result in sorted_results[:2]]
            
            print("Using cached model selection results:")
            print("Tuning best 2 models:")
            for model_name, dataset in best_models:
                f1_score = next(r[1]['cv_f1_mean'] for r in sorted_results if r[1]['model_name'] == model_name and r[1]['dataset'] == dataset)
                print(f"  - {model_name} on {dataset}: F1={f1_score:.4f}")
            print()
            
            # Hyperparameter tuning using validation set
            tuned_results = tune_best_models_proper(best_models, splits, force_retune=args.force_retune)
            
            if tuned_results:
                best_tuned = max(tuned_results.items(), key=lambda x: x[1]['best_score'])
                best_model_key = best_tuned[0]
                best_model_info = best_tuned[1]
                
                print(f"Best tuned model: {best_model_key} with Validation F1={best_model_info['best_score']:.4f}")
                print("Tuning completed successfully!")
            
            print("=" * 60)
            return
            
        if args.evaluation:
            import pandas as pd
            import os
            import pickle
            import json
            from src.models.model_evaluation import evaluate_final_model_properly
            from src.utils.data_splits import load_data_splits
            
            print("=" * 60)
            print("FINAL EVALUATION")
            print("=" * 60)
            
            # Check if data splits exist
            splits = load_data_splits()
            if splits is None:
                print("ERROR: Data splits not found!")
                print("Please run model selection first: python main.py --model-selection")
                return
            
            # Find the best tuned model
            tuning_dir = "outputs/hyperparameter_tuning"
            if not os.path.exists(tuning_dir):
                print("ERROR: No hyperparameter tuning results found!")
                print("Please run tuning first: python main.py --tuning")
                return
            
            # Load all tuning results
            tuned_results = {}
            for file in os.listdir(tuning_dir):
                if file.endswith("_best_params.json"):
                    model_dataset = file.replace("_best_params.json", "")
                    
                    # Load parameters
                    with open(os.path.join(tuning_dir, file), 'r') as f:
                        params = json.load(f)
                    
                    # Load model
                    model_file = os.path.join(tuning_dir, f"{model_dataset}_tuned_model.pkl")
                    if os.path.exists(model_file):
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)
                        
                        tuned_results[model_dataset] = {
                            **params,
                            'model': model
                        }
            
            if not tuned_results:
                print("ERROR: No tuned models found!")
                print("Please run tuning first: python main.py --tuning")
                return
            
            # Find best tuned model
            best_tuned = max(tuned_results.items(), key=lambda x: x[1]['best_score'])
            best_model_key = best_tuned[0]
            best_model_info = best_tuned[1]
            
            print(f"Best tuned model: {best_model_key} with Validation F1={best_model_info['best_score']:.4f}")
            print()
            
            # Extract model details
            model_name = best_model_info['model_name']
            dataset_name = best_model_info['dataset']
            tuned_model = best_model_info['model']
            
            # Final evaluation on held-out test set
            final_test_results = evaluate_final_model_properly(
                tuned_model, 
                splits,
                dataset_name,
                model_name
            )
            
            if final_test_results:
                print("EVALUATION SUMMARY:")
                print(f"  Test F1 (weighted): {final_test_results['test_f1_weighted']:.4f}")
                print(f"  Test Accuracy: {final_test_results['test_accuracy']:.4f}")
                print(f"  Fake News F1: {final_test_results['fake_news_f1']:.4f}")
                print(f"  Real News F1: {final_test_results['real_news_f1']:.4f}")
                if final_test_results['test_auc']:
                    print(f"  Test AUC: {final_test_results['test_auc']:.4f}")
                print()
                print("✅ Check outputs/final_evaluation/ for detailed results")
            
            print("=" * 60)
            return
            
        # Future features
        if args.predict:
            print("Single prediction feature coming soon!")
            
    except ImportError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
