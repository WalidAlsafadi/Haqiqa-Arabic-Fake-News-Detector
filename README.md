# Palestinian Fake News Detector

A production-ready Arabic NLP system that detects fake news in headlines and articles related to Palestine. Built with a scientifically rigorous machine learning pipeline featuring proper train/validation/test splits, comprehensive evaluation, and deployment-ready models.

## Project Overview

This project addresses the critical issue of misinformation about Palestine across social media platforms. Using advanced machine learning techniques and proper scientific methodology, the system achieves **91.48% F1-score** and **96.74% AUC** on held-out test data.

## Key Features

- **Arabic Text Classification** — Classifies news as real or fake with high confidence
- **Scientifically Rigorous Pipeline** — Follows ML best practices with proper data splits
- **High Performance** — 91.48% F1-score, 91.38% accuracy, 96.74% AUC
- **Advanced Arabic Processing** — Multi-level text cleaning optimized for Arabic
- **Production-Ready Models** — Saved models ready for deployment
- **Comprehensive Evaluation** — Detailed metrics, plots, and analysis
- **Modular Architecture** — Individual phase execution with professional output

## Model Performance

### Performance Metrics (Test Set - Never Seen During Training)
- **Test F1 (weighted)**: **91.48%**
- **Test Accuracy**: **91.38%** 
- **Test AUC**: **96.74%**
- **Real News F1**: **93.84%**
- **Fake News F1**: **85.66%**

### Scientific Methodology
- **Proper Data Splits**: 60% train, 20% validation, 20% test (consistent throughout)
- **No Data Leakage**: TF-IDF fitted only on training data
- **Validation-Based Tuning**: Hyperparameters optimized on validation set
- **Held-Out Testing**: Final evaluation on completely unseen test data
- **Reproducible Results**: Consistent splits saved and reused across pipeline

### Technical Architecture
- **Model**: XGBoost with optimized hyperparameters
- **Features**: TF-IDF vectorization (5000 features, Arabic-optimized)
- **Preprocessing**: Three cleaning levels (minimal, aggressive, transformers-ready) 
- **Evaluation**: Confusion matrix, ROC curve, precision-recall curves

## Tech Stack

- **Python 3.12** - Core programming language
- **Scikit-learn & XGBoost** - Machine learning frameworks
- **NLTK** - Arabic text processing  
- **Streamlit** - Web interface for deployment
- **Pandas/NumPy** - Data manipulation
- **Matplotlib/Seaborn** - Visualization and plots
- **Pickle** - Model serialization and data persistence

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/WalidAlsafadi/Palestine-Fake-News-Detector
cd Palestine-Fake-News-Detector

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the ML Pipeline

#### Complete Pipeline (Recommended)
```bash
# Run full scientific pipeline
python main.py --all
```

#### Individual Phases (For Development/Testing)
```bash
# Data preparation and cleaning
python main.py --data-prep

# Model selection with cross-validation
python main.py --model-selection

# Hyperparameter tuning on validation set
python main.py --tuning

# Final evaluation on test set
python main.py --evaluation

# Show configuration
python main.py --config
```

### Running the Web App
```bash
# Launch Streamlit interface
streamlit run app/streamlit_app.py
```

## Dataset Information
- **Size**: 4,521 Arabic news articles (after cleaning)
- **Time Period**: 2023-2025
- **Languages**: Arabic with Palestinian dialect support
- **Sources**: Verified real news and labeled fake content
- **Preprocessing**: Three cleaning levels for different model types

## Project Structure

```
Palestine-Fake-News-Detector/
├── main.py                         # Complete ML pipeline with individual phases
├── requirements.txt                # Python dependencies
├── app/
│   └── streamlit_app.py           # Web interface for predictions
├── src/
│   ├── config/
│   │   └── settings.py            # Pipeline configuration
│   ├── preprocessing/
│   │   └── text_cleaner.py        # Arabic text processing
│   ├── models/
│   │   ├── model_selection.py     # Model comparison
│   │   ├── hyperparameter_tuning_proper.py  # Parameter optimization
│   │   └── model_evaluation.py    # Final evaluation with plots
│   └── utils/
│       └── data_splits.py         # Consistent train/val/test splits
├── data/
│   ├── raw/                       # Original dataset
│   ├── processed/                 # Cleaned datasets & splits
├── models/
│   └── trained/                   # Production-ready models
├── outputs/
│   ├── model_selection/           # Cross-validation results
│   ├── hyperparameter_tuning/     # Tuning results  
│   └── final_evaluation/          # Test metrics & visualizations
└── notebooks/                     # Exploratory data analysis
```

## Scientific Methodology

This project follows rigorous machine learning practices:

1. **Consistent Data Splitting**: Same train/validation/test splits used throughout
2. **No Data Leakage**: TF-IDF vectorizer fitted only on training data
3. **Proper Validation**: Hyperparameters tuned using validation set only
4. **Unbiased Evaluation**: Test set held out until final evaluation
5. **Reproducible Results**: All splits and models saved for consistency

## Results Summary

The final model achieves state-of-the-art performance on Arabic fake news detection:

- **Model**: XGBoost with optimized hyperparameters
- **Dataset**: Minimal cleaning (best performing)
- **Cross-validation F1**: 90.49% (5-fold CV on train+validation)
- **Validation F1**: 89.52% (hyperparameter tuning)
- **Test F1**: 91.48% (final unbiased evaluation)

All results are scientifically valid with proper methodology.

## Future Enhancements

- **Deep Learning Models**: Experiment with BERT-based Arabic models
- **API Development**: RESTful API for programmatic access
- **Real-time Processing**: Live news feed monitoring
- **Multi-platform Deployment**: Docker containerization
- **Enhanced UI**: React.js frontend with advanced features
- **Continuous Learning**: Model updates with new data

## Contact

**Walid Alsafadi**  
Email: walid.k.alsafadi@gmail.com  
GitHub: [@WalidAlsafadi](https://github.com/WalidAlsafadi)  
LinkedIn: [WalidAlsafadi](https://linkedin.com/in/WalidAlsafadi)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{alsafadi2025palestine,
  title={Palestinian Fake News Detection Using Machine Learning},
  author={Alsafadi, Walid},
  year={2025},
  publisher={GitHub},
  url={https://github.com/WalidAlsafadi/Palestine-Fake-News-Detector}
}
```
