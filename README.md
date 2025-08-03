# Palestine Fake News Detector

A professional machine learning system for detecting fake news in Palestinian Arabic content. Built with industry-standard ML practices, achieving **90.61% accuracy** and **96.53% AUC** on held-out test data.

## 🎯 Project Overview

This project addresses misinformation about Palestine using advanced machine learning and proper scientific methodology. The system provides reliable fake news detection for Arabic content with production-ready deployment.

## ✨ Key Features

- **High-Performance Arabic NLP** — 90.61% test accuracy, 96.53% AUC
- **Professional ML Pipeline** — 5-fold cross-validation, proper train/validation/test splits
- **Clean Architecture** — Modular, well-documented, industry-standard code
- **Multi-Level Text Processing** — Optimized Arabic cleaning for different model types
- **Production Ready** — Streamlit web app with trained models
- **Comprehensive Evaluation** — Detailed metrics, visualizations, and analysis

## 📊 Performance Metrics

### Test Set Results (Never Seen During Training)

- **Accuracy**: 90.61%
- **AUC**: 96.53%
- **F1-Score**: 90.49%
- **Precision**: 91.30%
- **Recall**: 89.72%

### Scientific Methodology

- **Data Splits**: 60% train, 20% validation, 20% test (consistent throughout)
- **Cross-Validation**: 5-fold CV on training data for model selection
- **No Data Leakage**: Proper train/validation methodology
- **Reproducible**: Fixed random seeds and saved splits

## 🛠️ Tech Stack

- **Python 3.12** - Core programming language
- **XGBoost** - High-performance gradient boosting
- **Scikit-learn** - Machine learning framework
- **NLTK** - Arabic text processing
- **Streamlit** - Web application framework
- **Pandas/NumPy** - Data manipulation
- **Matplotlib/Seaborn** - Visualization

## 🚀 Quick Start

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

### Usage

#### Complete ML Pipeline

```bash
# Run full pipeline (data prep → model selection → tuning → evaluation)
python main.py --all
```

#### Individual Components

```bash
# Data preparation
python main.py --data-prep

# Model selection with 5-fold cross-validation
python main.py --model-selection

# Hyperparameter tuning
python main.py --tuning

# Final evaluation
python main.py --evaluation
```

#### Web Application

```bash
# Launch Streamlit app
streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```
Palestine-Fake-News-Detector/
├── main.py                    # Complete ML pipeline entry point
├── requirements.txt           # Project dependencies
├── app/
│   └── streamlit_app.py      # Web application interface
├── src/
│   ├── config/
│   │   └── settings.py       # Configuration parameters
│   ├── preprocessing/
│   │   └── text_cleaner.py   # Arabic text processing
│   ├── models/
│   │   ├── model_selection.py    # Cross-validation & model comparison
│   │   ├── hyperparameter_tuning.py  # Parameter optimization
│   │   └── model_evaluation.py   # Final evaluation & metrics
│   └── utils/
│       └── data_splits.py    # Consistent data splitting
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned datasets
├── models/
│   └── trained/              # Saved models & vectorizers
├── outputs/                  # Results, plots, and reports
└── notebooks/                # Exploratory data analysis
```

## 📈 Methodology

This project follows ML industry best practices:

1. **Proper Data Splitting**: Consistent 60/20/20 train/validation/test splits
2. **Cross-Validation**: 5-fold CV on training data for model selection
3. **No Data Leakage**: TF-IDF fitted only on training data
4. **Validation Tuning**: Hyperparameters optimized on validation set
5. **Unbiased Testing**: Final evaluation on held-out test set
6. **Reproducible Results**: Fixed random seeds and saved artifacts

## 🎯 Results Summary

The optimized XGBoost model achieves excellent performance:

- **Model**: XGBoost with class imbalance handling
- **Features**: TF-IDF (5000 features, Arabic-optimized)
- **Preprocessing**: Minimal cleaning (best performing)
- **Cross-Validation F1**: 89.72% ± 1.82% (5-fold CV)
- **Final Test Accuracy**: 90.61%
- **Test AUC**: 96.53%

All results follow proper ML validation methodology.

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
