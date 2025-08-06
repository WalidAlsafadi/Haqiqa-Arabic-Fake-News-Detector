# Palestine Fake News Detector

A comprehensive machine learning system for detecting fake news in Palestinian Arabic content. Features both **traditional ML** (XGBoost) and **state-of-the-art transformer models** (AraBERT) with production-ready deployment capabilities.

## 🎯 Project Overview

This project addresses misinformation about Palestine using advanced machine learning techniques and proper scientific methodology. The system provides reliable fake news detection for Arabic content with multiple model approaches and deployment options.

## ✨ Key Features

### 🤖 Dual Model Approach

- **Traditional ML**: XGBoost with TF-IDF (90.61% accuracy, 96.53% AUC)
- **Transformer Model**: Fine-tuned AraBERT (93.48% accuracy, 98.1% AUC)

### 🔬 Professional ML Pipeline

- **Rigorous Validation**: 5-fold cross-validation, proper train/validation/test splits
- **Clean Architecture**: Modular, well-documented, industry-standard code
- **Multi-Level Processing**: Optimized Arabic cleaning for different model types
- **Production Ready**: CLI tools, web apps, and API-ready inference

### 📊 Comprehensive Evaluation

- **Detailed Metrics**: Classification reports, confusion matrices, ROC/PR curves
- **Scientific Methodology**: No data leakage, reproducible results
- **Multiple Approaches**: Traditional ML + Deep Learning comparison

## 📊 Performance Comparison

### AraBERT (Transformer Model)

- **Accuracy**: 93.48%
- **AUC**: 98.1%
- **Weighted F1-Score**: 93.53%
- **Real News F1**: 95.37%
- **Fake News F1**: 89.01%

### XGBoost (Traditional ML)

- **Accuracy**: 90.61%
- **AUC**: 96.53%
- **Weighted F1-Score**: 90.79%
- **Real News F1**: 93.19%
- **Fake News F1**: 84.85%

## 🛠️ Tech Stack

### Core ML & Data Science

- **Python 3.12** - Core programming language
- **Scikit-learn** - Traditional ML framework
- **XGBoost** - High-performance gradient boosting
- **Pandas/NumPy** - Data manipulation

### Deep Learning & Transformers

- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face transformer models
- **AraBERT** - Arabic BERT model (aubmindlab/bert-base-arabertv02)
- **Datasets** - Efficient data loading

### Arabic NLP

- **NLTK** - Arabic text processing
- **Arabic-Stopwords** - Arabic stopword removal
- **Sentencepiece** - Subword tokenization

### Deployment & Visualization

- **Streamlit** - Web application framework
- **Matplotlib/Seaborn** - Visualization
- **Joblib** - Model persistence

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

#### Traditional ML Pipeline (XGBoost)

```bash
# Run complete ML pipeline
python main.py --all

# Individual components
python main.py --data-prep
python main.py --model-selection
python main.py --tuning
python main.py --evaluation
```

#### AraBERT (Transformer Model)

```bash
# Train AraBERT model
python run_arabert_training.py

# Evaluate AraBERT model
python run_arabert_evaluation.py

# Interactive fake news detection
python detect_fake_news.py
```

#### Web Applications

```bash
# Traditional ML web app
streamlit run app/streamlit_app.py

# AraBERT inference (coming soon)
python -m src.transformers.arabert.inference
```

## 📁 Project Structure

```
Palestine-Fake-News-Detector/
├── main.py                      # Traditional ML pipeline entry point
├── run_arabert_training.py      # AraBERT training script
├── run_arabert_evaluation.py    # AraBERT evaluation script
├── detect_fake_news.py          # Interactive AraBERT CLI
├── requirements.txt             # Project dependencies
├── app/
│   └── streamlit_app.py        # Web application interface
├── src/
│   ├── config/
│   │   └── settings.py         # Configuration parameters
│   ├── preprocessing/
│   │   └── text_cleaner.py     # Arabic text processing
│   ├── models/                 # Traditional ML models
│   │   ├── model_selection.py
│   │   ├── hyperparameter_tuning.py
│   │   └── model_evaluation.py
│   ├── transformers/           # Transformer models
│   │   └── arabert/
│   │       ├── training.py     # AraBERT training
│   │       ├── evaluation.py   # AraBERT evaluation
│   │       ├── inference.py    # AraBERT inference
│   │       └── model_utils.py  # Model utilities
│   └── utils/
│       └── data_splits.py      # Consistent data splitting
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned datasets
├── models/
│   ├── trained/                # Traditional ML models
│   └── arabert/                # AraBERT models
├── outputs/                    # Results, plots, and reports
│   ├── final_evaluation/       # Traditional ML results
│   └── arabert_evaluation/     # AraBERT results
└── notebooks/                  # Exploratory data analysis
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

### AraBERT (State-of-the-Art)

- **Model**: Fine-tuned AraBERT (aubmindlab/bert-base-arabertv02)
- **Preprocessing**: Transformer-optimized Arabic cleaning
- **Training**: 7 epochs with early stopping
- **Test Accuracy**: 93.48%
- **Test AUC**: 98.1%

### XGBoost (Traditional ML Baseline)

- **Model**: XGBoost with class imbalance handling
- **Features**: TF-IDF (5000 features, Arabic-optimized)
- **Preprocessing**: Minimal cleaning (best performing)
- **Cross-Validation F1**: 89.72% ± 1.82% (5-fold CV)
- **Test Accuracy**: 90.61%
- **Test AUC**: 96.53%

Both models follow proper ML validation methodology with consistent data splits and no data leakage.

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
