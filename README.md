# Haqiqa - Arabic Fake News Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AraBERT](https://img.shields.io/badge/AraBERT-96.22%25%20F1-orange.svg)](https://huggingface.co/aubmindlab/bert-base-arabertv02)
[![XGBoost](https://img.shields.io/badge/XGBoost-94.50%25%20F1-green.svg)](https://xgboost.readthedocs.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**Haqiqa** (حقيقة - "Truth") is a production-ready system for detecting fake news in Arabic content. Features state-of-the-art **AraBERT (96.22% F1)** and **XGBoost (94.50% F1)** models with comprehensive evaluation on 13,750 verified samples.

## 🏆 Performance Results

| Model       | F1-Score   | Accuracy   | AUC        | Inference  |
| ----------- | ---------- | ---------- | ---------- | ---------- |
| **AraBERT** | **96.22%** | **96.22%** | **99.57%** | **~200ms** |
| **XGBoost** | **94.50%** | **94.51%** | **98.94%** | **<50ms**  |

## 🚀 Quick Start

### Option 1: Pre-trained Models (Recommended)

```bash
# Clone and setup
git clone https://github.com/WalidAlsafadi/Palestine-Fake-News-Detector.git
cd Palestine-Fake-News-Detector
pip install -r requirements.txt

# Immediate testing with pre-trained XGBoost model
python inference.py
# Choose option 1 (XGBoost) - ready to use!
```

### Option 2: Complete Training Pipeline

```bash
# Train your own models from scratch
python main.py --data-prep              # Data preparation
python main.py --model-selection        # Compare 5 ML algorithms
python main.py --hyperparameter-tuning  # Optimize best model
python main.py --final-evaluation       # Test set evaluation

# Then test with your trained models
python inference.py
```

## 📊 Dataset & Methodology

- **Size**: 13,750 Arabic news samples
- **Sources**: UCAS academic dataset + Kaggle Arabic verification data
- **Distribution**: 60% train (8,250) | 20% validation (2,750) | 20% test (2,750)
- **Balance**: 46.6% real news, 53.4% fake news
- **Evaluation**: Stratified splits, 5-fold cross-validation, held-out test set

## 🛠️ Architecture

### Core Components

- **Text Processing**: 3 cleaning strategies (minimal/aggressive/transformers)
- **ML Pipeline**: Model selection → Hyperparameter tuning → Evaluation
- **Models**: XGBoost, SVM, Random Forest, Logistic Regression, Naive Bayes
- **Deep Learning**: Fine-tuned AraBERT for Arabic text classification
- **Inference**: Smart model loading and comparison interface

### Technical Stack

- **ML**: Scikit-learn, XGBoost, PyTorch, Transformers (HuggingFace)
- **NLP**: NLTK, Arabic-Stopwords, custom Arabic normalization
- **Web**: Next.js frontend, Gradio API backend
- **Data**: Pandas, NumPy, stratified sampling

## 📁 Project Structure

```
Palestine-Fake-News-Detector/
├── inference.py              # Smart model selection & testing
├── main.py                   # Complete ML pipeline
├── src/
│   ├── config/settings.py    # Centralized configuration
│   ├── preprocessing/        # Arabic text cleaning
│   ├── ml_algorithms/        # Model selection, tuning, evaluation
│   ├── transformers/arabert/ # AraBERT training & evaluation
│   └── utils/data_splits.py  # Consistent data splitting
├── data/processed/           # Clean datasets (13,750 samples)
├── saved_models/             # Trained models (XGBoost + AraBERT)
├── outputs/                  # Results, metrics, visualizations
└── webapp/                   # Web interface (Haqiqa brand)
```

## 🔧 Advanced Usage

### Adding Your Own Dataset

Want to test Haqiqa on your Arabic news dataset? Your CSV needs `text` and `label` columns (0=Real, 1=Fake):

```python
# Add your dataset to the pipeline
from src.config.settings import DATASET_PATHS
DATASET_PATHS['my_dataset'] = 'data/processed/my_dataset.csv'

# Run evaluation
python main.py --model-selection  # Test on your data
```

### Using Haqiqa in Your Projects

Want to integrate Haqiqa's trained model into your own application?

```python
from src.ml_algorithms.model_selection import load_best_pipeline
pipeline, results = load_best_pipeline()

def detect_fake_news(arabic_text):
    prediction = pipeline.predict([arabic_text])[0]
    confidence = pipeline.predict_proba([arabic_text])[0].max()
    return "Fake" if prediction == 1 else "Real", confidence

# Usage example
result, confidence = detect_fake_news("خبر عاجل من فلسطين...")
print(f"This news is {result} (confidence: {confidence:.2%})")
```

### Web Interface

```bash
# Run the modern Arabic web interface (Haqiqa brand)
cd webapp/frontend && npm install && npm run dev
# Access at http://localhost:3000
```

## 📄 License & Contact

**Licensed under Apache 2.0** - see [LICENSE](LICENSE) for details.

**Author**: Walid Alsafadi | **GitHub**: [@WalidAlsafadi](https://github.com/WalidAlsafadi)

## 🙏 Acknowledgments

- **Dr. Tareq Altalmas** - UCAS NLP Course supervision
- **aubmindlab** - AraBERT Arabic BERT model
- **HuggingFace** - Transformers library and model hosting
- **UCAS Students** - Dataset collection and annotation

---

**Haqiqa (حقيقة)** - Bringing truth to Arabic news through AI. ⭐ _Star this repo if it helps your work!_
