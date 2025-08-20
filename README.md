# Haqiqa - Arabic Fake News Detector 🔍

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AraBERT](https://img.shields.io/badge/AraBERT-96.22%25%20F1-orange.svg)](https://huggingface.co/aubmindlab/bert-base-arabertv02)
[![XGBoost](https://img.shields.io/badge/XGBoost-94.50%25%20F1-green.svg)](https://xgboost.readthedocs.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**Haqiqa** (حقيقة - "Truth") is a production-ready system for detecting fake news in Arabic content. Features state-of-the-art **AraBERT (96.22% F1)** and **XGBoost (94.50% F1)** models with comprehensive evaluation on 13,750 verified samples.

## � Live Demo

- **🚀 Web App**: [haqiqaa.vercel.app](https://haqiqaa.vercel.app) - Complete Arabic interface
- **🤖 API**: [HuggingFace Space](https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/) - Direct model access
- **📱 Repository**: [GitHub](https://github.com/WalidAlsafadi/Haqiqa-Arabic-Fake-News-Detector)

## 🏆 Performance Achievements

| Model       | F1-Score   | Accuracy   | AUC        | Inference Speed |
| ----------- | ---------- | ---------- | ---------- | --------------- |
| **AraBERT** | **96.22%** | **96.22%** | **99.57%** | **~500ms**      |
| **XGBoost** | **94.50%** | **94.51%** | **98.94%** | **~100ms**      |

_Trained and evaluated on 13,750 Arabic news articles_

## 🚀 Quick Usage

### Try the Live Web App

Visit **[haqiqaa.vercel.app](https://haqiqaa.vercel.app)** for the full Arabic interface with real-time analysis.

### Use Pre-trained Models Locally

```bash
# Clone and setup
git clone https://github.com/WalidAlsafadi/Haqiqa-Arabic-Fake-News-Detector.git
cd Haqiqa-Arabic-Fake-News-Detector
pip install -r requirements.txt

# Test with pre-trained models
python inference.py
```

### API Integration

```python
import requests

# Using HuggingFace Space API
response = requests.post(
    "https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/api/predict",
    json={"data": ["أخبار عاجلة من فلسطين", "arabert"]}
)
result = response.json()
```

## 📊 Research & Dataset

- **Dataset Size**: 13,750 Arabic news articles
- **Sources**: UCAS academic collection + Kaggle Arabic verification data
- **Evaluation**: Stratified 60/20/20 split with 5-fold cross-validation
- **Balance**: 46.6% real news, 53.4% fake news
- **Focus**: Palestinian and general Arabic news content

## 🛠️ Technical Architecture

### Production Stack

- **Frontend**: Next.js 15 + TypeScript + Tailwind CSS → [Vercel](https://haqiqaa.vercel.app)
- **Backend**: Gradio API + AraBERT + XGBoost → [HuggingFace Spaces](https://huggingface.co/spaces/WalidAlsafadi/Haqiqa-Arabic-Fake-News-Detector)
- **Models**: Fine-tuned AraBERT, optimized XGBoost
- **Infrastructure**: Automated deployment pipeline

### Core Components

- **Text Processing**: 3 cleaning strategies (minimal/aggressive/transformers-ready)
- **ML Pipeline**: Model selection → Hyperparameter tuning → Evaluation
- **Deep Learning**: Fine-tuned AraBERT for Arabic text classification
- **Web Interface**: RTL Arabic support with confidence visualization

## 📁 Project Structure

```
Haqiqa-Arabic-Fake-News-Detector/
├── 📊 Research & Training
│   ├── inference.py              # Smart model testing interface
│   ├── main.py                   # Complete ML pipeline
│   ├── src/                      # Core ML algorithms & preprocessing
│   ├── data/processed/           # Clean datasets (13,750 samples)
│   ├── saved_models/             # Trained models (AraBERT + XGBoost)
│   └── outputs/                  # Results, metrics, visualizations
└── 🌐 Production Web App
    ├── webapp/frontend/          # Next.js Arabic interface
    └── webapp/backend/           # Gradio API server
```

## 🔧 For Developers

### Training Your Own Models

```bash
# Complete training pipeline
python main.py --data-prep              # Data preparation
python main.py --model-selection        # Compare 5 ML algorithms
python main.py --hyperparameter-tuning  # Optimize best model
python main.py --final-evaluation       # Test set evaluation
```

### Adding Custom Datasets

```python
# Your CSV needs 'text' and 'label' columns (0=Real, 1=Fake)
from src.config.settings import DATASET_PATHS
DATASET_PATHS['my_dataset'] = 'data/processed/my_dataset.csv'
python main.py --model-selection
```

### Integration Example

```python
from src.ml_algorithms.model_selection import load_best_pipeline
pipeline, results = load_best_pipeline()

def detect_fake_news(arabic_text):
    prediction = pipeline.predict([arabic_text])[0]
    confidence = pipeline.predict_proba([arabic_text])[0].max()
    return "Fake" if prediction == 1 else "Real", confidence

# Usage
result, confidence = detect_fake_news("خبر عاجل من فلسطين...")
print(f"This news is {result} (confidence: {confidence:.2%})")
```

### Local Web Development

```bash
# Frontend (Next.js)
cd webapp/frontend && npm install && npm run dev
# → http://localhost:3000

# Backend (Gradio)
cd webapp/backend && python app.py
# → http://localhost:7860
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
