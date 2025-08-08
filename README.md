# Palestine Fake News Detector 🇵🇸

A comprehensive machine learning system for detecting fake news in Palestinian Arabic content. Features **state-of-the-art transformer models** (AraBERT) with modern web applications and production-ready deployment capabilities.

## 🎯 Project Overview

This project addresses misinformation about Palestine using advanced machine learning techniques and proper scientific methodology. The system provides reliable fake news detection for Arabic content with a professional web interface and robust backend API.

## ✨ Key Features

### 🤖 Advanced ML Model

- **Transformer Model**: Fine-tuned AraBERT (94.2% accuracy, 98.1% AUC)
- **Arabic Specialized**: Optimized for Palestinian Arabic dialect
- **Real-time Inference**: Fast predictions with confidence scores

### 🌐 Modern Web Application

- **Frontend**: Next.js 15 with TypeScript, Tailwind CSS, Arabic RTL support
- **Backend**: Gradio-based API with PyTorch and Transformers
- **Responsive**: Perfect mobile and desktop experience
- **Production Ready**: Deployed on Vercel (frontend) and Hugging Face Spaces (backend)

### 🔬 Professional ML Pipeline

- **Rigorous Validation**: 5-fold cross-validation, proper train/validation/test splits
- **Clean Architecture**: Modular, well-documented, industry-standard code
- **Multi-Level Processing**: Optimized Arabic text preprocessing
- **Reproducible Results**: Consistent methodology and evaluation

## 📊 Model Performance

### AraBERT Fine-tuned Model

- **Accuracy**: 94.2%
- **AUC**: 98.1%
- **Weighted F1-Score**: 94.1%
- **Real News F1**: 95.7%
- **Fake News F1**: 91.8%
- **Inference Time**: ~100ms per prediction

## 🛠️ Tech Stack

### Frontend (Next.js Application)

- **Framework**: Next.js 15.4.6 with App Router
- **Language**: TypeScript 5.2.2
- **Styling**: Tailwind CSS with Arabic RTL support
- **UI**: Custom shadcn/ui components
- **Deployment**: Vercel

### Backend (ML API)

- **Deep Learning**: PyTorch + Transformers (Hugging Face)
- **Web Framework**: Gradio
- **Model**: Fine-tuned AraBERT
- **Deployment**: Hugging Face Spaces

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

- **Next.js + Tailwind** - Modern web application framework
- **Streamlit** - Traditional web application framework
- **Matplotlib/Seaborn** - Visualization
- **Joblib** - Model persistence
- **Gradio** - Hugging Face Space deployment

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

**حقيقة (Haqiqa) - Modern Arabic Web App**

Professional Arabic RTL web application for Palestine fake news detection.

```bash
# Navigate to modern frontend
cd webapp/frontend

# Install optimized dependencies
npm install

# Run development server
npm run dev

# Build for production deployment
npm run build
```

Visit `http://localhost:3000`

**Features:**

- ✨ Full Arabic RTL support with Cairo font
- 🎨 Modern responsive design with Tailwind CSS
- 📱 Mobile-optimized interface
- 🔍 Real-time news analysis
- 📧 Contact form integration
- 🚀 Production-ready for Vercel deployment

**Traditional Streamlit App**

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
├── requirements.txt             # Project dependencies
├── app/                        # Modern web application
│   ├── backend/               # Gradio ML API server
│   │   ├── app.py            # AraBERT model integration
│   │   ├── requirements.txt  # Backend dependencies
│   │   └── README.md         # Backend documentation
│   ├── frontend/             # Next.js web application
│   │   ├── app/              # Next.js app directory
│   │   ├── components/       # React components
│   │   ├── public/           # Static assets
│   │   ├── package.json      # Frontend dependencies
│   │   └── README.md         # Frontend documentation
│   ├── deploy.ps1            # Windows deployment script
│   ├── deploy.sh             # Unix deployment script
│   └── README.md             # App documentation
├── src/                      # Core ML pipeline and utilities
│   ├── config/              # Configuration settings
│   ├── models/              # Model definitions and training
│   ├── preprocessing/       # Data cleaning and preparation
│   ├── transformers/        # AraBERT training and inference
│   └── utils/               # Helper functions
├── data/                    # Dataset storage
│   ├── raw/                 # Original unprocessed data
│   └── processed/           # Cleaned and prepared datasets
├── models/                  # Trained model artifacts
│   ├── trained/             # Traditional ML models
│   └── arabert/             # AraBERT fine-tuned models
├── outputs/                 # Training results and evaluations
│   ├── model_selection/     # Model comparison results
│   ├── hyperparameter_tuning/ # Optimization results
│   ├── final_evaluation/    # Final model performance
│   └── arabert_evaluation/  # AraBERT specific results
├── notebooks/               # Jupyter notebooks for analysis
└── reports/                 # Research documentation and figures
```

│ ├── components/ # React components
│ ├── public/ # Static assets
│ ├── package.json
│ └── tailwind.config.ts
├── src/
│ ├── config/
│ │ └── settings.py # Configuration parameters
│ ├── preprocessing/
│ │ └── text_cleaner.py # Arabic text processing
│ ├── models/ # Traditional ML models
│ │ ├── model_selection.py
│ │ ├── hyperparameter_tuning.py
│ │ └── model_evaluation.py
│ ├── transformers/ # Transformer models
│ │ └── arabert/
│ │ ├── training.py # AraBERT training
│ │ ├── evaluation.py # AraBERT evaluation
│ │ ├── inference.py # AraBERT inference
│ │ └── model_utils.py # Model utilities
│ └── utils/
│ └── data_splits.py # Consistent data splitting
├── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Cleaned datasets
├── models/
│ ├── trained/ # Traditional ML models
│ └── arabert/ # AraBERT models
├── outputs/ # Results, plots, and reports
│ ├── final_evaluation/ # Traditional ML results
│ └── arabert_evaluation/ # AraBERT results
└── notebooks/ # Exploratory data analysis

````

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

## 🚀 Live Demo & Deployment

### Frontend Demo

The modern web application is ready for deployment and features:

- **Portfolio-ready design** with Arabic RTL support
- **Real-time AraBERT integration** via Hugging Face Spaces
- **Responsive interface** optimized for all devices
- **Professional UI/UX** with modern design patterns

### Hugging Face Space

Experience the AraBERT model live:

- **Model:** [walidalsafadi/arabert-fake-news-detector](https://huggingface.co/spaces/walidalsafadi/arabert-fake-news-detector)
- **API:** Available for real-time predictions
- **Performance:** 93.48% accuracy on Palestinian Arabic news

### Deployment Options

1. **Vercel/Netlify** - Frontend deployment
2. **Hugging Face Spaces** - Model hosting (already deployed)
3. **Docker** - Complete containerized deployment
4. **Traditional hosting** - Via build output

## Contact

**Walid Alsafadi**
- Email: walid.k.alsafadi@gmail.com
- GitHub: [@WalidAlsafadi](https://github.com/WalidAlsafadi)
- Hugging Face: [@WalidAlsafadi](https://huggingface.co/WalidAlsafadi)
- LinkedIn: [in/WalidAlsafadi](https://linkedin.com/in/WalidAlsafadi)

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
````
