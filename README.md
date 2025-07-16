# 🇵🇸 Palestinian Fake News Detector

A production-ready Arabic NLP application that detects fake news in headlines and short articles related to Palestine. Built with a complete machine learning pipeline and deployed with a modern Streamlit interface.

## 🧠 Project Overview

Misinformation about Palestine has surged across platforms like Telegram and social media. This app uses a robust XGBoost model trained on real-world Arabic news data (2023–2025) to classify whether input is **real** or **fake**, with high confidence and transparency.

## 🚀 Features

- 🔍 **Arabic Text Classification** — identifies news as **حقيقي (real)** or **كاذب (fake)**
- 📊 **Confidence Score** — easy-to-read reverse confidence output
- 🧹 **Advanced Arabic Preprocessing** — includes normalization, diacritics removal, and stopword filtering
- 🧪 **TF-IDF + XGBoost Pipeline** — built using Scikit-learn pipelines and trained with cross-validation
- 🖥 **Deployed Arabic UI** — live app with prediction history, cleaning preview, and character count
- 📊 **Professional EDA** — detailed analysis on data distribution and imbalance
- 📁 **Modular Codebase** — clean structure with logging and separation of concerns

## 🧪 Model & Dataset

- **Model**: XGBoost (with TF-IDF features), trained as a full pipeline
- **Evaluation**:
  - F1-score: **0.846**
  - Accuracy: **0.91**
  - 5-fold CV: Mean F1 **0.829**, Std **±0.018**
- **Data**:
  - ~5,352 Arabic news entries
  - Date range: 2023–2025
  - Sources: Al Jazeera (real), Tibyan & Misbar (fake), other social posts

## 📦 Tech Stack

- Python 3.12
- Scikit-learn
- XGBoost
- Streamlit
- NLTK
- Pandas / NumPy
- Arabic-Stopwords
- Matplotlib, Seaborn, WordCloud (EDA)

## ▶️ How to Run

```bash
$ git clone https://github.com/WalidAlsafadi/Palestine-Fake-News-Detector
$ cd Palestine-Fake-News-Detector

# 2. Create and activate virtual env (optional but recommended)
$ python -m venv venv
$ source venv/bin/activate  # or venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Run the app
$ streamlit run app/app.py
```

> Requires Python ≥ 3.10 and an internet connection for Streamlit assets.

## 🌍 Live Demo

The app is live on **Streamlit Cloud**:
🔗 [View Demo](https://palestine-fake-news-detector.streamlit.app/)

## 📁 Project Structure

```
├── app.py                    # Arabic Streamlit UI
├── src/
│   ├── preprocessing/        # Arabic text cleaning
│   ├── features/             # TF-IDF vectorizer
│   ├── training/             # Model selection, training
│   └── utils/                # Logging system
├── models/                   # Saved model pipeline (.pkl)
├── data/                     # Cleaned datasets
├── outputs/                  # Evaluation reports, test logs
└── requirements.txt
```

## 🔭 Future Enhancements

- **Frontend Modernization**: Build a React.js web interface for a smoother, more interactive user experience.
- **API-Driven Backend**: Serve the trained ML model via a REST API using FastAPI or Flask, separating backend logic from the UI.
- **User & History Database**:
  - Add user authentication and session management.
  - Store prediction history per user in a secure database (e.g., PostgreSQL, SQLite).
  - Enable analytics dashboards or personalized feedback.
- **Model Upgrades**:
  - Experiment with deep learning models (e.g., CNN, BiLSTM).
  - Explore transformer-based models (e.g., AraBERT, XLM-R).
  - Try lightweight LLMs fine-tuned on Arabic/PNA news.
- **Transfer Learning**: Leverage pre-trained embeddings and Arabic-language models for faster convergence and better accuracy.
- **Brand & UX Identity**: Package the app under a recognizable brand with a custom domain, logo, and consistent UI design.
- **Continuous Improvement**:
  - Automate data collection and labeling pipelines.
  - Implement feedback loop to refine model over time.
- **Multilingual Support**: Expand to detect fake news in English, or other regional languages.
- **Data Dashboard**: Build a live dashboard to show prediction trends, platform sources, and time-based fake news distribution.

## 📬 Contact

Developed by **Walid Alsafadi**  
📧 walid.k.alsafadi@gmail.com  
🔗 [GitHub](https://github.com/WalidAlsafadi) | [LinkedIn](https://linkedin.com/in/WalidAlsafadi) | [X](https://x.com/WalidAlsafadi)
