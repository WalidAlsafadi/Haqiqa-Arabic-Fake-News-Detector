---
title: Haqiqa Arabic Fake News Detector
emoji: 📰
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.40.0
app_file: app.py
pinned: false
license: apache-2.0
---

# حقيقة (Haqiqa) - AI-Powered Arabic News Verification API 🤖

**High-performance machine learning API for detecting fake news in Arabic content**

🌐 **Live API**: [walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space](https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/)

---

## 🎯 What This API Does

Haqiqa API provides instant fake news detection for Arabic text using advanced machine learning models. It analyzes news content and returns confidence scores to help identify potentially misleading information.

## ✨ Key Features

### 🚀 **Dual AI Models**

<div align="left">

| Model       | Accuracy | Speed  | Best For         |
| ----------- | -------- | ------ | ---------------- |
| **AraBERT** | 96.22%   | ~500ms | Maximum accuracy |
| **XGBoost** | 94.51%   | ~100ms | Fast inference   |

</div>

### ⚡ **Performance**

- Real-time analysis (100-500ms response time)
- Supports multiple concurrent requests
- Optimized for Arabic language processing
- 99%+ uptime on HuggingFace Spaces

### 🔒 **Privacy & Security**

- No data logging or storage
- Input validation and sanitization
- Rate limiting ready
- CORS enabled for web integration

## 🌐 How to Use the API

### 1. **Web Interface** (Easiest)

Visit the [live interface](https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/) and paste Arabic text for instant results.

### 2. **REST API Integration**

**Python Example:**

```python
import requests

response = requests.post(
    "https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/api/predict",
    json={
        "data": [
            "النص العربي المراد تحليله",  # Your Arabic text
            "arabert"  # Model: "arabert" or "xgboost"
        ]
    }
)

result = response.json()
print(f"Prediction: {result['data']['prediction']}")
print(f"Confidence: {result['data']['confidence']}%")
```

**JavaScript Example:**

```javascript
const response = await fetch(
  "https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/api/predict",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data: ["النص العربي المراد تحليله", "arabert"],
    }),
  }
);

const result = await response.json();
console.log(result.data);
```

**cURL Example:**

```bash
curl -X POST "https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/api/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": ["النص العربي المراد تحليله", "arabert"]}'
```

## 📊 API Response Format

```json
{
  "data": {
    "model": "arabert",
    "prediction": "Real", // "Real" or "Fake"
    "confidence": 95.5, // Overall confidence (0-100)
    "real_prob": 95.5, // Probability of being real (0-100)
    "fake_prob": 4.5 // Probability of being fake (0-100)
  }
}
```

## 🎯 Use Cases

- **News Verification**: Verify authenticity of news articles
- **Social Media Monitoring**: Check posts and messages
- **Content Moderation**: Filter potentially false information
- **Research**: Academic studies on misinformation
- **Browser Extensions**: Real-time fact-checking tools

## 🛠️ Technical Specifications

### Input Requirements

- **Text**: Arabic content (any length, optimized for news articles)
- **Format**: Plain text string
- **Encoding**: UTF-8
- **Models**: Choose between "arabert" or "xgboost"

### Performance Metrics

<div align="left">

**AraBERT Model:**

- Accuracy: 96.22%
- AUC: 99.57%
- F1-Score: 96.22%
- Processing: ~500ms

**XGBoost Model:**

- Accuracy: 94.51%
- AUC: 98.94%
- F1-Score: 94.50%
- Processing: ~100ms

</div>

## 🚀 Deployment Options

### **Hugging Face Spaces** (Current)

- Free hosting with automatic scaling
- Built-in SSL and global CDN
- No setup required - just use the API

### **Self-Hosting with Docker**

```bash
# Clone and build
git clone https://github.com/WalidAlsafadi/Haqiqa-Arabic-Fake-News-Detector.git
cd webapp/backend
docker build -t haqiqa-api .
docker run -p 7860:7860 haqiqa-api
```

### **Cloud Deployment**

- Compatible with AWS, GCP, Azure
- Supports both CPU and GPU instances
- Auto-scaling capabilities available

## 🔧 Development Setup

```bash
# Clone repository
git clone https://github.com/WalidAlsafadi/Haqiqa-Arabic-Fake-News-Detector.git
cd webapp/backend

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
# API available at http://localhost:7860
```

## 📁 Project Structure

```
backend/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── models/            # Pre-trained model files
├── utils/             # Helper functions
└── README.md          # This file
```

## 🤝 Integration Examples

### **Frontend Integration**

Perfect for web applications, mobile apps, and browser extensions.

### **Backend Services**

Integrate into existing APIs and microservices architectures.

### **Research Tools**

Use for academic research and misinformation studies.

## 📊 System Requirements

- **Memory**: ~1.2GB RAM minimum
- **CPU**: Any modern CPU (GPU optional for faster inference)
- **Storage**: ~500MB for model files
- **Network**: Stable internet connection for API calls

## 🤝 Support & Contributing

For questions, issues, or contributions:

- Open issues on GitHub
- Submit pull requests for improvements
- Contact the development team

## 📄 License

Licensed under the Apache 2.0 License - see [LICENSE](../../LICENSE) for details.

---

**حقيقة (Haqiqa)** - Advanced AI technology for Arabic news verification.
