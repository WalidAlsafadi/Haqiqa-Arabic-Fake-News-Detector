---
title: Haqiqa Arabic Fake News Detector
emoji: 📰
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
---

# Haqiqa Arabic Fake News Detection API 🤖

A high-performance Gradio API for detecting fake news in Arabic text using fine-tuned AraBERT and XGBoost models.

🌐 **Live API**: [walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space](https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/)

## 🎯 Models Available

| Model       | Accuracy | Speed  | Best For         |
| ----------- | -------- | ------ | ---------------- |
| **AraBERT** | 96.22%   | ~500ms | Maximum accuracy |
| **XGBoost** | 94.51%   | ~100ms | Fast inference   |

## 🚀 API Usage

### Gradio Interface

Visit the live link above and paste Arabic text to get instant results.

### Python Integration

```python
import requests

response = requests.post(
    "https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/api/predict",
    json={
        "data": [
            "أخبار عاجلة من فلسطين حول أحداث مهمة",  # Arabic text
            "arabert"  # or "xgboost"
        ]
    }
)

result = response.json()
print(f"Prediction: {result['data']}")
```

### cURL Example

```bash
curl -X POST "https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/api/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": ["النص العربي المراد تحليله", "arabert"]}'
```

## � Response Format

```json
{
  "data": {
    "model": "arabert",
    "prediction": "Real", // or "Fake"
    "confidence": 95.5,
    "real_prob": 95.5,
    "fake_prob": 4.5
  }
}
```

- **Input**: Arabic text (max 512 tokens)
- **Output**: Binary classification (Real/Fake) with confidence scores

## 🌐 Deployment Options

### Option 1: Hugging Face Spaces

- Supports CPU Basic (free tier)
- Automatic scaling
- Built-in SSL and CDN

### Option 2: Docker

```bash
# Build container
docker build -t fake-news-detector .

# Run container
docker run -p 7860:7860 fake-news-detector
```

### Option 3: Cloud Platforms

- Compatible with AWS, GCP, Azure
- Supports both CPU and GPU instances
- Auto-scaling capabilities

## 📊 Performance

- **Accuracy**: 94.2% on Palestinian news dataset
- **Inference Time**: ~100ms per prediction (CPU)
- **Memory Usage**: ~1.2GB RAM
- **Concurrent Users**: Supports multiple simultaneous requests

## 🔒 Security

- Input validation and sanitization
- Rate limiting ready
- No data logging for privacy
- CORS enabled for frontend integration

## 📈 Monitoring

The application provides:

- Health check endpoints
- Performance metrics
- Error tracking
- Usage analytics ready

## 🤝 API Integration

The backend exposes endpoints that can be consumed by:

- Next.js frontend
- Mobile applications
- Third-party services
- API clients

## 📄 License

**Licensed under Apache 2.0** - see [LICENSE](LICENSE) for details.
