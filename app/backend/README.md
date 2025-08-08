# AraBERT Arabic Fake News Detector - Backend

A high-performance backend service for detecting fake news in Arabic text using a fine-tuned AraBERT model.

## 🚀 Features

- **Advanced ML Model**: Fine-tuned AraBERT model trained on Palestinian news data
- **Real-time Detection**: Fast inference with confidence scores
- **Arabic NLP**: Specialized for Arabic text processing with RTL support
- **Production Ready**: Optimized for deployment on various platforms
- **API Interface**: Clean Gradio interface for easy integration

## 🛠 Technology Stack

- **Deep Learning**: PyTorch + Transformers (Hugging Face)
- **NLP Model**: AraBERT (Arabic BERT)
- **Web Framework**: Gradio
- **Language**: Python 3.8+

## 📦 Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

The app will be available at `http://localhost:7860`

## 🔧 Model Configuration

- **Base Model**: `aubmindlab/bert-base-arabertv02`
- **Fine-tuned Model**: `WalidAlsafadi/arabert-fake-news-detector`
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

MIT License - see LICENSE file for details
