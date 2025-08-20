# حقيقة (Haqiqa) - Arabic Fake News Detection Web Application

Professional web application for detecting fake news in Arabic content, featuring a modern Next.js frontend and powerful AraBERT backend API.

## 🌟 Features

### 🎨 Frontend: Modern Arabic Web Interface

- ✨ **Full Arabic RTL Support** - Native right-to-left layout with Cairo font
- 📱 **Responsive Design** - Perfect mobile and desktop experience
- 🔍 **Real-time Analysis** - Instant news credibility detection
- 🎯 **Dual Model Selection** - Choose between AraBERT and XGBoost models
- � **Confidence Visualization** - Interactive circular progress indicators
- 🚀 **Production Ready** - Optimized for Vercel deployment

### 🤖 Backend: AraBERT ML API

- 🧠 **High Accuracy Models** - AraBERT: 96.22%, XGBoost: 94.51%
- ⚡ **Fast Inference** - Sub-second response times
- 🌐 **RESTful API** - Clean JSON endpoints
- � **Detailed Results** - Confidence scores and probability distributions
- � **Production Ready** - Deployed on Hugging Face Spaces

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+ and pip (for backend)

### Frontend Setup

1. **Navigate to frontend directory:**

   ```bash
   cd webapp/frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Configure environment:**

   ```bash
   cp .env.example .env.local
   # Edit .env.local with your Hugging Face Space URL
   ```

4. **Run development server:**

   ```bash
   npm run dev
   ```

   Visit [http://localhost:3000](http://localhost:3000)

5. **Build for production:**
   ```bash
   npm run build && npm run start
   ```

### Backend Setup

1. **Navigate to backend directory:**

   ```bash
   cd webapp/backend
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server:**

   ```bash
   python app.py
   ```

   Access API at [http://localhost:7860](http://localhost:7860)

## 📊 Model Performance

| Model       | Accuracy | F1-Score | AUC    | Speed  |
| ----------- | -------- | -------- | ------ | ------ |
| **AraBERT** | 96.22%   | 96.22%   | 99.57% | ~500ms |
| **XGBoost** | 94.51%   | 94.50%   | 98.94% | ~100ms |

_Trained on 13,750 Arabic news articles_

## 🔧 Tech Stack

### Frontend

- **Framework:** Next.js 15 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **UI Components:** Radix UI + shadcn/ui
- **Icons:** Lucide React
- **Visualization:** react-circular-progressbar

### Backend

- **Framework:** Gradio
- **ML Models:** AraBERT, XGBoost
- **Language:** Python
- **Deployment:** Hugging Face Spaces

## 📁 Project Structure

```
webapp/
├── frontend/                 # Next.js application
│   ├── app/                 # App router pages
│   │   ├── api/analyze/     # Backend API integration
│   │   ├── layout.tsx       # Root layout
│   │   └── page.tsx         # Home page
│   ├── components/          # React components
│   │   ├── ui/             # shadcn/ui components
│   │   ├── About.tsx       # About section
│   │   ├── NewsAnalyzer.tsx # Main analyzer interface
│   │   └── ...             # Other components
│   ├── lib/                # Utilities
│   │   └── hf.ts           # Hugging Face API client
│   └── public/             # Static assets
│
├── backend/                 # Gradio API server
│   ├── app.py              # Main API application
│   ├── requirements.txt    # Python dependencies
│   └── *.pkl              # Trained model files
│
├── deploy.sh               # Deployment scripts
├── deploy.ps1
└── README.md               # This file
```

## 🚀 Deployment

### Frontend (Vercel)

1. **Connect repository to Vercel**
2. **Configure build settings:**
   - Build Command: `cd webapp/frontend && npm run build`
   - Output Directory: `webapp/frontend/.next`
3. **Set environment variables:**
   - `HF_SPACE_URL`: Your Hugging Face Space URL

### Backend (Hugging Face Spaces)

1. **Create new Space on Hugging Face:**

   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Choose SDK: Gradio
   - Set visibility: Public

2. **Upload backend files:**

   ```bash
   cd webapp/backend
   # Upload all files to your Space repository
   ```

3. **Your Space will be available at:**
   `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

## 🔑 Environment Variables

### Frontend (.env.local)

```bash
HF_SPACE_URL="https://your-username-your-space.hf.space"
HF_API_TOKEN=""  # Optional, for private spaces
```

## 📱 API Usage

### Analyze Text Endpoint

```http
POST /api/analyze
Content-Type: application/json

{
  "text": "النص الإخباري المراد تحليله",
  "model": "arabert" | "xgboost"
}
```

### Response

```json
{
  "ok": true,
  "data": {
    "model": "arabert",
    "prediction": "Real" | "Fake",
    "confidence": 95.5,
    "real_prob": 95.5,
    "fake_prob": 4.5
  }
}
```

## 🛠️ Development

### Code Quality

```bash
# Frontend linting
cd webapp/frontend
npm run lint

# Type checking
npx tsc --noEmit
```

### Testing

```bash
# Frontend build test
npm run build

# Backend functionality test
cd webapp/backend
python app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For questions and support:

- 📧 Email: [Contact Information]
- 🐛 Issues: [GitHub Issues](https://github.com/WalidAlsafadi/Haqiqa-Arabic-Fake-News-Detector/issues)

---

Made with ❤️ for Arabic content verification

- Upload `app.py`
- Upload `requirements.txt`
- Upload `README.md`

4. **Your Space will automatically build and deploy!**

   - URL will be: `https://YOUR_USERNAME-arabert-fake-news-detector.hf.space/`

5. **Run locally:**

   ```bash
   npm run dev
   ```

6. **Deploy to Vercel:**
   - Push to GitHub
   - Connect repository to [Vercel](https://vercel.com)
   - Deploy with zero configuration

## 📁 Project Structure

```
webapp/
├── backend/                 # Gradio backend for Hugging Face Spaces
│   ├── app.py              # Main Gradio application
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # Backend documentation
│
└── frontend/               # Next.js frontend application
    ├── app/                # Next.js App Router
    │   ├── globals.css     # Global styles with Tailwind
    │   ├── layout.tsx      # Root layout with RTL support
    │   ├── page.tsx        # Main application page
    │   └── providers/      # React providers (theme)
    ├── public/             # Static assets
    ├── package.json        # Dependencies and scripts
    ├── tailwind.config.js  # Tailwind CSS configuration
    └── README.md           # Frontend documentation
```

## 🛠 Technology Stack

### Backend

- **Gradio** - Web interface framework
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face transformers library
- **AraBERT** - Pre-trained Arabic BERT model

### Frontend

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling and responsive design
- **@gradio/client** - Hugging Face Spaces integration

## 🔧 Configuration

### Backend Configuration

The backend loads your fine-tuned AraBERT model from Hugging Face Hub. Update the model path in `backend/app.py`:

```python
model_path = "YOUR_USERNAME/arabert-fake-news-detector"  # Your model on Hugging Face
```

### Frontend Configuration

Update the Hugging Face Space URL in `frontend/app/page.tsx`:

```typescript
const app = await client("YOUR_USERNAME/arabert-fake-news-detector");
```

## 🌟 Features

### Backend Features

- ✅ Real-time Arabic text analysis
- ✅ Confidence scoring and probabilities
- ✅ RTL-optimized Gradio interface
- ✅ Error handling and validation
- ✅ Example news samples
- ✅ Optimized for CPU deployment (free tier)

### Frontend Features

- ✅ Professional, portfolio-ready design
- ✅ Arabic RTL interface
- ✅ Dark/Light mode toggle
- ✅ Responsive design (mobile-first)
- ✅ Smooth animations and transitions
- ✅ Real-time prediction display
- ✅ Detailed confidence indicators
- ✅ Input validation and error handling

## 🚀 Deployment Instructions

### 1. Deploy Backend to Hugging Face Spaces

```bash
# 1. Create account on huggingface.co
# 2. Create new Space with Gradio SDK
# 3. Upload files from webapp/backend/
# 4. Space will auto-build and deploy
```

### 2. Deploy Frontend to Vercel

```bash
# 1. Push code to GitHub
git add .
git commit -m "Add webapp with backend and frontend"
git push origin main

# 2. Connect to Vercel
# 3. Import GitHub repository
# 4. Deploy with zero configuration
```

### 3. Connect Frontend to Backend

After both are deployed, update the frontend's backend URL:

```typescript
// In frontend/app/page.tsx
const app = await client("YOUR_USERNAME/arabert-fake-news-detector");
```

## 📱 Usage

1. **Enter Arabic news text** in the input field
2. **Click "تحليل الخبر"** (Analyze News) or press Ctrl + Enter
3. **View results** with prediction, confidence score, and detailed probabilities

## 🔍 API Integration

The frontend communicates with the Gradio backend using `@gradio/client`:

```typescript
// Connect to Hugging Face Space
const app = await client("username/space-name");

// Call prediction endpoint
const response = await app.predict("/predict", [arabicText]);

// Process response
const result = parseGradioResponse(response);
```

## 🎨 Customization

### Styling

- Modify `frontend/tailwind.config.js` for design tokens
- Update `frontend/app/globals.css` for custom styles
- Change colors, fonts, and animations as needed

### Backend Model

- Replace model path in `backend/app.py`
- Adjust preprocessing and postprocessing logic
- Modify confidence thresholds

### Frontend Behavior

- Update prediction parsing logic in `frontend/app/page.tsx`
- Modify error handling and validation
- Customize animations and transitions

## 🌐 Live Demo

- **Backend**: `https://YOUR_USERNAME-arabert-fake-news-detector.hf.space/`
- **Frontend**: `https://arabert-fake-news-detector.vercel.app/`

## 📄 Environment Variables

### Frontend (.env.local)

```bash
# Optional: Custom backend URL
NEXT_PUBLIC_GRADIO_URL=https://your-space.hf.space
```

### Backend

No environment variables required - uses Hugging Face Hub for model loading.

## 🐛 Troubleshooting

### Common Issues

1. **Model loading errors**:

   - Ensure your model is public on Hugging Face Hub
   - Check model path in `backend/app.py`

2. **Frontend connection errors**:

   - Verify Hugging Face Space URL
   - Check CORS settings (Gradio handles this automatically)

3. **Deployment issues**:
   - Check requirements.txt for correct dependencies
   - Ensure all files are uploaded to respective platforms

### Debug Mode

Enable debug mode in frontend:

```typescript
// Add to app/page.tsx
console.log("Response:", response);
```

## 📈 Performance Optimization

### Backend

- Model runs on CPU (free tier compatible)
- Optimized tokenization and inference
- Proper error handling for timeouts

### Frontend

- Lazy loading and code splitting
- Optimized bundle size
- Fast refresh during development

## 🔒 Security

- Client-side only (no sensitive data stored)
- HTTPS enforcement
- Input sanitization and validation
- Rate limiting handled by Hugging Face Spaces

## 📞 Support

For issues or questions:

1. Check the troubleshooting section
2. Review logs in browser console
3. Check Hugging Face Space logs
4. Open an issue in the repository

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both backend and frontend
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for hosting and transformers library
- **AraBERT team** for the pre-trained Arabic model
- **Vercel** for frontend hosting
- **Next.js and Tailwind CSS** for the development framework

---

**🚀 Ready to deploy your Arabic fake news detector!**

**Built with ❤️ by Walid Alsafadi**
