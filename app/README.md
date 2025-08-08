# Palestine Fake News Detector - Web Applications

Professional web applications for detecting fake news in Palestinian Arabic content. Featuring a modern Next.js frontend and powerful AraBERT backend API.

## 🌟 Applications Overview

### 🎨 Frontend: حقيقة (Haqiqa) - Modern Arabic Web App

Professional Arabic RTL web application built with Next.js 15, TypeScript, and Tailwind CSS.

**Key Features:**

- ✨ Full Arabic RTL support with Cairo font
- 📱 Perfect responsive design with mobile navigation
- 🔍 Real-time news analysis interface
- 📧 Contact form integration
- 🚀 Production-ready for Vercel deployment
- 🎯 Smooth scrolling with perfect mobile experience

### 🤖 Backend: AraBERT ML API

High-performance Gradio-based API for Arabic fake news detection.

**Key Features:**

- 🧠 Fine-tuned AraBERT model (94.2% accuracy)
- ⚡ Fast inference (~100ms per prediction)
- 🌐 RESTful API endpoints
- 📊 Confidence scores and detailed results
- 🚀 Ready for Hugging Face Spaces deployment

## 🚀 Quick Start

### Frontend Development

1. **Navigate to frontend:**

   ```bash
   cd app/frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Run development server:**

   ```bash
   npm run dev
   ```

4. **Build for production:**
   ```bash
   npm run build
   ```

### Backend Development

1. **Navigate to backend:**

   ```bash
   cd app/backend
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server:**

   ```bash
   python app.py
   ```

   Access at: `http://localhost:7860`

Visit [http://localhost:3000](http://localhost:3000) to view the application.

### Backend Deployment (Hugging Face Spaces)

1. **Navigate to backend directory:**

   ```bash
   cd webapp/backend
   ```

2. **Create a new Hugging Face Space:**

   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose SDK: Gradio
   - Set visibility: Public
   - Name your space (e.g., `arabert-fake-news-detector`)

3. **Upload files to your Space:**

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
