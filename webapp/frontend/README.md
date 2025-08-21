# حقيقة (Haqiqa) - Arabic News Verification Platform 📰

**Advanced AI-powered platform for detecting fake news in Arabic content**

🌐 **Live Application**: [haqiqaa.vercel.app](https://haqiqaa.vercel.app)

---

## 🎯 What is Haqiqa?

Haqiqa (حقيقة - meaning "Truth" in Arabic) is an intelligent web application that helps users verify the authenticity of Arabic news content. Using state-of-the-art AI models, it analyzes text and provides confidence scores to help combat misinformation in Arabic media.

## ✨ Key Features

### 🔍 **Instant Analysis**

- Real-time fake news detection
- Visual confidence indicators with circular progress bars
- Detailed probability breakdown (Real vs Fake)

### 🤖 **Dual AI Models**

- **AraBERT**: High accuracy (96.22%) for detailed analysis
- **XGBoost**: Fast processing (94.51%) for quick verification

### 📱 **Mobile-First Design**

- Responsive layout optimized for all devices
- Native Arabic RTL support
- Clean, intuitive interface

### 🎨 **Smart User Experience**

- Input section positioned first on mobile
- Results displayed clearly at bottom
- Character counter and model selection
- Professional Arabic typography with Cairo font

## 🚀 How to Use

1. **Visit the Application**: Go to [haqiqaa.vercel.app](https://haqiqaa.vercel.app)
2. **Enter Arabic Text**: Paste your news content in the text area
3. **Choose Model**: Select AraBERT (accurate) or XGBoost (fast)
4. **Get Results**: View confidence scores and authenticity assessment

## 📊 Model Performance

<div align="left">

**AraBERT (Transformers)**

- Accuracy: 96.22%
- AUC: 99.57%
- F1-Score: 96.22%

**XGBoost (Machine Learning)**

- Accuracy: 94.51%
- AUC: 98.94%
- F1-Score: 94.50%

</div>

## 🛠️ Technical Features

- **Framework**: Next.js 15 with TypeScript
- **Styling**: Tailwind CSS with RTL support
- **UI Components**: shadcn/ui design system
- **Deployment**: Vercel with automatic scaling
- **Performance**: Optimized for speed and reliability

## 🏗️ Development Setup

```bash
# Clone and install
git clone https://github.com/WalidAlsafadi/Haqiqa-Arabic-Fake-News-Detector.git
cd webapp/frontend
npm install

# Configure environment
cp .env.example .env.local
# Add your HuggingFace Space URL

# Start development
npm run dev
# Visit http://localhost:3000
```

## 📁 Project Structure

```
frontend/
├── app/              # Next.js app router
├── components/       # React components
│   ├── NewsAnalyzer.tsx  # Main analysis interface
│   ├── About.tsx         # About section
│   └── ui/              # Reusable UI components
├── public/           # Static assets
└── lib/             # Utilities and helpers
```

## 🌐 Environment Variables

For local development, create `.env.local`:

```bash
HF_SPACE_URL="https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space"
NEXT_PUBLIC_FORMSPREE_ID="your_contact_form_id"  # Optional for contact form
```

## 🚀 Deployment

The application is deployed on Vercel with automatic deployments from the main branch.

### Manual Deployment

```bash
npm run build    # Build for production
npm run start    # Test production build locally
```

## 📱 Browser Support

- Modern browsers with ES2020+ support
- Mobile Safari and Chrome
- Optimized for Arabic RTL text rendering

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

---

**حقيقة (Haqiqa)** - Bringing truth to Arabic news through advanced AI technology.
