# حقيقة (Haqiqa) - Palestine Fake News Detector Frontend

Modern Arabic RTL web application for detecting fake news in Palestinian content. Built with Next.js 15, TypeScript, and Tailwind CSS.

## 🌟 Features

- **🔍 Real-time Analysis**: Instant fake news detection for Arabic text
- **📱 Responsive Design**: Optimized for all devices and screen sizes
- **🎨 Arabic RTL Support**: Full right-to-left layout with Cairo font
- **✨ Modern UI/UX**: Clean, professional interface with smooth animations
- **📧 Contact Integration**: Working contact form with Formspree
- **🚀 Production Ready**: Optimized for Vercel deployment

## 🛠️ Tech Stack

- **Framework**: Next.js 15.4.6 with App Router
- **Language**: TypeScript 5.2.2
- **Styling**: Tailwind CSS 3.3.3 with RTL support
- **UI Components**: Custom shadcn/ui components
- **Icons**: Lucide React
- **Font**: Cairo (Google Fonts)
- **Deployment**: Vercel-ready

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- npm or yarn package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/WalidAlsafadi/Palestine-Fake-News-Detector
cd Palestine-Fake-News-Detector/webapp/frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

### Build for Production

```bash
# Create production build
npm run build

# Test production build locally
npm run start
```

## 📁 Project Structure

```
webapp/frontend/
├── app/                    # Next.js App Router
│   ├── globals.css        # Global styles with Cairo font
│   ├── layout.tsx         # Root layout with RTL support
│   └── page.tsx           # Main page component
├── components/            # React components
│   ├── ui/               # Reusable UI components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── input.tsx
│   │   └── textarea.tsx
│   ├── About.tsx         # About section
│   ├── Contact.tsx       # Contact form
│   ├── Developer.tsx     # Developer information
│   ├── Footer.tsx        # Site footer
│   ├── Header.tsx        # Navigation header
│   ├── Hero.tsx          # Hero section
│   └── NewsAnalyzer.tsx  # News analysis component
├── public/               # Static assets
│   ├── haqiqa-logo.svg  # Main logo
│   └── palestine-map.svg # Palestine map icon
├── lib/                  # Utility functions
└── types/               # TypeScript type definitions
```

## 🎨 Design System

### Colors

- **Primary**: #799EFF (Brand blue)
- **Background**: #F8FAFC (Light gray)
- **Text**: #1E293B (Dark slate)
- **Accent**: #10B981 (Green for success)
- **Error**: #EF4444 (Red for errors)

### Typography

- **Font Family**: Cairo (Arabic-optimized)
- **Headings**: Font weights 600-700
- **Body**: Font weight 400
- **RTL Support**: Full Arabic text direction

### Components

- **Cards**: Elevated design with subtle shadows
- **Buttons**: Primary and secondary variants
- **Forms**: Clean inputs with proper validation
- **Navigation**: Responsive header with mobile menu

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file for local development:

```bash
# Contact form endpoint (Formspree)
NEXT_PUBLIC_FORMSPREE_ID=your_formspree_id

# Analytics (optional)
NEXT_PUBLIC_GA_ID=your_google_analytics_id
```

### Tailwind CSS

The project uses a custom Tailwind configuration with:

- RTL support via `dir="rtl"` attribute
- Custom color palette
- Arabic typography optimizations
- Responsive breakpoints

## 🚀 Deployment

### Vercel (Recommended)

1. Connect your GitHub repository to Vercel
2. Configure environment variables
3. Deploy automatically on push to main branch

```bash
# Deploy to Vercel
npx vercel

# Production deployment
npx vercel --prod
```

### Manual Deployment

```bash
# Build the application
npm run build

# The output will be in the .next folder
# Deploy the contents to your hosting provider
```

## 🧪 Development

### Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run start    # Start production server
npm run lint     # Run ESLint
npm run type-check # Run TypeScript compiler
```

### Code Quality

- **ESLint**: Configured with Next.js recommended rules
- **TypeScript**: Strict mode enabled
- **Prettier**: Code formatting (optional)

## 🌐 Internationalization

The application is built with Arabic RTL support:

- Text direction: Right-to-left
- Layout mirroring: Automatic with Tailwind
- Font optimization: Cairo font for Arabic text
- Cultural considerations: Palestinian context

## 📱 Mobile Optimization

- **Responsive Design**: Mobile-first approach
- **Touch Interactions**: Optimized for touch devices
- **Performance**: Lightweight and fast loading
- **Accessibility**: ARIA labels and semantic HTML

## 🔒 Security

- **Form Validation**: Client and server-side validation
- **Content Security**: Protected against XSS
- **Environment Variables**: Secure configuration
- **HTTPS**: SSL/TLS encryption in production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines

- Follow TypeScript best practices
- Use semantic commit messages
- Test components thoroughly
- Maintain RTL compatibility
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## 🙏 Acknowledgments

- **AraBERT**: For Arabic language processing
- **Tailwind CSS**: For utility-first styling
- **Next.js**: For the React framework
- **Vercel**: For hosting and deployment
- **Cairo Font**: For Arabic typography

---

**حقيقة (Haqiqa)** - Bringing truth to Palestinian news through technology.
