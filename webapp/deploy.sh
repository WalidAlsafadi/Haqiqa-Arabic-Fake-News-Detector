#!/bin/bash

# AraBERT Fake News Detector - Deployment Script
# This script helps you deploy both backend and frontend

echo "🚀 AraBERT Fake News Detector Deployment Script"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -d "webapp" ]; then
    print_error "Please run this script from the Palestine-Fake-News-Detector root directory"
    exit 1
fi

echo ""
echo "This script will help you deploy your AraBERT Fake News Detector webapp."
echo ""

# Backend deployment instructions
echo "📱 BACKEND DEPLOYMENT (Hugging Face Spaces)"
echo "============================================="
print_info "1. Go to https://huggingface.co/spaces"
print_info "2. Click 'Create new Space'"
print_info "3. Choose SDK: Gradio"
print_info "4. Name: arabert-fake-news-detector"
print_info "5. Upload files from webapp/backend/ directory:"
echo "   - app.py"
echo "   - requirements.txt" 
echo "   - README.md"
print_info "6. Your space will be available at:"
echo "   https://YOUR_USERNAME-arabert-fake-news-detector.hf.space/"
echo ""

# Frontend setup
echo "💻 FRONTEND SETUP"
echo "=================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ first."
    echo "Download from: https://nodejs.org/"
    exit 1
fi

print_status "Node.js is installed: $(node --version)"

# Navigate to frontend directory
cd webapp/frontend

# Check if package.json exists
if [ ! -f "package.json" ]; then
    print_error "package.json not found in webapp/frontend directory"
    exit 1
fi

# Install dependencies
print_info "Installing frontend dependencies..."
if npm install; then
    print_status "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

echo ""
echo "🔧 CONFIGURATION"
echo "================"
print_warning "Before running, you need to update the backend URL in app/page.tsx"
print_info "Replace 'walidalsafadi' with your Hugging Face username on line ~35:"
print_info 'const app = await client("YOUR_USERNAME/arabert-fake-news-detector")'
echo ""

# Ask if user wants to start development server
read -p "Do you want to start the development server now? (y/n): " start_dev

if [ "$start_dev" = "y" ] || [ "$start_dev" = "Y" ]; then
    print_info "Starting development server..."
    print_status "Frontend will be available at: http://localhost:3000"
    npm run dev
else
    echo ""
    echo "🎯 NEXT STEPS"
    echo "============="
    print_info "1. Update the backend URL in app/page.tsx"
    print_info "2. Start development server: npm run dev"
    print_info "3. Deploy backend to Hugging Face Spaces"
    print_info "4. Deploy frontend to Vercel (push to GitHub first)"
    echo ""
    print_status "Setup complete! Your webapp is ready for deployment."
fi

echo ""
echo "📚 USEFUL COMMANDS"
echo "=================="
echo "Frontend development: cd webapp/frontend && npm run dev"
echo "Frontend build:       cd webapp/frontend && npm run build"
echo "Frontend production:  cd webapp/frontend && npm start"
echo ""

echo "🔗 DEPLOYMENT PLATFORMS"
echo "======================="
echo "Backend:  https://huggingface.co/spaces"
echo "Frontend: https://vercel.com"
echo "GitHub:   https://github.com"
echo ""

print_status "Deployment script completed!"
echo "For detailed instructions, see webapp/README.md"
