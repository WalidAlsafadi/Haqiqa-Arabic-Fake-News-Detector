# AraBERT Fake News Detector - Windows Deployment Script
# This script helps you deploy both backend and frontend on Windows

Write-Host "🚀 AraBERT Fake News Detector Deployment Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Function to print colored output
function Write-Success {
    param($Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param($Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error {
    param($Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Info {
    param($Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

# Check if we're in the right directory
if (-not (Test-Path "webapp")) {
    Write-Error "Please run this script from the Palestine-Fake-News-Detector root directory"
    exit 1
}

Write-Host ""
Write-Host "This script will help you deploy your AraBERT Fake News Detector webapp." -ForegroundColor White
Write-Host ""

# Backend deployment instructions
Write-Host "📱 BACKEND DEPLOYMENT (Hugging Face Spaces)" -ForegroundColor Magenta
Write-Host "=============================================" -ForegroundColor Magenta
Write-Info "1. Go to https://huggingface.co/spaces"
Write-Info "2. Click 'Create new Space'"
Write-Info "3. Choose SDK: Gradio"
Write-Info "4. Name: arabert-fake-news-detector"
Write-Info "5. Upload files from webapp/backend/ directory:"
Write-Host "   - app.py" -ForegroundColor Gray
Write-Host "   - requirements.txt" -ForegroundColor Gray
Write-Host "   - README.md" -ForegroundColor Gray
Write-Info "6. Your space will be available at:"
Write-Host "   https://YOUR_USERNAME-arabert-fake-news-detector.hf.space/" -ForegroundColor Gray
Write-Host ""

# Frontend setup
Write-Host "💻 FRONTEND SETUP" -ForegroundColor Magenta
Write-Host "=================" -ForegroundColor Magenta

# Check if Node.js is installed
try {
    $nodeVersion = node --version 2>$null
    if ($nodeVersion) {
        Write-Success "Node.js is installed: $nodeVersion"
    } else {
        throw "Node.js not found"
    }
} catch {
    Write-Error "Node.js is not installed. Please install Node.js 18+ first."
    Write-Host "Download from: https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Navigate to frontend directory
Set-Location "webapp\frontend"

# Check if package.json exists
if (-not (Test-Path "package.json")) {
    Write-Error "package.json not found in webapp\frontend directory"
    exit 1
}

# Install dependencies
Write-Info "Installing frontend dependencies..."
try {
    npm install
    Write-Success "Dependencies installed successfully"
} catch {
    Write-Error "Failed to install dependencies"
    exit 1
}

Write-Host ""
Write-Host "🔧 CONFIGURATION" -ForegroundColor Magenta
Write-Host "================" -ForegroundColor Magenta
Write-Warning "Before running, you need to update the backend URL in app\page.tsx"
Write-Info "Replace 'walidalsafadi' with your Hugging Face username on line ~35:"
Write-Info 'const app = await client("YOUR_USERNAME/arabert-fake-news-detector")'
Write-Host ""

# Ask if user wants to start development server
$startDev = Read-Host "Do you want to start the development server now? (y/n)"

if ($startDev -eq "y" -or $startDev -eq "Y") {
    Write-Info "Starting development server..."
    Write-Success "Frontend will be available at: http://localhost:3000"
    npm run dev
} else {
    Write-Host ""
    Write-Host "🎯 NEXT STEPS" -ForegroundColor Magenta
    Write-Host "=============" -ForegroundColor Magenta
    Write-Info "1. Update the backend URL in app\page.tsx"
    Write-Info "2. Start development server: npm run dev"
    Write-Info "3. Deploy backend to Hugging Face Spaces"
    Write-Info "4. Deploy frontend to Vercel (push to GitHub first)"
    Write-Host ""
    Write-Success "Setup complete! Your webapp is ready for deployment."
}

Write-Host ""
Write-Host "📚 USEFUL COMMANDS" -ForegroundColor Magenta
Write-Host "==================" -ForegroundColor Magenta
Write-Host "Frontend development: cd webapp\frontend && npm run dev" -ForegroundColor Gray
Write-Host "Frontend build:       cd webapp\frontend && npm run build" -ForegroundColor Gray
Write-Host "Frontend production:  cd webapp\frontend && npm start" -ForegroundColor Gray
Write-Host ""

Write-Host "🔗 DEPLOYMENT PLATFORMS" -ForegroundColor Magenta
Write-Host "=======================" -ForegroundColor Magenta
Write-Host "Backend:  https://huggingface.co/spaces" -ForegroundColor Gray
Write-Host "Frontend: https://vercel.com" -ForegroundColor Gray
Write-Host "GitHub:   https://github.com" -ForegroundColor Gray
Write-Host ""

Write-Success "Deployment script completed!"
Write-Host "For detailed instructions, see webapp\README.md" -ForegroundColor Yellow
