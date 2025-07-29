#!/bin/bash
# Install voice processing dependencies for the backend

echo "🎤 Installing voice assistant dependencies..."

# Navigate to backend directory
cd backend

echo "📦 Installing Python dependencies..."
pip install SpeechRecognition==3.10.0
pip install pyaudio==0.2.11
pip install pydub==0.25.1
pip install librosa==0.10.1

echo "🎵 Installing system audio dependencies..."

# Check if on Ubuntu/Debian
if command -v apt-get &> /dev/null; then
    echo "📦 Installing system audio packages for Ubuntu/Debian..."
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-pyaudio flac
    
# Check if on macOS
elif command -v brew &> /dev/null; then
    echo "📦 Installing system audio packages for macOS..."
    brew install portaudio flac
    
# Check if on CentOS/RHEL/Fedora
elif command -v yum &> /dev/null; then
    echo "📦 Installing system audio packages for CentOS/RHEL..."
    sudo yum install -y portaudio-devel flac
    
elif command -v dnf &> /dev/null; then
    echo "📦 Installing system audio packages for Fedora..."
    sudo dnf install -y portaudio-devel flac
    
else
    echo "⚠️ Unknown system. Please install portaudio and flac manually."
    echo "For Ubuntu/Debian: sudo apt-get install portaudio19-dev python3-pyaudio flac"
    echo "For macOS: brew install portaudio flac"
    echo "For CentOS/RHEL: sudo yum install portaudio-devel flac"
fi

echo "✅ Voice assistant dependencies installation complete!"
echo ""
echo "🚀 You can now start the backend server with voice support:"
echo "cd backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8004"
echo ""
echo "🎤 Voice features available:"
echo "- Speech-to-text conversion"
echo "- Voice chat with AI responses"
echo "- Audio file storage (optional)"
echo "- Multiple recognition engines (Google + Sphinx fallback)"
