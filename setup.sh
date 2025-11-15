#!/bin/bash

echo "Setting up Smart Trash Bin Detection System..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for .env file
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "IMPORTANT: Please edit .env file and add your API keys:"
    echo "  - GEMINI_API_KEY"
    echo "  - ELEVENLABS_API_KEY"
    echo ""
else
    echo ".env file already exists."
fi

echo ""
echo "Setup complete!"
echo ""
echo "To run the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Make sure .env file has your API keys"
echo "  3. Run: python main.py"
echo ""

