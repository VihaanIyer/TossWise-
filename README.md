# Smart Trash Bin Detection System

An intelligent trash bin system that uses computer vision and AI to help users correctly dispose of waste items. The system detects items in your hand, classifies them into the appropriate bin (recycling, compost, or landfill), and provides voice feedback.

## Features

- **Object Detection**: Uses YOLOv8 to detect food items and objects in real-time
- **AI Classification**: Leverages Google's Gemini API to determine the correct waste bin
- **Voice Feedback**: Uses ElevenLabs for natural text-to-speech responses
- **Voice Input**: Accepts questions from users via microphone
- **Real-time Processing**: Processes camera feed in real-time

## Requirements

- Python 3.8+
- Webcam
- Microphone (for voice input)
- API Keys:
  - Google Gemini API key
  - ElevenLabs API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/VihaanIyer/Bin-.git
cd Bin-
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   - Edit `.env` and add your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```

## Getting API Keys

### Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### ElevenLabs API Key
1. Sign up at [ElevenLabs](https://elevenlabs.io/)
2. Go to your profile settings
3. Copy your API key to your `.env` file

## Usage

Make sure your virtual environment is activated, then run:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

### Controls

- **'d'** - Trigger detection of items in front of camera
- **'s'** - Listen for a question (speak when prompted)
- **'q'** - Quit the application

### How It Works

1. **Detection**: When you press 'd' or hold an item in front of the camera, the system detects what you're holding using YOLOv8
2. **Classification**: The detected item is sent to Gemini AI, which determines which bin it should go into
3. **Feedback**: The system speaks the result and explanation using ElevenLabs TTS
4. **Questions**: Press 's' to ask questions about recycling or waste disposal

## Project Structure

```
Bin-/
├── main.py                 # Main application entry point
├── object_detector.py      # YOLOv8 object detection
├── gemini_classifier.py    # Gemini API integration
├── tts_handler.py          # ElevenLabs TTS integration
├── voice_input.py          # Speech recognition for questions
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## Technical Details

- **Object Detection**: YOLOv8 (nano model for speed)
- **AI Model**: Google Gemini Pro
- **TTS**: ElevenLabs API
- **Speech Recognition**: Google Speech Recognition API
- **Computer Vision**: OpenCV

## Troubleshooting

### Camera Issues
- Make sure your webcam is connected and not being used by another application
- Try changing the camera index in `main.py` if you have multiple cameras

### Audio Issues
- Ensure your microphone permissions are granted
- Check that PyAudio is properly installed (may require system audio libraries)

### API Errors
- Verify your API keys are correct in the `.env` file
- Check your API quotas/limits
- Ensure you have internet connection

## Future Enhancements

- Hardware integration for actual trash bin control
- Multi-language support
- Improved object detection accuracy
- Custom voice training
- Mobile app integration

## License

This project is open source and available for modification and distribution.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

