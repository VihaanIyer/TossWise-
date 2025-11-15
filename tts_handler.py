"""
ElevenLabs Text-to-Speech Handler with Local Fallback
Converts text responses to speech using ElevenLabs or local TTS as fallback
"""

import os
import subprocess
import platform
from dotenv import load_dotenv

load_dotenv()


class TTSHandler:
    def __init__(self, voice_id=None):
        """
        Initialize TTS Handler with ElevenLabs (optional) and local fallback
        
        Args:
            voice_id: Optional specific voice ID to use for ElevenLabs
        """
        self.use_elevenlabs = False
        self.client = None
        self.voice_id = voice_id
        
        # Try to initialize ElevenLabs if API key is available
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if api_key and api_key != 'your_elevenlabs_api_key_here':
            try:
                from elevenlabs.client import ElevenLabs
                from elevenlabs import play
                self.client = ElevenLabs(api_key=api_key)
                self.play_func = play
                self.use_elevenlabs = True
                
                # If no voice_id provided, use default voice (Rachel - friendly female voice)
                if not self.voice_id:
                    self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
                print("ElevenLabs TTS initialized")
            except Exception as e:
                print(f"ElevenLabs initialization failed, using local TTS: {e}")
                self.use_elevenlabs = False
        
        # Determine local TTS method based on OS
        self.system = platform.system()
        if not self.use_elevenlabs:
            if self.system == "Darwin":  # macOS
                print("Using macOS 'say' command for TTS")
            elif self.system == "Linux":
                print("Using 'espeak' for TTS (install with: sudo apt-get install espeak)")
            else:
                print("Using pyttsx3 for TTS")
                try:
                    import pyttsx3
                    self.local_tts = pyttsx3.init()
                except ImportError:
                    print("pyttsx3 not available, TTS may not work")
                    self.local_tts = None
    
    def speak(self, text, save_to_file=None):
        """
        Convert text to speech and play it
        
        Args:
            text: Text to convert to speech
            save_to_file: Optional file path to save audio
        """
        if self.use_elevenlabs:
            try:
                # Generate audio using ElevenLabs
                audio = self.client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_monolingual_v1"
                )
                
                # Convert generator to bytes
                audio_bytes = b"".join(audio)
                
                if save_to_file:
                    with open(save_to_file, 'wb') as f:
                        f.write(audio_bytes)
                
                # Play the audio
                self.play_func(audio_bytes)
                return
            except Exception as e:
                print(f"ElevenLabs TTS error, falling back to local: {e}")
        
        # Fallback to local TTS
        try:
            if self.system == "Darwin":  # macOS
                # Use macOS 'say' command
                subprocess.run(['say', text], check=False)
            elif self.system == "Linux":
                # Use espeak
                subprocess.run(['espeak', text], check=False)
            else:
                # Use pyttsx3
                if hasattr(self, 'local_tts') and self.local_tts:
                    self.local_tts.say(text)
                    self.local_tts.runAndWait()
                else:
                    print(f"TTS: {text}")  # Fallback to print
        except Exception as e:
            print(f"Local TTS error: {e}")
            print(f"TTS: {text}")  # Final fallback to print
    
    def get_audio_bytes(self, text):
        """
        Get audio bytes without playing (for async operations)
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio bytes or None
        """
        if self.use_elevenlabs:
            try:
                # Generate audio using the client
                audio = self.client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_monolingual_v1"
                )
                # Convert generator to bytes
                return b"".join(audio)
            except Exception as e:
                print(f"Error generating audio: {e}")
                return None
        else:
            # Local TTS doesn't return bytes easily
            return None

