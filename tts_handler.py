"""
ElevenLabs Text-to-Speech Handler with Local Fallback
Converts text responses to speech using ElevenLabs or local TTS as fallback
"""

import os
import subprocess
import platform
from dotenv import load_dotenv

# Load .env from the same directory as this file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
# Also try loading from current directory as fallback
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
        print(f"DEBUG: API key found: {'Yes' if api_key else 'No'}")
        if api_key:
            print(f"DEBUG: API key length: {len(api_key)}, starts with: {api_key[:10]}...")
        
        if api_key and api_key != 'your_elevenlabs_api_key_here':
            try:
                from elevenlabs.client import ElevenLabs
                from elevenlabs import play
                self.client = ElevenLabs(api_key=api_key)
                # play is a module, use play.play function
                self.play_func = play.play
                
                # Use voice_id from parameter, env var, or default to provided voice
                if not self.voice_id:
                    self.voice_id = os.getenv('ELEVENLABS_VOICE_ID', 'g6xIsTj2HwM6VR4iXFCw')
                
                # Test the voice to see if it works
                try:
                    # Try a quick test conversion with newer model (free tier compatible)
                    test_audio = self.client.text_to_speech.convert(
                        voice_id=self.voice_id,
                        text="test",
                        model_id="eleven_turbo_v2_5"  # Updated model for free tier
                    )
                    # Consume the generator to test
                    list(test_audio)  # This will raise an error if voice is invalid
                    self.use_elevenlabs = True
                    print(f"✅ ElevenLabs TTS initialized with voice ID: {self.voice_id}")
                except Exception as voice_error:
                    error_str = str(voice_error)
                    if 'voice_limit_reached' in error_str or 'voice_not_found' in error_str.lower() or '400' in error_str:
                        print(f"⚠️  Voice ID {self.voice_id} not accessible. Trying standard voices...")
                        # Try using standard/premade voices instead
                        standard_voices = [
                            '21m00Tcm4TlvDq8ikWAM',  # Rachel - friendly female
                            'pNInz6obpgDQGcFmaJgB',  # Adam - deep male
                            'EXAVITQu4vr4xnSDxMaL',  # Bella - soft female
                        ]
                        for std_voice in standard_voices:
                            try:
                                test_audio = self.client.text_to_speech.convert(
                                    voice_id=std_voice,
                                    text="test",
                                    model_id="eleven_turbo_v2_5"  # Updated model for free tier
                                )
                                list(test_audio)
                                self.voice_id = std_voice
                                self.use_elevenlabs = True
                                print(f"✅ ElevenLabs TTS initialized with standard voice: {std_voice} (Rachel)")
                                break
                            except:
                                continue
                        if not self.use_elevenlabs:
                            print(f"❌ Could not find working voice. Using local TTS.")
                            self.use_elevenlabs = False
                    else:
                        raise voice_error
                        
            except ImportError as e:
                print(f"❌ ElevenLabs package not installed: {e}")
                print("   Install with: pip install elevenlabs")
                self.use_elevenlabs = False
            except Exception as e:
                print(f"❌ ElevenLabs initialization failed, using local TTS: {e}")
                import traceback
                traceback.print_exc()
                self.use_elevenlabs = False
        else:
            print(f"⚠️  ElevenLabs API key not found or invalid. Using local TTS fallback.")
        
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
                print(f"🔊 Using ElevenLabs to speak: {text[:50]}...")
                # Generate audio using ElevenLabs with free tier compatible model
                audio = self.client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_turbo_v2_5"  # Updated model for free tier
                )
                
                # Convert generator to bytes
                audio_bytes = b"".join(audio)
                
                if save_to_file:
                    with open(save_to_file, 'wb') as f:
                        f.write(audio_bytes)
                
                # Play the audio
                self.play_func(audio_bytes)
                print(f"✅ ElevenLabs audio played successfully")
                return
            except Exception as e:
                print(f"❌ ElevenLabs TTS error, falling back to local: {e}")
                import traceback
                traceback.print_exc()
        
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
                # Generate audio using the client with free tier compatible model
                audio = self.client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_turbo_v2_5"  # Updated model for free tier
                )
                # Convert generator to bytes
                return b"".join(audio)
            except Exception as e:
                print(f"Error generating audio: {e}")
                return None
        else:
            # Local TTS doesn't return bytes easily
            return None

