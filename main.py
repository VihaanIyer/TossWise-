"""
Smart Trash Bin Detection System
Main application that integrates object detection, Gemini AI, and ElevenLabs TTS
"""

import cv2
import time
import threading
import os
import json
from pathlib import Path
from datetime import datetime
from object_detector import ObjectDetector
from gemini_classifier import TrashClassifier
from tts_handler import TTSHandler
from voice_input import VoiceInputHandler
from bin_layout_analyzer import BinLayoutAnalyzer

# Optional Arduino servo controller import
try:
    from Arduino_Scripts.trash_bin_servo import MultiArduinoServoController
except (ImportError, ModuleNotFoundError) as e:
    MultiArduinoServoController = None
    # Log that Arduino support is disabled (but don't fail startup)
    pass

# Modify this when you want to switch cameras (Continuity Cam often shows as index 1 or 2)
# Try 0 for built-in webcam, 1 or 2 for Continuity Camera
CAMERA_INDEX = 0


def get_language_from_location(location):
    """
    Determine language based on location.
    Returns 'hungarian' for Budapest, 'english' for others.
    """
    if location and 'budapest' in location.lower():
        return 'hungarian'
    return 'english'


def get_closing_message(language):
    """
    Get closing message in the appropriate language.
    """
    if language == 'hungarian':
        return "Ha van kérdésed, szólj."
    return "If you have any questions, let me know."


def format_item_response(item, bin_name, bin_color, bin_position, language):
    """
    Format item classification response in the appropriate language.
    Handles alternative bins and no-bin-available cases.
    Uses position (left/middle/right) instead of "usually [color]".
    """
    if language == 'hungarian':
        # Translate colors to Hungarian
        color_translations = {
            'blue': 'kék',
            'green': 'zöld',
            'black': 'fekete',
            'grey': 'szürke',
            'gray': 'szürke',
            'gray and black': 'szürke és fekete',
            'black or grey': 'fekete vagy szürke',
            'black or gray': 'fekete vagy szürke'
        }
        # bin_name already contains "kuka" (e.g., "újrahasznosítás kuka")
        # We need to change "kuka" to "kukába" (into the bin)
        if ' kuka' in bin_name:
            bin_name_final = bin_name.replace(' kuka', ' kukába')
        elif bin_name.endswith('kuka'):
            bin_name_final = bin_name.replace('kuka', 'kukába')
        else:
            bin_name_final = f"{bin_name} kukába"
        
        # Handle no bin available case
        if item.get('no_bin_available'):
            return item.get('explanation', f"{item['item']} nem mehet egyik kukába sem.")
        
        # Handle alternative bin case
        if item.get('alternative'):
            return item.get('explanation', f"{item['item']} megy a {bin_name} kukába.")
        
        # Translate color
        color_hungarian = color_translations.get(bin_color.lower(), bin_color)
        
        # Use position if available, otherwise fallback to color
        if bin_position:
            return f"{item['item']} megy a {color_hungarian} {bin_name_final} {bin_position}"
        else:
            return f"{item['item']} megy a {bin_name_final} általában {color_hungarian}"
    else:  # English
        # Handle no bin available case
        if item.get('no_bin_available'):
            return item.get('explanation', f"{item['item']} can't go into any available bin.")
        
        # Handle alternative bin case
        if item.get('alternative'):
            return item.get('explanation', f"{item['item']} goes into {bin_name}.")
        
        # Use position if available, otherwise fallback to color
        if bin_position:
            return f"{item['item']} goes into {bin_color} {bin_name} {bin_position}"
        else:
            return f"{item['item']} goes into {bin_name} usually {bin_color}"


class Logger:
    """Professional logging system for the smart trash bin"""
    
    @staticmethod
    def timestamp():
        """Get formatted timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def log_detection(items):
        """Log detection event"""
        print(f"\n{'='*80}")
        print(f"🔍 DETECTION EVENT - {Logger.timestamp()}")
        print(f"{'='*80}")
        print(f"Items Detected: {len(items)}")
        for i, item in enumerate(items, 1):
            print(f"  [{i}] {item['class'].upper():<20} Confidence: {item['confidence']:.2%}")
        print(f"{'='*80}\n")
    
    @staticmethod
    def log_llm_request(prompt_preview, image_sent=True):
        """Log LLM API request"""
        print(f"\n{'─'*80}")
        print(f"📤 LLM REQUEST - {Logger.timestamp()}")
        print(f"{'─'*80}")
        print(f"Image Sent: {'✅ Yes' if image_sent else '❌ No'}")
        print(f"Prompt Preview:")
        print(f"  {prompt_preview[:200]}..." if len(prompt_preview) > 200 else f"  {prompt_preview}")
        print(f"{'─'*80}\n")
    
    @staticmethod
    def log_llm_response(response_text, items_classified):
        """Log LLM API response"""
        print(f"\n{'─'*80}")
        print(f"📥 LLM RESPONSE - {Logger.timestamp()}")
        print(f"{'─'*80}")
        print(f"Raw Response:")
        print(f"  {response_text}")
        print(f"\nParsed Classifications: {len(items_classified)} items")
        for i, item in enumerate(items_classified, 1):
            print(f"  [{i}] {item['item'].upper()}")
            print(f"      └─ Bin: {item['bin_type'].upper()}")
            print(f"      └─ Explanation: {item['explanation'][:100]}...")
        print(f"{'─'*80}\n")
    
    @staticmethod
    def log_classification_summary(classifications):
        """Log final classification summary"""
        print(f"\n{'='*80}")
        print(f"✅ CLASSIFICATION COMPLETE - {Logger.timestamp()}")
        print(f"{'='*80}")
        print(f"Total Items Classified: {len(classifications)}\n")
        
        for i, item in enumerate(classifications, 1):
            print(f"  ITEM #{i}: {item['item'].upper()}")
            print(f"  {'─'*76}")
            print(f"  Bin Assignment: {item['bin_type'].upper()}")
            print(f"  Full Explanation:")
            print(f"    {item['explanation']}")
            print()
        
        print(f"{'='*80}\n")
    
    @staticmethod
    def log_tts_output(text):
        """Log TTS output"""
        print(f"\n🔊 TTS OUTPUT - {Logger.timestamp()}")
        print(f"  Speaking: \"{text}\"")
        print()
    
    @staticmethod
    def log_question(question):
        """Log user question"""
        print(f"\n{'─'*80}")
        print(f"❓ USER QUESTION - {Logger.timestamp()}")
        print(f"{'─'*80}")
        print(f"Question: {question}")
        print(f"{'─'*80}\n")
    
    @staticmethod
    def log_answer(answer):
        """Log LLM answer to question"""
        print(f"\n{'─'*80}")
        print(f"💬 LLM ANSWER - {Logger.timestamp()}")
        print(f"{'─'*80}")
        print(f"Answer: {answer}")
        print(f"{'─'*80}\n")
    
    @staticmethod
    def log_error(error_msg, context=""):
        """Log error"""
        print(f"\n{'!'*80}")
        print(f"❌ ERROR - {Logger.timestamp()}")
        print(f"{'!'*80}")
        if context:
            print(f"Context: {context}")
        print(f"Error: {error_msg}")
        print(f"{'!'*80}\n")
    
    @staticmethod
    def log_system_event(event_msg):
        """Log system events"""
        print(f"\n⚙️  SYSTEM EVENT - {Logger.timestamp()}")
        print(f"  {event_msg}\n")
    
    @staticmethod
    def log_realtime_detection(items):
        """Log real-time detection (single line)"""
        if items:
            item_names = [f"{item['class']} ({item['confidence']:.2%})" for item in items]
            print(f"\r👁️  [{Logger.timestamp().split()[1]}] Seeing: {', '.join(item_names)}", end='', flush=True)
        else:
            print(f"\r👁️  [{Logger.timestamp().split()[1]}] Seeing: nothing", end='', flush=True)


class SmartTrashBin:
    def __init__(self):
        """
        Initialize the smart trash bin system
        """
        Logger.log_system_event("Initializing Smart Trash Bin System...")

        # Toggle this if you want deep LLM debug logs (request/response/summary)
        self.llm_debug_logs = False
        
        # Initialize components
        try:
            Logger.log_system_event("Loading YOLOv8 object detector...")
            self.detector = ObjectDetector()
            Logger.log_system_event("YOLOv8 loaded successfully")
        except Exception as e:
            Logger.log_error(str(e), "ObjectDetector initialization")
            raise
        
        # Load bin layout from location-specific JSON file or main metadata file
        self.bin_layout_metadata = None
        self.location = os.getenv('BIN_LOCATION', None)  # Can be set via environment variable
        
        # Determine language based on location
        self.language = get_language_from_location(self.location)
        Logger.log_system_event(f"Language set to: {self.language.upper()} (based on location: {self.location or 'default'})")
        
        try:
            Logger.log_system_event("Initializing Gemini AI classifier...")
            self.classifier = TrashClassifier(language=self.language)
            Logger.log_system_event("Gemini AI initialized successfully")
        except Exception as e:
            Logger.log_error(str(e), "TrashClassifier initialization")
            raise
        
        # Try to load location-specific bin layout first
        if self.location:
            location_file = f"bin_layout_{self.location}.json"
            if os.path.exists(location_file):
                try:
                    with open(location_file, 'r') as f:
                        self.bin_layout_metadata = json.load(f)
                        # Update classifier with bin layout
                        self.classifier.bin_layout = self.classifier._load_bin_layout()
                        self.classifier.update_bin_context(self.bin_layout_metadata)
                        identified_bins = len(self.bin_layout_metadata.get("bins", [])) if isinstance(self.bin_layout_metadata, dict) else len(self.bin_layout_metadata or [])
                        Logger.log_system_event(f"Loaded {identified_bins} bins from location-specific file: {location_file}")
                        
                        # Log detailed bin information on initial load
                        Logger.log_system_event("📋 BIN CONFIGURATION:")
                        for i, bin_info in enumerate(self.bin_layout_metadata.get("bins", []), 1):
                            bin_type = bin_info.get('type', 'unknown')
                            color = bin_info.get('color', 'N/A')
                            sign = bin_info.get('sign', 'N/A')
                            label = bin_info.get('label', 'N/A')
                            Logger.log_system_event(f"  Bin {i}: {bin_type.upper()} | Color: {color} | Sign: {sign} | Label: {label}")
                        
                        Logger.log_system_event("📋 Available bin types: " + ", ".join([
                            bin_info.get('type', 'unknown') 
                            for bin_info in self.bin_layout_metadata.get("bins", [])
                        ]))
                except Exception as e:
                    Logger.log_error(str(e), f"Loading location file {location_file}")
        
        # Fallback to main bin_layout_metadata.json
        if self.bin_layout_metadata is None:
            cached_layout = BinLayoutAnalyzer.load_cached_bins()
            if cached_layout:
                self.bin_layout_metadata = cached_layout
                # Update classifier with bin layout
                self.classifier.bin_layout = self.classifier._load_bin_layout()
                self.classifier.update_bin_context(cached_layout)
                cached_bins = len(cached_layout.get("bins", [])) if isinstance(cached_layout, dict) else len(cached_layout or [])
                Logger.log_system_event(f"Loaded {cached_bins} bins from bin_layout_metadata.json.")
                
                # Log detailed bin information on initial load
                Logger.log_system_event("📋 BIN CONFIGURATION:")
                for i, bin_info in enumerate(cached_layout.get("bins", []), 1):
                    bin_type = bin_info.get('type', 'unknown')
                    color = bin_info.get('color', 'N/A')
                    sign = bin_info.get('sign', 'N/A')
                    label = bin_info.get('label', 'N/A')
                    Logger.log_system_event(f"  Bin {i}: {bin_type.upper()} | Color: {color} | Sign: {sign} | Label: {label}")
                
                Logger.log_system_event("📋 Available bin types: " + ", ".join([
                    bin_info.get('type', 'unknown') 
                    for bin_info in cached_layout.get("bins", [])
                ]))
            else:
                Logger.log_system_event("No bin layout found. Please configure bins using the web app first.")
                Logger.log_system_event("Run: python web_app.py and configure your bin layout")
        
        try:
            Logger.log_system_event("Initializing TTS handler...")
            self.tts = TTSHandler(language=self.language)
            Logger.log_system_event("TTS handler initialized")
        except Exception as e:
            Logger.log_error(str(e), "TTSHandler initialization")
            raise
        
        # Initialize voice input for questions
        try:
            Logger.log_system_event("Initializing voice input handler...")
            self.voice_input = VoiceInputHandler()
            Logger.log_system_event("Voice input handler initialized")
        except Exception as e:
            Logger.log_error(str(e), "VoiceInputHandler initialization")
            self.voice_input = None
            Logger.log_system_event("Voice input disabled - continuing without question support")
        
        # Initialize Arduino servo controller
        self.servo_controller = None
        if MultiArduinoServoController is not None:
            try:
                Logger.log_system_event("Initializing Arduino servo controller...")
                arduino_configs = {
                    'arduino_1': {'port': os.getenv('ARDUINO_1_PORT', '/dev/tty.usbmodem12101')},
                    'arduino_2': {'port': os.getenv('ARDUINO_2_PORT', '/dev/tty.usbmodem12301')}
                }
                self.servo_controller = MultiArduinoServoController(arduino_configs)
                connection_status = self.servo_controller.connect_all()
                connected_count = sum(1 for status in connection_status.values() if status)
                if connected_count > 0:
                    Logger.log_system_event(f"Arduino servo controller initialized - {connected_count}/{len(arduino_configs)} Arduinos connected")
                else:
                    Logger.log_system_event("Arduino servo controller initialized but no Arduinos connected - bin opening disabled")
            except Exception as e:
                Logger.log_error(str(e), "ArduinoServoController initialization")
                self.servo_controller = None
                Logger.log_system_event("Arduino servo controller disabled - continuing without bin opening")
        else:
            Logger.log_system_event("Arduino servo controller not available (pyserial not installed) - continuing without bin opening")
        
        # State management
        self.last_detection_time = 0
        self.detection_cooldown = 0.5  # Short cooldown after classification to allow new detections
        self.last_trash_check_time = 0
        self.trash_check_interval = 0.1  # Very frequent checks for instant detection
        self.checking_trash = False  # Flag to prevent multiple simultaneous trash checks
        self.current_item = None
        self.current_classifications = []  # Store all classifications
        self.processing_question = False
        self.person_detected_time = None  # Track when person was detected for timing
        self.last_spoken_text = []  # Store last spoken items for "repeat" functionality
        self.listening_for_questions = False  # Track if we're listening for questions
        self.processing_detection = False  # Track if we're currently processing a detection
        self.detected_bags = []  # Store detected bags
        self.current_bag_index = 0  # Track which bag we're asking about

        # Proper classification cooldown tracking (shared with run loop)
        self.last_classification_time = 0.0
        
        # File watcher for bin layout reload
        self.last_reload_time = 0
        self.reload_signal_path = Path("reload_signal.txt")
        
        # Insights tracking
        self.insights_data = {
            'items': [],  # List of all classified items
            'bin_counts': {},  # Count of items per bin type
            'contamination': {},  # Items that went to wrong bins
            'start_time': time.time()
        }
        
        # Initialize bin counts and contamination tracking from bin layout
        if self.bin_layout_metadata:
            for bin_info in self.bin_layout_metadata.get('bins', []):
                bin_type = bin_info.get('type', '').lower()
                self.insights_data['bin_counts'][bin_type] = 0
                self.insights_data['contamination'][bin_type] = {
                    'wrong_items': [],
                    'total_items': 0
                }
        
        # Initialize insights file
        self.save_insights_data()
        
        Logger.log_system_event("System fully initialized and ready!")
        Logger.log_system_event("Bin layout reload monitoring active - will auto-reload when web app updates bins")
        print("\n" + "="*80)
        print("SYSTEM READY - Automatic trash detection active")
        print("="*80)
        print("Controls: 'q' to quit")
        print("="*80 + "\n")
    
    def _find_closest_bin(self, bin_type: str):
        """
        Find the closest available bin type to the requested bin type.
        Returns bin info dict or None.
        """
        if not self.bin_layout_metadata or not self.bin_layout_metadata.get("bins"):
            return None
        
        # Simple mapping: recycling -> compost -> landfill (fallback order)
        fallback_map = {
            "recycling": ["compost", "landfill"],
            "compost": ["recycling", "landfill"],
            "landfill": ["recycling", "compost"],
        }
        
        available_types = [bin_info.get("type", "").lower() for bin_info in self.bin_layout_metadata.get("bins", [])]
        
        # Check fallback options
        for fallback_type in fallback_map.get(bin_type.lower(), []):
            if fallback_type in available_types:
                # Find the bin info
                for bin_info in self.bin_layout_metadata.get("bins", []):
                    if bin_info.get("type", "").lower() == fallback_type:
                        return bin_info
        
        return None
    
    def save_insights_data(self):
        """Save insights data to JSON file for web app to read"""
        insights_file = Path("insights_data.json")
        try:
            insights_file.write_text(json.dumps(self.insights_data, indent=2))
        except Exception as e:
            Logger.log_error(f"Failed to save insights: {e}", "save_insights_data")
    
    def reload_bin_layout(self):
        """
        Reload bin layout from file without restarting the system
        Called automatically when web app updates bin configuration
        """
        Logger.log_system_event("🔄 Reloading bin layout from file...")
        
        try:
            # Reload location-specific or main metadata file
            if self.location:
                location_file = f"bin_layout_{self.location}.json"
                if os.path.exists(location_file):
                    with open(location_file, 'r') as f:
                        self.bin_layout_metadata = json.load(f)
                        # Update classifier with new bin layout
                        self.classifier.bin_layout = self.classifier._load_bin_layout()
                        self.classifier.system_prompt = self.classifier._build_system_prompt()
                        self.classifier.update_bin_context(self.bin_layout_metadata)
                        
                        bin_count = len(self.bin_layout_metadata.get("bins", []))
                        Logger.log_system_event(f"✅ Reloaded {bin_count} bins from {location_file}")
                        
                        # Log detailed bin information
                        Logger.log_system_event("📋 BIN CONFIGURATION:")
                        for i, bin_info in enumerate(self.bin_layout_metadata.get("bins", []), 1):
                            bin_type = bin_info.get('type', 'unknown')
                            color = bin_info.get('color', 'N/A')
                            sign = bin_info.get('sign', 'N/A')
                            label = bin_info.get('label', 'N/A')
                            Logger.log_system_event(f"  Bin {i}: {bin_type.upper()} | Color: {color} | Sign: {sign} | Label: {label}")
                        
                        Logger.log_system_event("📋 Available bin types: " + ", ".join([
                            bin_info.get('type', 'unknown') 
                            for bin_info in self.bin_layout_metadata.get("bins", [])
                        ]))
                        return True
            
            # Fallback to main metadata
            cached_layout = BinLayoutAnalyzer.load_cached_bins()
            if cached_layout:
                self.bin_layout_metadata = cached_layout
                self.classifier.bin_layout = self.classifier._load_bin_layout()
                self.classifier.system_prompt = self.classifier._build_system_prompt()
                self.classifier.update_bin_context(cached_layout)
                
                bin_count = len(cached_layout.get("bins", []))
                Logger.log_system_event(f"✅ Reloaded {bin_count} bins from bin_layout_metadata.json")
                
                # Log detailed bin information
                Logger.log_system_event("📋 BIN CONFIGURATION:")
                for i, bin_info in enumerate(cached_layout.get("bins", []), 1):
                    bin_type = bin_info.get('type', 'unknown')
                    color = bin_info.get('color', 'N/A')
                    sign = bin_info.get('sign', 'N/A')
                    label = bin_info.get('label', 'N/A')
                    Logger.log_system_event(f"  Bin {i}: {bin_type.upper()} | Color: {color} | Sign: {sign} | Label: {label}")
                
                Logger.log_system_event("📋 Available bin types: " + ", ".join([
                    bin_info.get('type', 'unknown') 
                    for bin_info in cached_layout.get("bins", [])
                ]))
                return True
            
            Logger.log_system_event("⚠️ No bin layout file found to reload")
            return False
            
        except Exception as e:
            Logger.log_error(str(e), "Reloading bin layout")
            return False
    
    def select_best_frame(self, frames):
        """
        Select the best frame from multiple captures based on image quality
        Uses Laplacian variance to measure focus/sharpness
        """
        if not frames or len(frames) == 0:
            return None
        
        if len(frames) == 1:
            return frames[0]
        
        best_frame = None
        best_score = 0
        
        for frame in frames:
            # Convert to grayscale for focus calculation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Calculate Laplacian variance (higher = sharper/more focused)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var > best_score:
                best_score = laplacian_var
                best_frame = frame
        
        Logger.log_system_event(f"Selected best frame from {len(frames)} captures (sharpness score: {best_score:.2f})")
        return best_frame
    
    def _speak_with_interruption_listening(self, text, enable_interruption=True):
        """
        Speak text while optionally listening for interruptions.
        Used for closing message and question answers.
        
        Args:
            text: Text to speak
            enable_interruption: If True, listen for interruptions (for question answers)
                                 If False, just speak normally (for closing message)
            
        Returns:
            True if interrupted, False if completed normally
        """
        if not text:
            return False
        
        # Start speaking (non-blocking)
        self.tts.speak(text, interruptible=enable_interruption)
        
        # Wait a moment for TTS to start
        time.sleep(0.05)  # Minimal delay for TTS initialization
        
        # Only listen for interruptions if enabled (for question answers)
        interrupted = False
        if enable_interruption and self.voice_input:
            Logger.log_system_event("Listening for interruptions while speaking...")
            while self.tts.is_speaking:
                try:
                    # Use a short timeout to check frequently
                    question = self.voice_input.listen_once(timeout=0.5)
                    if question:
                        Logger.log_system_event(f"Interruption detected: {question}")
                        # Stop speaking immediately
                        self.tts.stop_speaking()
                        interrupted = True
                        # Process the interruption as a question
                        self._handle_question(question)
                        break
                except Exception as e:
                    # Ignore timeout errors, continue listening
                    if "timeout" not in str(e).lower() and "WaitTimeoutError" not in str(type(e).__name__):
                        Logger.log_error(str(e), "Interruption listening")
                # Small sleep to prevent CPU spinning
                time.sleep(0.05)  # Reduced for faster response
        
        # Wait for TTS thread to finish (if not interrupted)
        if not interrupted and self.tts.speak_thread and self.tts.speak_thread.is_alive():
            self.tts.speak_thread.join()
        
        return interrupted
    
    def _listen_for_questions(self, duration=2):
        """
        Listen for user questions for a specified duration
        
        Args:
            duration: How long to listen in seconds
        """
        if not self.voice_input:
            return
        
        self.listening_for_questions = True
        Logger.log_system_event(f"Listening for questions for {duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration and self.listening_for_questions:
            try:
                question = self.voice_input.listen_once(timeout=1)
                if question:
                    Logger.log_system_event(f"Question detected: {question}")
                    self._handle_question(question)
                    # Continue listening for more questions
                    start_time = time.time()  # Reset timer after answering
            except Exception as e:
                Logger.log_error(str(e), "Question listening")
                # No delay - check immediately
        
        self.listening_for_questions = False
        Logger.log_system_event("Question listening period ended")
    
    def _open_bins_sequentially(self, classifications):
        """
        Open bins sequentially based on classifications, in parallel with TTS.
        Opens bins in order: bin1 -> wait 1.5s -> bin2 -> wait 7s -> close all.
        
        Args:
            classifications: List of classification dicts with 'bin_type' field
        """
        if not self.servo_controller:
            return
        
        try:
            # Extract unique bin_types in order they appear
            bin_types = []
            seen = set()
            for item in classifications:
                bin_type = item.get('bin_type', '').lower()
                if bin_type and bin_type not in seen:
                    bin_types.append(bin_type)
                    seen.add(bin_type)
            
            if not bin_types:
                return
            
            Logger.log_system_event(f"Opening {len(bin_types)} bin(s) sequentially: {', '.join(bin_types)}")
            
            # Open bins sequentially with 1.5s delay between each
            for bin_type in bin_types:
                self.servo_controller.move_servo_up(bin_type)
                if bin_type != bin_types[-1]:  # Don't wait after last bin
                    time.sleep(1.5)
            
            # Wait 7 seconds after all bins are open
            time.sleep(7.0)
            
            # Close all bins
            for bin_type in bin_types:
                self.servo_controller.move_servo_down(bin_type)
            
            Logger.log_system_event("All bins closed")
        except Exception as e:
            Logger.log_error(str(e), "Opening bins sequentially")
    
    def _handle_bag_detection(self):
        """
        Handle detected trash bags by asking user about contents
        """
        if not self.voice_input:
            Logger.log_system_event("Voice input not available. Cannot ask about bag contents.")
            return
        
        num_bags = len(self.detected_bags)
        
        if num_bags == 1:
            # Single bag
            bag = self.detected_bags[0]
            question = f"I see a trash bag. What is mostly in this bag?"
            Logger.log_tts_output(question)
            self.tts.speak(question)
            
            # Listen for answer (10 seconds)
            answer = self.voice_input.listen_once(timeout=10)
            
            if answer:
                Logger.log_system_event(f"User said bag contains: {answer}")
                # Classify based on answer
                classification = self.classifier.classify_bag_contents(answer)
                
                bin_name = classification.get('bin_name', classification.get('bin_type', 'bin'))
                bin_color = classification.get('bin_color', 'blue')
                
                response = f"This bag goes into the {bin_name} usually {bin_color}."
                Logger.log_tts_output(response)
                self.tts.speak(response)
                
                # Store for display
                self.current_classifications = [{
                    'item': f"Bag ({answer})",
                    'bin_type': classification['bin_type'],
                    'bin_name': bin_name,
                    'bin_color': bin_color,
                    'explanation': classification['explanation']
                }]
            else:
                Logger.log_system_event("No answer received for bag contents.")
                self.tts.speak("I couldn't hear your answer. Please try again.")
        else:
            # Multiple bags
            Logger.log_tts_output(f"I see {num_bags} trash bags. Let me ask about each one.")
            self.tts.speak(f"I see {num_bags} trash bags. Let me ask about each one.")
            # No delay - speak immediately
            
            bag_classifications = []
            
            for i, bag in enumerate(self.detected_bags, 1):
                question = f"What is mostly in bag {i}?"
                Logger.log_tts_output(question)
                self.tts.speak(question)
                
                # Listen for answer (10 seconds)
                answer = self.voice_input.listen_once(timeout=10)
                
                if answer:
                    Logger.log_system_event(f"Bag {i} contains: {answer}")
                    # Classify based on answer
                    classification = self.classifier.classify_bag_contents(answer)
                    
                    bin_name = classification.get('bin_name', classification.get('bin_type', 'bin'))
                    bin_color = classification.get('bin_color', 'blue')
                    
                    response = f"Bag {i} goes into the {bin_name} usually {bin_color}."
                    Logger.log_tts_output(response)
                    self.tts.speak(response)
                    # No delay - speak immediately
                    
                    bag_classifications.append({
                        'item': f"Bag {i} ({answer})",
                        'bin_type': classification['bin_type'],
                        'bin_name': bin_name,
                        'bin_color': bin_color,
                        'explanation': classification['explanation']
                    })
                else:
                    Logger.log_system_event(f"No answer received for bag {i}.")
                    self.tts.speak(f"I couldn't hear your answer for bag {i}.")
                    # No delay - speak immediately
            
            # Store all bag classifications
            self.current_classifications = bag_classifications
            
            # Summary
            if bag_classifications:
                summary = "That's all the bags."
                Logger.log_tts_output(summary)
                self.tts.speak(summary)
        
        # Add closing message
        closing_msg = get_closing_message(self.language)
        Logger.log_tts_output(closing_msg)
        self.tts.speak(closing_msg)
    
    def _handle_question(self, question):
        """
        Handle a user question
        
        Args:
            question: The user's question text
        """
        if not question:
            return
        
        Logger.log_system_event(f"Processing question: {question}")
        
        # Get answer from classifier
        answer = self.classifier.answer_question(question, self.current_classifications)
        
        if answer is None:
            # Question not relevant - don't respond to chit chat
            Logger.log_system_event("Question not relevant - ignoring chit chat")
            return  # Silent - don't respond to non-relevant questions
        else:
            # Clean up answer text - remove "Answer:" prefix if present
            answer_text = answer.strip()
            if answer_text.startswith("Answer:"):
                answer_text = answer_text[7:].strip()
            elif answer_text.startswith("answer:"):
                answer_text = answer_text[7:].strip()
            
            # Relevant question - speak the answer (interruptible for follow-up questions)
            Logger.log_tts_output(answer_text)
            interrupted = self._speak_with_interruption_listening(answer_text, enable_interruption=True)
            # Continue listening for follow-up questions (if not interrupted)
            if self.listening_for_questions and not interrupted:
                pass  # Continue listening in the _listen_for_questions loop
    
    def _check_trash_async(self, frame, check_time):
        """
        First check with YOLOv8 (excluding person), then call Gemini Vision only if objects detected
        
        Args:
            frame: Camera frame (numpy array in BGR format) - already captured
            check_time: Timestamp when check was initiated
        """
        try:
            # First, use YOLOv8 to detect objects (excluding person) with 0.20 confidence threshold
            Logger.log_system_event("Checking for objects with YOLOv8 (excluding person, min confidence: 0.20)...")
            yolo_detections = self.detector.detect_objects(frame, filter_trash_only=True, min_confidence=0.20)
            
            # Only proceed if YOLOv8 detected objects (excluding person)
            if not yolo_detections or len(yolo_detections) == 0:
                
                return
            
            # Log what YOLOv8 detected
            Logger.log_system_event(f"YOLOv8 detected {len(yolo_detections)} object(s) (excluding person):")
            for det in yolo_detections:
                Logger.log_system_event(f"  - {det['class']} (confidence: {det['confidence']:.2f})")
            
            # Record detection time for timing measurement
            self.person_detected_time = time.time()
            
            # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
            # Make sure we're working with a fresh copy
            snapshot_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            
            Logger.log_system_event(f"Objects detected! Calling Gemini Vision for classification...")
            Logger.log_system_event(f"Frame shape: {snapshot_rgb.shape}, timestamp: {check_time}")
            
            # Now call Gemini Vision for accurate classification
            self.processing_detection = True
            
            # Run analysis in background thread to prevent UI freeze
            analysis_thread = threading.Thread(
                target=self._process_detection_async,
                args=(snapshot_rgb, yolo_detections),
                daemon=True
            )
            analysis_thread.start()
        except Exception as e:
            Logger.log_error(f"Error in _check_trash_async: {str(e)}", "Background trash detection")
            import traceback
            Logger.log_error(traceback.format_exc(), "Background trash detection")
            # Make sure flags are reset even on error
            self.checking_trash = False
            self.processing_detection = False
        finally:
            # Reset checking_trash flag immediately after starting thread
            # The processing_detection flag will be reset by the thread when done
            self.checking_trash = False
    
    def _process_detection_async(self, frame, yolo_detections=None):
        """
        Process detection in background thread to prevent UI freeze
        
        Args:
            frame: Camera frame (numpy array in RGB format)
            yolo_detections: List of objects detected by YOLOv8 (optional)
        """
        try:
            self.process_detection([], frame=frame, yolo_detections=yolo_detections)
        except Exception as e:
            Logger.log_error(str(e), "Background detection processing")
        finally:
            self.processing_detection = False
    
    def process_detection(self, food_items, frame=None, yolo_detections=None):
        """
        Process detected items and provide classification using vision
        Analyzes the image directly for trash items (no bag detection)
        Uses the same approach as web app bin layout analysis
        
        Args:
            food_items: List of detected food items (unused, kept for compatibility)
            frame: Current camera frame (numpy array in RGB format)
            yolo_detections: List of objects detected by YOLOv8 (optional, for hints)
        """
        if frame is None:
            Logger.log_error("No frame provided for analysis", "process_detection")
            return

        # Clear previous classifications when starting new detection
        self.current_classifications = []
        self.current_item = None
        
        # Convert numpy array to PIL Image (same as web app does)
        try:
            from PIL import Image
            import numpy as np
            
            # Ensure frame is RGB numpy array
            if not isinstance(frame, np.ndarray):
                Logger.log_error("Frame is not a numpy array", "process_detection")
                return
            
            # Convert to PIL Image (same method as web app)
            # Frame is already RGB from cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) in _check_trash_async
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Already RGB, convert to PIL directly (no BGR conversion needed)
                # Use .copy() to ensure we have a fresh array that won't be overwritten
                frame_array = frame.copy().astype('uint8')
                pil_image = Image.fromarray(frame_array, 'RGB')
            else:
                Logger.log_error(f"Invalid frame shape: {frame.shape}", "process_detection")
                return
            
            Logger.log_system_event(f"Image converted to PIL: {pil_image.size}, mode: {pil_image.mode}")
            
        except Exception as e:
            Logger.log_error(f"Image conversion error: {str(e)}", "process_detection")
            return
        
        # Single photo, single scan - analyze immediately with Gemini Vision
        # Use the same direct API call approach as web app
        # Pass YOLOv8 detections as hints (optional)
        try:
            Logger.log_system_event("Calling Gemini Vision API to analyze image...")
            # Convert YOLOv8 detections to format expected by classifier
            detected_items = None
            if yolo_detections:
                detected_items = [{'class': det['class'], 'confidence': det['confidence']} for det in yolo_detections]
            classifications = self.classifier.classify_item_from_image(pil_image, detected_items=detected_items)
            
            # Log the raw response for debugging
            raw_response = getattr(self.classifier, 'last_raw_response', 'No response')
            Logger.log_system_event(f"Gemini Vision response: {raw_response[:200]}...")
            Logger.log_system_event(f"Found {len(classifications)} trash item(s)")
            
        except Exception as e:
            Logger.log_error(f"Gemini Vision API call failed: {str(e)}", "process_detection")
            import traceback
            Logger.log_error(traceback.format_exc(), "process_detection")
            classifications = []
        
        # Only proceed if trash items were actually found
        if not classifications or len(classifications) == 0:
            Logger.log_system_event("No trash items found in image")
            # Set cooldown only when we actually processed (even if no trash found)
            # This prevents rapid re-processing of the same frame
            self.last_classification_time = time.time()
            return
        
        # Store classifications for display
        self.current_item = classifications[0]
        self.current_classifications = classifications
        Logger.log_system_event(f"Successfully classified {len(classifications)} item(s)")
        
        # Track items for insights
        for item in classifications:
            item_name = item.get('item', 'unknown')
            preferred_bin_type = item.get('bin_type', '').lower()
            bin_available = item.get('bin_available', True)
            alternative_bin = item.get('alternative_bin')
            
            # Determine actual bin the item went to
            actual_bin_type = preferred_bin_type
            is_contaminated = False
            
            if not bin_available:
                if alternative_bin:
                    # Item went to alternative bin because preferred bin is not available
                    actual_bin_type = alternative_bin.get('bin_type', '').lower()
                    # This IS contamination - item doesn't belong in this bin, it's just acceptable
                    # The bin is no longer 100% clean
                    is_contaminated = True
                else:
                    # No bin available - mark as contamination
                    is_contaminated = True
            elif alternative_bin:
                # Item went to alternative bin instead of preferred
                actual_bin_type = alternative_bin.get('bin_type', '').lower()
                if preferred_bin_type != actual_bin_type:
                    # This IS contamination - item doesn't belong in this bin
                    is_contaminated = True
            
            # Add to insights
            self.insights_data['items'].append({
                'item': item_name,
                'preferred_bin': preferred_bin_type,
                'actual_bin': actual_bin_type,
                'timestamp': time.time(),
                'contaminated': is_contaminated
            })
            
            # Update bin counts
            if actual_bin_type in self.insights_data['bin_counts']:
                self.insights_data['bin_counts'][actual_bin_type] += 1
            else:
                # Initialize if bin type not in tracking
                self.insights_data['bin_counts'][actual_bin_type] = 1
                if actual_bin_type not in self.insights_data['contamination']:
                    self.insights_data['contamination'][actual_bin_type] = {
                        'wrong_items': [],
                        'total_items': 0
                    }
            
            # Update contamination tracking
            if actual_bin_type in self.insights_data['contamination']:
                self.insights_data['contamination'][actual_bin_type]['total_items'] += 1
                # Mark as contamination if item went to a different bin than preferred
                # This includes alternative bins - they make the bin "not 100% clean"
                if is_contaminated or (preferred_bin_type != actual_bin_type and preferred_bin_type):
                    # Item is in wrong bin (either no bin available or alternative bin)
                    self.insights_data['contamination'][actual_bin_type]['wrong_items'].append({
                        'item': item_name,
                        'should_be_in': preferred_bin_type
                    })
        
        # Save insights data
        self.save_insights_data()
        
        # Set cooldown timestamp AFTER successful classification
        # This allows new detections after a short cooldown
        self.last_classification_time = time.time()
        
        # Only speak if trash items were found
        # Helper function to get bin color
        def get_bin_color(bin_type):
            bin_type_lower = bin_type.lower()
            if bin_type_lower == 'recycling':
                return 'blue'
            elif bin_type_lower == 'compost':
                return 'green'
            elif bin_type_lower == 'landfill':
                return 'black or grey'
            return bin_type
        
        # Calculate and log response time if person detection time was recorded
        if self.person_detected_time is not None:
            response_time = time.time() - self.person_detected_time
            Logger.log_system_event(f"⏱️  RESPONSE TIME: {response_time:.2f} seconds (person detected → speaking)")
            self.person_detected_time = None  # Reset after logging
        
        # Helper function to get bin name, color, and position
        def get_bin_info(item):
            # Check if bin is available
            bin_available = item.get('bin_available', True)
            alternative_bin = item.get('alternative_bin')
            
            # If bin is not available and we have an alternative
            if not bin_available and alternative_bin:
                # Return alternative bin info
                return (
                    alternative_bin.get('bin_name', 'bin'),
                    alternative_bin.get('bin_color', 'N/A'),
                    alternative_bin.get('bin_position', None),
                    True,  # is_alternative
                    item.get('bin_type'),  # preferred_bin_type
                    item.get('bin_name'),  # preferred_bin_name
                )
            elif not bin_available:
                # No alternative found - return preferred bin info (will need to find closest)
                preferred_type = item.get('bin_type', 'landfill')
                preferred_name = item.get('bin_name', 'bin')
                return (
                    preferred_name,
                    item.get('bin_color', get_bin_color(preferred_type)),
                    item.get('bin_position', None),
                    False,  # is_alternative
                    preferred_type,  # preferred_bin_type
                    preferred_name,  # preferred_bin_name
                )
            
            # Bin is available - normal case
            bin_type = item.get('bin_type')
            if not bin_type:
                return (
                    item.get('bin_name', item.get('bin_type', 'bin')),
                    item.get('bin_color', get_bin_color(bin_type)),
                    item.get('bin_position', None),
                    False,  # is_alternative
                    None,  # preferred_bin_type
                    None,  # preferred_bin_name
                )
            else:
                return (
                    item.get('bin_name', 'bin'),
                    item.get('bin_color', 'N/A'),
                    item.get('bin_position', None),
                    False,  # is_alternative
                    None,  # preferred_bin_type
                    None,  # preferred_bin_name
                )
        
        # Store spoken text for "repeat" functionality
        self.last_spoken_text = []
        
        # Start bin opening in parallel thread (runs while TTS speaks)
        if self.servo_controller and classifications:
            threading.Thread(
                target=self._open_bins_sequentially,
                args=(classifications,),
                daemon=True
            ).start()
        
        if len(classifications) == 1:
            # Single item - combine with closing message for no break
            item = classifications[0]
            bin_info = get_bin_info(item)
            bin_name, bin_color, bin_position, is_alternative, preferred_type, preferred_name = bin_info
            
            # Format response based on whether it's an alternative
            if is_alternative:
                if self.language == "hungarian":
                    response = f"{item['item']} általában a {preferred_name} kukába megy, de mivel az nem elérhető, mehet a {bin_color} {bin_name} {bin_position or ''} kukába"
                else:
                    response = f"{item['item']} should go into {preferred_name}, but since it's not available, it can go into {bin_color} {bin_name} {bin_position or ''}"
            elif not item.get('bin_available', True):
                # No alternative found - need to find closest bin
                closest_bin = self._find_closest_bin(preferred_type)
                if closest_bin:
                    if self.language == "hungarian":
                        response = f"{item['item']} a {preferred_name} kukába megy, de az nem elérhető. Keresse meg a legközelebbi {preferred_name} kukát"
                    else:
                        response = f"{item['item']} should go into {preferred_name}, but it's not available. Find the closest {preferred_name} bin"
                else:
                    if self.language == "hungarian":
                        response = f"{item['item']} a {preferred_name} kukába megy, de az nem elérhető"
                    else:
                        response = f"{item['item']} should go into {preferred_name}, but it's not available"
            else:
                # Normal case - bin is available
                response = format_item_response(item, bin_name, bin_color, bin_position, self.language)
            
            # Speak item classification (non-interruptible)
            Logger.log_tts_output(response)
            self.tts.speak(response, interruptible=False)
            # Wait for speech to finish
            if self.tts.speak_thread and self.tts.speak_thread.is_alive():
                self.tts.speak_thread.join()
            
            # Now speak closing message (non-interruptible, just speak it)
            closing_msg = get_closing_message(self.language)
            self.last_spoken_text.append(response)
            self.last_spoken_text.append(closing_msg)
            Logger.log_tts_output(closing_msg)
            self.tts.speak(closing_msg, interruptible=False)
            # Wait for speech to finish
            if self.tts.speak_thread and self.tts.speak_thread.is_alive():
                self.tts.speak_thread.join()
            interrupted = False  # No interruption during initial classification
        elif len(classifications) > 1:
            # Multiple items - combine all into one continuous speech to eliminate gaps
            all_responses = []
            for i, item in enumerate(classifications, 1):
                bin_info = get_bin_info(item)
                bin_name, bin_color, bin_position, is_alternative, preferred_type, preferred_name = bin_info
                
                # Format response based on whether it's an alternative
                if is_alternative:
                    if self.language == "hungarian":
                        response = f"{item['item']} általában a {preferred_name} kukába megy, de mivel az nem elérhető, mehet a {bin_color} {bin_name} {bin_position or ''} kukába"
                    else:
                        response = f"{item['item']} should go into {preferred_name}, but since it's not available, it can go into {bin_color} {bin_name} {bin_position or ''}"
                elif not item.get('bin_available', True):
                    # No alternative found
                    closest_bin = self._find_closest_bin(preferred_type)
                    if closest_bin:
                        if self.language == "hungarian":
                            response = f"{item['item']} a {preferred_name} kukába megy, de az nem elérhető. Keresse meg a legközelebbi {preferred_name} kukát"
                        else:
                            response = f"{item['item']} should go into {preferred_name}, but it's not available. Find the closest {preferred_name} bin"
                    else:
                        if self.language == "hungarian":
                            response = f"{item['item']} a {preferred_name} kukába megy, de az nem elérhető"
                        else:
                            response = f"{item['item']} should go into {preferred_name}, but it's not available"
                else:
                    # Normal case
                    response = format_item_response(item, bin_name, bin_color, bin_position, self.language)
                
                self.last_spoken_text.append(response)
                all_responses.append(response)
            
            # Combine all items with periods and speak (non-interruptible)
            combined_items = ". ".join(all_responses)
            Logger.log_tts_output(combined_items)
            self.tts.speak(combined_items, interruptible=False)
            # Wait for speech to finish
            if self.tts.speak_thread and self.tts.speak_thread.is_alive():
                self.tts.speak_thread.join()
            
            # Now speak closing message (non-interruptible, just speak it)
            closing_msg = get_closing_message(self.language)
            self.last_spoken_text.append(closing_msg)
            Logger.log_tts_output(closing_msg)
            self.tts.speak(closing_msg, interruptible=False)
            # Wait for speech to finish
            if self.tts.speak_thread and self.tts.speak_thread.is_alive():
                self.tts.speak_thread.join()
            interrupted = False  # No interruption during initial classification
        
        # Listen for questions for 2 seconds after closing message (only if not interrupted)
        if self.voice_input and not interrupted:
            self._listen_for_questions(2)
    
    
    def run(self):
        """
        Main application loop
        """
        # Initialize camera (set CAMERA_INDEX at top of file to switch feeds)
        Logger.log_system_event(f"Opening camera index {CAMERA_INDEX}. Change CAMERA_INDEX to target Continuity Camera.")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            Logger.log_error(
                f"Camera index {CAMERA_INDEX} could not be opened. "
                "If you're trying to use Continuity Camera, unlock the iPhone and select it in another app.",
                "Camera initialization"
            )
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\nCamera started. YOLOv8 is running continuously!")
        print("System is AUTOMATIC - items will be classified automatically when detected!")
        print("Controls:")
        print("  'q' - Quit")
        
        frame_count = 0
        # No longer tracking person detection - using trash detection instead
        
        Logger.log_system_event("Camera started. Using YOLOv8 for object detection (excluding person)!")
        Logger.log_system_event("When objects are detected by YOLOv8, image will be analyzed using Gemini Vision for accurate classification!")
        Logger.log_system_event("Controls: 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check for bin layout reload signal (check every 30 frames to avoid overhead)
                if frame_count % 30 == 0:
                    if self.reload_signal_path.exists():
                        try:
                            signal_time = float(self.reload_signal_path.read_text())
                            if signal_time > self.last_reload_time:
                                self.reload_bin_layout()
                                self.last_reload_time = signal_time
                                # Remove signal file after processing
                                self.reload_signal_path.unlink()
                        except (ValueError, OSError) as e:
                            Logger.log_error(str(e), "Reading reload signal")
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Check for trash using Gemini Vision directly (instant detection)
                frame_count += 1
                current_time = time.time()
                time_elapsed = current_time - self.last_classification_time > self.detection_cooldown
                trash_check_elapsed = current_time - self.last_trash_check_time > self.trash_check_interval
                
                # Only check for trash if:
                # 1. Enough time has passed since last classification (cooldown)
                # 2. Enough time has passed since last trash check (throttling to reduce API calls)
                # 3. We're not already processing a detection
                # 4. We're not already checking for trash (prevent multiple simultaneous API calls)
                if not self.processing_detection and not self.checking_trash and time_elapsed and trash_check_elapsed:
                    # Update last trash check time immediately to prevent rapid calls
                    self.last_trash_check_time = current_time
                    self.checking_trash = True
                    
                    # CRITICAL: Make a deep copy of the frame to prevent it from being overwritten
                    # by subsequent camera reads in the main loop
                    frame_copy = frame.copy()
                    
                    # Capture single clear photo and analyze immediately
                    # Pass a copy so the frame doesn't get overwritten by new camera reads
                    trash_check_thread = threading.Thread(
                        target=self._check_trash_async,
                        args=(frame_copy, current_time),
                        daemon=True
                    )
                    trash_check_thread.start()
                
                # Draw status indicator
                if self.processing_detection:
                    status_text = "ANALYZING..."
                    cv2.putText(frame, status_text, 
                              (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show current classification results if available
                if self.current_classifications:
                    y_offset = 60
                    for i, item in enumerate(self.current_classifications[:3]):  # Show up to 3 items
                        # Get bin color
                        bin_color = (255, 255, 255)  # white default
                        if item['bin_type'].lower() == 'recycling':
                            bin_color = (255, 100, 0)  # blue-ish
                        elif item['bin_type'].lower() == 'compost':
                            bin_color = (0, 255, 0)  # green
                        elif item['bin_type'].lower() == 'landfill':
                            bin_color = (100, 100, 100)  # grey
                        
                        text = f"{item['item']} -> {item['bin_type'].upper()} BIN"
                        cv2.putText(frame, text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, bin_color, 2)
                        y_offset += 25
                    if len(self.current_classifications) > 3:
                        cv2.putText(frame, f"... and {len(self.current_classifications) - 3} more", 
                                   (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show instructions
                status_text = "PERSON DETECTION MODE"
                if self.listening_for_questions:
                    status_text += " - Listening for questions..."
                status_text += " - Press 'q' to quit"
                cv2.putText(frame, status_text,
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the frame
                cv2.imshow('Smart Trash Bin - Detection Running', frame)
                
                # Handle keyboard input ONCE per frame (fix double waitKey)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    Logger.log_system_event("Quit key pressed. Shutting down...")
                    break
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.listening_for_questions = False  # Stop listening
            if self.voice_input:
                self.voice_input.stop_listening()
            if self.servo_controller:
                self.servo_controller.disconnect_all()
            print("System shut down successfully.")


def main():
    """
    Entry point for the application
    """
    try:
        app = SmartTrashBin()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()