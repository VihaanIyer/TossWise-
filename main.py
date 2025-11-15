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

# Modify this when you want to switch cameras (Continuity Cam often shows as index 1 or 2)
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
                        import json
                        self.bin_layout_metadata = json.load(f)
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
        
        # State management
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # Cooldown between detections to prevent spam and reduce API calls
        self.last_trash_check_time = 0
        self.trash_check_interval = 2.0  # Only check for trash every 2 seconds to reduce lag (increased from 0.5)
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
        
        # File watcher for bin layout reload
        self.last_reload_time = 0
        self.reload_signal_path = Path("reload_signal.txt")
        
        Logger.log_system_event("System fully initialized and ready!")
        Logger.log_system_event("Bin layout reload monitoring active - will auto-reload when web app updates bins")
        print("\n" + "="*80)
        print("SYSTEM READY - Automatic trash detection active")
        print("="*80)
        print("Controls: 'q' to quit")
        print("="*80 + "\n")
    
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
                time.sleep(0.05)  # Optimized delay
        
        self.listening_for_questions = False
        Logger.log_system_event("Question listening period ended")
    
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
            time.sleep(0.05)  # Optimized delay
            
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
                    time.sleep(0.05)
                    
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
                    time.sleep(0.05)
            
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
            # Relevant question - speak the answer
            Logger.log_tts_output(f"Answer: {answer}")
            self.tts.speak(answer)
            # Continue listening for follow-up questions
            if self.listening_for_questions:
                time.sleep(0.05)  # Optimized delay  # Brief pause before continuing to listen
    
    def _check_trash_async(self, frame, check_time):
        """
        Check for trash objects in background thread to prevent blocking main loop
        
        Args:
            frame: Camera frame (numpy array in BGR format)
            check_time: Timestamp when check was initiated
        """
        try:
            # Use Roboflow to detect trash objects (this can take time, so it's in background)
            trash_detections = self.detector.detect_trash_objects(frame)
            
            # Only trigger analysis if trash objects are detected
            if trash_detections and len(trash_detections) > 0:
                # Record detection time for timing measurement
                self.person_detected_time = time.time()
                Logger.log_system_event(f"Trash objects detected ({len(trash_detections)} items)! Analyzing image...")
                
                # Log detected items
                for det in trash_detections:
                    Logger.log_system_event(f"  - {det['class']} (confidence: {det['confidence']:.2f})")
                
                # Use current frame directly for fastest response
                snapshot_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Analyze image for trash using Gemini Vision (in background thread)
                Logger.log_system_event("Starting background analysis of image...")
                self.processing_detection = True
                # Run analysis in background thread to prevent UI freeze
                analysis_thread = threading.Thread(
                    target=self._process_detection_async,
                    args=(snapshot_rgb,),
                    daemon=True
                )
                analysis_thread.start()
        except Exception as e:
            Logger.log_error(str(e), "Background trash detection")
        finally:
            self.checking_trash = False
    
    def _process_detection_async(self, frame):
        """
        Process detection in background thread to prevent UI freeze
        
        Args:
            frame: Camera frame (numpy array in RGB format)
        """
        try:
            self.process_detection([], frame=frame)
        except Exception as e:
            Logger.log_error(str(e), "Background detection processing")
        finally:
            self.processing_detection = False
    
    def process_detection(self, food_items, frame=None):
        """
        Process detected items and provide classification using vision
        First checks for trash bags, then falls back to individual items
        
        Args:
            food_items: List of detected food items
            frame: Current camera frame (numpy array)
        """
        if frame is None:
            Logger.log_error("No frame provided for analysis", "process_detection")
            return
        
        # First, check for trash bags
        Logger.log_system_event("Checking for trash bags in image...")
        try:
            bags = self.classifier.detect_bags_in_image(frame)
            Logger.log_system_event(f"Bag detection result: {len(bags)} bags found")
        except Exception as e:
            Logger.log_error(str(e), "Bag detection")
            bags = []
        
        # If bags detected, handle bag workflow
        if bags and len(bags) > 0:
            self.detected_bags = bags
            self.current_bag_index = 0
            Logger.log_system_event(f"Found {len(bags)} trash bag(s). Starting bag content questions...")
            self._handle_bag_detection()
            return
        
        # No bags found, proceed with individual item detection
        Logger.log_system_event("No bags detected. Analyzing image for individual trash items...")
        Logger.log_llm_request("Analyzing image for all visible trash/waste items", image_sent=True)
        
        # Use Gemini Vision to identify trash in the image
        try:
            classifications = self.classifier.classify_item_from_image(frame, detected_items=None)
            raw_response = getattr(self.classifier, 'last_raw_response', 'Response received')
            Logger.log_llm_response(raw_response, classifications)
        except Exception as e:
            Logger.log_error(str(e), "Gemini Vision API call")
            classifications = []
        
        # Only proceed if trash items were actually found
        if not classifications or len(classifications) == 0:
            Logger.log_system_event("No trash items found in image. No action taken.")
            return
        
        # Store classifications for display
        self.current_item = classifications[0]
        self.current_classifications = classifications
        
        # Log classification summary
        Logger.log_classification_summary(classifications)
        
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
            # Handle items with no bin available
            if item.get('no_bin_available'):
                # Return preferred bin info for display
                bin_name = item.get('bin_name', 'bin')
                bin_color = item.get('bin_color', 'N/A')
                bin_position = item.get('bin_position', None)
                return bin_name, bin_color, bin_position
            
            bin_type = item.get('bin_type')
            if not bin_type:
                bin_name = item.get('bin_name', 'bin')
                bin_color = item.get('bin_color', 'N/A')
                bin_position = item.get('bin_position', None)
            else:
                bin_name = item.get('bin_name', item.get('bin_type', 'bin'))
                bin_color = item.get('bin_color', get_bin_color(bin_type))
                bin_position = item.get('bin_position', None)
            return bin_name, bin_color, bin_position
        
        # Store spoken text for "repeat" functionality
        self.last_spoken_text = []
        
        if len(classifications) == 1:
            # Single item - natural, fast response
            item = classifications[0]
            bin_name, bin_color, bin_position = get_bin_info(item)
            # Use format: "item goes into [color] bin_name position"
            response = format_item_response(item, bin_name, bin_color, bin_position, self.language)
            self.last_spoken_text.append(response)
            Logger.log_tts_output(response)
            self.tts.speak(response)  # This is blocking, will wait for audio to finish
            # Add closing message
            closing_msg = get_closing_message(self.language)
            self.last_spoken_text.append(closing_msg)
            Logger.log_tts_output(closing_msg)
            self.tts.speak(closing_msg)  # This is blocking, will wait for audio to finish
        elif len(classifications) > 1:
            # Multiple items - speak each one naturally
            for i, item in enumerate(classifications, 1):
                bin_name, bin_color, bin_position = get_bin_info(item)
                response = format_item_response(item, bin_name, bin_color, bin_position, self.language)
                self.last_spoken_text.append(response)
                Logger.log_tts_output(response)
                self.tts.speak(response)  # This is blocking, will wait for audio to finish
                # No sleep needed - TTS is already blocking
            # Add closing message after all items
            closing_msg = get_closing_message(self.language)
            self.last_spoken_text.append(closing_msg)
            Logger.log_tts_output(closing_msg)
            self.tts.speak(closing_msg)  # This is blocking, will wait for audio to finish
        
        # Listen for questions for 2 seconds after speaking
        if self.voice_input:
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
        last_classification_time = 0
        # No longer tracking person detection - using trash detection instead
        
        Logger.log_system_event("Camera started. Using Roboflow YOLOv8 for trash detection!")
        Logger.log_system_event("When trash objects are detected, image will be analyzed using Gemini Vision for accurate classification!")
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
                
                # Run Roboflow YOLOv8 trash detection with throttling to reduce lag
                frame_count += 1
                current_time = time.time()
                time_elapsed = current_time - last_classification_time > self.detection_cooldown
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
                    
                    # Run Roboflow detection in background thread to prevent blocking
                    # Capture current frame for the thread
                    frame_copy = frame.copy()
                    trash_check_thread = threading.Thread(
                        target=self._check_trash_async,
                        args=(frame_copy, current_time),
                        daemon=True
                    )
                    trash_check_thread.start()
                
                # Draw status indicator
                if self.checking_trash:
                    status_text = "Checking for trash..."
                    cv2.putText(frame, status_text, 
                              (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif self.processing_detection:
                    status_text = "ANALYZING IMAGE..."
                    cv2.putText(frame, status_text, 
                              (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Check for quit key (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    Logger.log_system_event("Quit key pressed. Shutting down...")
                    break
                
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
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
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
