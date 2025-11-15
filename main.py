"""
Smart Trash Bin Detection System
Main application that integrates object detection, Gemini AI, and ElevenLabs TTS
"""

import cv2
import time
import threading
from datetime import datetime
from object_detector import ObjectDetector
from gemini_classifier import TrashClassifier
from tts_handler import TTSHandler


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
        
        try:
            Logger.log_system_event("Initializing Gemini AI classifier...")
            self.classifier = TrashClassifier()
            Logger.log_system_event("Gemini AI initialized successfully")
        except Exception as e:
            Logger.log_error(str(e), "TrashClassifier initialization")
            raise
        
        try:
            Logger.log_system_event("Initializing TTS handler...")
            self.tts = TTSHandler()
            Logger.log_system_event("TTS handler initialized")
        except Exception as e:
            Logger.log_error(str(e), "TTSHandler initialization")
            raise
        
        # Voice input disabled - system only detects trash
        self.voice_input = None
        Logger.log_system_event("Voice input disabled - focusing on trash detection only")
        
        # State management
        self.last_detection_time = 0
        self.detection_cooldown = 5  # seconds between detections (increased for automatic mode)
        self.current_item = None
        self.current_classifications = []  # Store all classifications
        self.processing_question = False
        
        Logger.log_system_event("System fully initialized and ready!")
        print("\n" + "="*80)
        print("SYSTEM READY - Automatic trash detection active")
        print("="*80)
        print("Controls: 'q' to quit")
        print("="*80 + "\n")
    
    def process_detection(self, food_items, frame=None):
        """
        Process detected food items and provide classification using vision
        Automatically classifies ALL items in the image
        
        Args:
            food_items: List of detected food items
            frame: Current camera frame (numpy array)
        """
        # Now using Gemini Vision to identify trash - no YOLOv8 trash detection
        # food_items parameter is kept for compatibility but not used
        if frame is None:
            Logger.log_error("No frame provided for analysis", "process_detection")
            return
        
        Logger.log_system_event("Analyzing image for trash items using Gemini Vision...")
        Logger.log_llm_request("Analyzing image for all visible trash/waste items", image_sent=True)
        
        # Use Gemini Vision to identify trash in the image
        # Pass None for detected_items since we're not using YOLOv8 detections
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
        if len(classifications) == 1:
            # Single item
            item = classifications[0]
            response = f"I see {item['item']}. This goes in the {item['bin_type'].upper()} bin. {item['explanation']}"
            Logger.log_tts_output(response)
            self.tts.speak(response)
        elif len(classifications) > 1:
            # Multiple items - speak each one
            for i, item in enumerate(classifications, 1):
                response = f"{item['item']} goes in the {item['bin_type'].upper()} bin."
                Logger.log_tts_output(response)
                self.tts.speak(response)
                time.sleep(0.5)  # Small pause between items
    
    
    def run(self):
        """
        Main application loop
        """
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
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
        person_detected_last_frame = False
        
        Logger.log_system_event("Camera started. YOLOv8 is detecting people only!")
        Logger.log_system_event("When a person is detected, image will be analyzed for trash using Gemini Vision!")
        Logger.log_system_event("Controls: 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Run YOLOv8 person detection every 5 frames
                frame_count += 1
                if frame_count % 5 == 0:
                    person_detected = self.detector.detect_person(frame)
                    
                    current_time = time.time()
                    time_elapsed = current_time - last_classification_time > self.detection_cooldown
                    
                    # If person just appeared (wasn't there before) or enough time has passed
                    if person_detected and (not person_detected_last_frame or time_elapsed):
                        Logger.log_system_event("Person detected! Capturing image and analyzing for trash...")
                        # Take snapshot of current frame
                        snapshot = frame.copy()
                        # Convert BGR to RGB for PIL
                        snapshot_rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
                        
                        # Analyze image for trash using Gemini Vision
                        # Pass empty list since we're not using YOLOv8 for trash detection
                        Logger.log_system_event("Sending image to Gemini Vision for trash analysis...")
                        self.process_detection([], frame=snapshot_rgb)
                        last_classification_time = current_time
                    
                    person_detected_last_frame = person_detected
                    
                    # Draw person detection indicator on frame (silent - no speech)
                    if person_detected:
                        cv2.putText(frame, "PERSON DETECTED - Checking for trash...", 
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
                cv2.putText(frame, "PERSON DETECTION MODE - Press 'q' to quit", 
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

