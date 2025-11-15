"""
Voice Input Handler
Listens for user questions using speech recognition
"""

import speech_recognition as sr
import threading
import queue


class VoiceInputHandler:
    def __init__(self):
        """
        Initialize speech recognition
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.question_queue = queue.Queue()
        
        # Adjust for ambient noise
        print("Adjusting for ambient noise... Please wait.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready to listen!")
    
    def listen_once(self, timeout=5):
        """
        Listen for a single question
        
        Args:
            timeout: Maximum time to wait for input
            
        Returns:
            Recognized text or None
        """
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Could not request results: {e}")
                return None
                
        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except Exception as e:
            print(f"Error in voice input: {e}")
            return None
    
    def start_continuous_listening(self):
        """
        Start continuous listening in a separate thread
        """
        if self.is_listening:
            return
        
        self.is_listening = True
        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()
    
    def stop_listening(self):
        """
        Stop continuous listening
        """
        self.is_listening = False
    
    def _listen_loop(self):
        """
        Internal loop for continuous listening
        """
        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"Question detected: {text}")
                    self.question_queue.put(text)
                except (sr.UnknownValueError, sr.WaitTimeoutError):
                    pass
                except sr.RequestError as e:
                    print(f"Recognition error: {e}")
                    
            except Exception as e:
                if self.is_listening:
                    print(f"Error in listening loop: {e}")
    
    def get_question(self):
        """
        Get a question from the queue (non-blocking)
        
        Returns:
            Question text or None if queue is empty
        """
        try:
            return self.question_queue.get_nowait()
        except queue.Empty:
            return None

