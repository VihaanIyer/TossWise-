"""
Gemini API Integration for Trash Classification
Determines which bin (recycling, compost, landfill) an item should go into
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()


class TrashClassifier:
    def __init__(self):
        """
        Initialize Gemini API client
        """
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # List available models to find vision-capable ones
        available_model_names = []
        standard_vision_models = []  # Prioritize these
        try:
            available_models = genai.list_models()
            for model in available_models:
                model_name = model.name.replace('models/', '')
                available_model_names.append(model_name)
                # Check if model supports vision and is a standard model (not audio/specialized)
                if hasattr(model, 'supported_generation_methods'):
                    if 'generateContent' in model.supported_generation_methods:
                        # Check if it's a standard vision model (exclude audio/specialized)
                        is_standard = (
                            any(x in model_name.lower() for x in ['flash', '1.5', '2.0', '2.5']) and
                            'audio' not in model_name.lower() and
                            'tts' not in model_name.lower() and
                            'native-audio' not in model_name.lower() and
                            'thinking' not in model_name.lower() and
                            'robotics' not in model_name.lower()
                        )
                        if is_standard:
                            standard_vision_models.append(model_name)
                            print(f"Found standard vision model: {model_name}")
        except Exception as e:
            print(f"Could not list models: {e}")
        
        # Use available models - prioritize free-tier friendly models
        # Build model list: prioritize models with better free tier quotas
        model_attempts = []
        
        # First, prioritize free-tier friendly models (better quotas)
        # Note: gemini-1.5-flash may not be available in v1beta, use 2.0+ models
        free_tier_models = [
            'gemini-2.0-flash',           # Good free tier quota, available in v1beta
            'gemini-2.5-flash',           # Latest, available in v1beta
            'gemini-1.5-flash',           # Try this if 2.0+ don't work
            'gemini-1.5-pro',             # Decent free tier
        ]
        
        for model_name in free_tier_models:
            if model_name not in model_attempts:
                model_attempts.append(model_name)
        
        # Then add other standard vision models we found (but skip preview/exp models)
        for model_name in standard_vision_models:
            # Skip preview/experimental models (they have stricter quotas)
            if 'preview' not in model_name.lower() and 'exp' not in model_name.lower():
                if model_name not in model_attempts:
                    model_attempts.append(model_name)
        
        # Then add known good models
        known_good_models = [
            'gemini-2.5-flash',           # Latest standard vision model
            'gemini-2.0-flash-001',       # Specific version
            'gemini-pro-vision',          # Legacy vision model
        ]
        
        for model_name in known_good_models:
            if model_name not in model_attempts:
                model_attempts.append(model_name)
        
        # Finally, text-only fallback
        model_attempts.append('gemini-pro')
        
        self.model = None
        self.model_name = None
        self.supports_vision = False
        
        for model_name in model_attempts:
            try:
                # Skip audio/specialized models
                if any(x in model_name.lower() for x in ['audio', 'tts', 'native-audio', 'thinking', 'robotics']):
                    print(f"Skipping specialized model: {model_name}")
                    continue
                
                # Try both with and without 'models/' prefix
                model_variants = [model_name, f"models/{model_name}"]
                loaded = False
                
                for variant in model_variants:
                    try:
                        self.model = genai.GenerativeModel(variant)
                        self.model_name = model_name  # Store without prefix for display
                        # Determine if it supports vision - standard flash/pro models do
                        if any(x in model_name.lower() for x in ['vision', '1.5', 'flash', '2.0', '2.5']):
                            self.supports_vision = True
                        print(f"Successfully loaded model: {model_name} (vision support: {self.supports_vision})")
                        loaded = True
                        break
                    except Exception as variant_error:
                        continue
                
                if loaded:
                    break
            except Exception as e:
                print(f"Failed to load {model_name}: {str(e)[:100]}")
                continue
        
        if self.model is None:
            raise ValueError("Could not initialize any Gemini model. Please check your API key and model availability.")
        
        # Initialize for logging
        self.last_raw_response = ""
        
        # System prompt for trash classification
        self.system_prompt = """You are a smart trash bin assistant. Your job is to classify items into the correct waste bin with DEFINITIVE answers.

BIN COLORS AND RULES:

1. BLUE BIN - RECYCLING (The "bookworm" bin)
   It loves:
   - Paper
   - Cardboard
   - Magazines
   - Cereal boxes
   - Clean plastics (varies by city)
   - Metal cans
   - Glass
   Avoid:
   - Greasy pizza boxes
   - Anything wet
   - Plastic bags (usually)

2. GREEN BIN - COMPOST/ORGANICS (The "nature kid" bin)
   It eats:
   - Food scraps
   - Fruit and veggie peels
   - Coffee grounds
   - Tea bags
   - Yard waste
   Avoid:
   - Plastic
   - Meat or dairy in some city systems
   - Anything non-biodegradable

3. BLACK/GREY BIN - LANDFILL/TRASH (The "last resort" bin)
   It swallows:
   - Non-recyclables
   - Styrofoam
   - Plastic wrappers
   - Broken mixed-material items
   - Contaminated items

IMPORTANT: Give DEFINITIVE answers. Say "This goes in [BIN]" or "This should go in [BIN]". Only use uncertain language like "typically" or "usually" if you genuinely cannot determine the correct bin."""
        self.bin_context = ""
        self.bin_layout_metadata = None
    
    def _ensure_pil_image(self, image):
        """
        Convert numpy arrays or file paths to PIL Images for Gemini API calls.
        """
        if image is None:
            return None
        
        if isinstance(image, Image.Image):
            return image
        
        # If it's a numpy array, convert from BGR (OpenCV) to RGB
        if hasattr(image, 'shape'):
            import numpy as np
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image[:, :, ::-1]
                return Image.fromarray(image_rgb)
            return Image.fromarray(image)
        
        # If it's a path-like object
        if isinstance(image, str):
            return Image.open(image)
        
        raise ValueError("Unsupported image format provided to Gemini classifier.")
    
    def classify_item(self, item_name, context="", image=None):
        """
        Classify an item into the appropriate trash bin
        
        Args:
            item_name: Name of the detected item (from YOLOv8)
            context: Additional context (optional)
            image: PIL Image or numpy array of the detected item (optional)
            
        Returns:
            Dictionary with bin type and explanation
        """
        prompt = f"{self.system_prompt}\n\nItem detected by camera: {item_name}"
        if context:
            prompt += f"\nContext: {context}"
        if self.bin_context:
            prompt += f"\nLocal bin labels: {self.bin_context}"
        prompt += "\n\nLook at the image and determine which bin this item should go into. Consider the material, condition, and type of item."
        
        try:
            if image is not None:
                # Convert numpy array to PIL Image if needed
                if not isinstance(image, Image.Image):
                    if hasattr(image, 'shape'):  # numpy array
                        # Convert BGR to RGB for PIL
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            import numpy as np
                            # OpenCV uses BGR, PIL uses RGB
                            image_rgb = image[:, :, ::-1]  # BGR to RGB
                            image = Image.fromarray(image_rgb)
                        else:
                            image = Image.fromarray(image)
                
                # Use vision model with image
                response = self.model.generate_content([prompt, image])
            else:
                # Text-only classification
                response = self.model.generate_content(prompt)
            
            response_text = response.text
            
            # Parse response
            bin_type = "landfill"  # default
            explanation = response_text
            
            # Try to extract bin type from response
            if "recycling" in response_text.lower():
                bin_type = "recycling"
            elif "compost" in response_text.lower():
                bin_type = "compost"
            elif "landfill" in response_text.lower():
                bin_type = "landfill"
            
            return {
                'bin_type': bin_type,
                'explanation': explanation,
                'item': item_name
            }
        except Exception as e:
            print(f"Error classifying item: {e}")
            return {
                'bin_type': 'landfill',
                'explanation': f"Unable to classify {item_name}. Please check local recycling guidelines.",
                'item': item_name
            }
    
    def classify_item_from_image(self, image, detected_items=None):
        """
        Classify items directly from image using vision
        
        Args:
            image: PIL Image or numpy array
            detected_items: List of detected items from YOLOv8 (optional)
            
        Returns:
            Dictionary with bin type and explanation
        """
        # Convert numpy array to PIL Image if needed
        if not isinstance(image, Image.Image):
            if hasattr(image, 'shape'):  # numpy array
                import numpy as np
                # OpenCV uses BGR, PIL uses RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = image[:, :, ::-1]  # BGR to RGB
                    image = Image.fromarray(image_rgb)
                else:
                    image = Image.fromarray(image)
        
        items_text = ""
        if detected_items:
            items_list = [item['class'] for item in detected_items]
            items_text = f"Note: YOLOv8 object detection suggested these items might be present: {', '.join(items_list)}. However, please look at the ACTUAL IMAGE and identify what trash/waste items are REALLY visible."
        
        bin_context_text = ""
        if self.bin_context:
            bin_context_text = f"Facility bin labels previously identified: {self.bin_context}. Use these actual bin labels/types when describing where each item should go."
        
        prompt = f"""{self.system_prompt}

{bin_context_text}

{items_text}

CRITICAL: Look at the ACTUAL IMAGE carefully. Ignore any incorrect detections. Identify ONLY trash/waste items that are ACTUALLY visible in the image. 

Focus on:
- Items that are trash/waste (bottles, food, containers, paper, etc.)
- Items held in hand or placed near the camera
- Items that need to be disposed of

IMPORTANT: Only identify items that are clearly trash/waste. Do NOT identify people, furniture, or other non-trash items. If you don't see any trash, respond with "No trash items found."

For EACH trash item you ACTUALLY see in the image, determine which waste bin it should go into.

IMPORTANT: List each item separately with its bin classification. Format your response like this:
- [Item 1 name]: [BIN_TYPE] - [brief explanation]
- [Item 2 name]: [BIN_TYPE] - [brief explanation]
- [Item 3 name]: [BIN_TYPE] - [brief explanation]

For example:
- Plastic bottle: RECYCLING - This goes in the blue recycling bin. Clean plastic bottles are recyclable.
- Pizza slice: COMPOST - This goes in the green compost bin. Food waste belongs in compost.
- Paper plate: RECYCLING - This goes in the blue recycling bin. Clean paper can be recycled.
- Napkin: COMPOST - This goes in the green compost bin. Paper napkins with food residue belong in compost.

Consider for each item:
- The material (plastic, paper, metal, organic, etc.)
- The condition (clean, contaminated, has food residue, etc.)
- The type of item

List ALL trash/waste items you can ACTUALLY see in the image. If you see a bottle, say "bottle". If you see food, say the food item. Be accurate to what's actually in the image."""
        
        try:
            # Try vision API if model supports it
            if self.supports_vision:
                try:
                    # Ensure image is PIL Image
                    if not isinstance(image, Image.Image):
                        import numpy as np
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            image_rgb = image[:, :, ::-1]  # BGR to RGB
                            image = Image.fromarray(image_rgb)
                        else:
                            image = Image.fromarray(image)
                    
                    # Send image to Gemini - standard format for vision models
                    print(f"📸 Sending image to {self.model_name} for analysis...")
                    # Standard format: [prompt, PIL_Image]
                    response = self.model.generate_content([prompt, image])
                    response_text = response.text
                    print("✅ Vision API successful! Image analyzed by Gemini.")
                except Exception as vision_error:
                    # If vision fails, try text-only with description
                    error_str = str(vision_error)
                    if 'quota' in error_str.lower() or '429' in error_str:
                        print(f"⚠️ Quota exceeded, using text-only classification")
                    else:
                        print(f"⚠️ Vision API failed, using text-only: {error_str[:100]}")
                    
                    # Use text-only classification with item names
                    text_prompt = f"""You are a smart trash bin assistant. Classify these items into the correct waste bin:
- RECYCLING: Paper, cardboard, plastic bottles/containers, metal cans, glass
- COMPOST: Food scraps, fruit peels, vegetable scraps, coffee grounds, tea bags
- LANDFILL: Non-recyclable plastics, styrofoam, mixed materials, contaminated items

Items detected: {items_text}

For each item, determine which bin it should go into. Format your response like this:
- [Item 1]: [BIN_TYPE] - [brief explanation]
- [Item 2]: [BIN_TYPE] - [brief explanation]

List all items with their bin classifications."""
                    
                    try:
                        response = self.model.generate_content(text_prompt)
                        response_text = response.text
                    except Exception as e:
                        # If even text fails, use simple rule-based classification
                        print(f"⚠️ Text API also failed, using rule-based classification")
                        # Rule-based returns formatted string, parse it
                        rule_based_result = self._rule_based_classification(detected_items)
                        # Parse the rule-based result into the expected format
                        items_classifications = self._parse_rule_based_response(rule_based_result, detected_items)
                        if items_classifications:
                            self.last_raw_response = rule_based_result
                            return items_classifications
                        response_text = rule_based_result
            else:
                # Text-only model - use item names
                print("⚠️ Model doesn't support vision, using text-only classification")
                text_prompt = f"""You are a smart trash bin assistant. Classify these items into the correct waste bin:
- RECYCLING: Paper, cardboard, plastic bottles/containers, metal cans, glass
- COMPOST: Food scraps, fruit peels, vegetable scraps, coffee grounds, tea bags
- LANDFILL: Non-recyclable plastics, styrofoam, mixed materials, contaminated items

Items detected: {items_text}

For each item, determine which bin it should go into. Format your response like this:
- [Item 1]: [BIN_TYPE] - [brief explanation]
- [Item 2]: [BIN_TYPE] - [brief explanation]

List all items with their bin classifications."""
                
                try:
                    response = self.model.generate_content(text_prompt)
                    response_text = response.text
                except Exception as e:
                    # If API fails, use rule-based classification
                    print(f"⚠️ API failed, using rule-based classification: {e}")
                    # Rule-based returns formatted string, parse it
                    rule_based_result = self._rule_based_classification(detected_items)
                    # Parse the rule-based result into the expected format
                    items_classifications = self._parse_rule_based_response(rule_based_result, detected_items)
                    if items_classifications:
                        self.last_raw_response = rule_based_result
                        return items_classifications
                    response_text = rule_based_result
            
            # Store raw response for logging
            self.last_raw_response = response_text
            
            # Check if response indicates no trash found
            response_lower = response_text.lower()
            if any(phrase in response_lower for phrase in ['no trash', 'no items', 'no waste', 'nothing', 'no visible']):
                # No trash found - return empty list
                return []
            
            # Parse response to extract all items and their classifications
            items_classifications = []
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Look for patterns like "- Item: BIN_TYPE - explanation" or "Item: BIN_TYPE"
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        item_name = parts[0].replace('-', '').strip()
                        rest = parts[1].strip()
                        
                        # Extract bin type
                        bin_type = "landfill"  # default
                        rest_lower = rest.lower()
                        if "recycling" in rest_lower:
                            bin_type = "recycling"
                        elif "compost" in rest_lower:
                            bin_type = "compost"
                        elif "landfill" in rest_lower:
                            bin_type = "landfill"
                        
                        # Extract explanation (everything after bin type)
                        explanation = rest
                        if '-' in rest:
                            explanation = rest.split('-', 1)[-1].strip()
                        
                        if item_name:
                            items_classifications.append({
                                'item': item_name,
                                'bin_type': bin_type,
                                'explanation': explanation
                            })
            
            # If no structured items found, try to parse the whole response
            if not items_classifications:
                # Fallback: try to extract any mentioned items
                response_lower = response_text.lower()
                bin_type = "landfill"
                if "recycling" in response_lower:
                    bin_type = "recycling"
                elif "compost" in response_lower:
                    bin_type = "compost"
                
                item_name = detected_items[0]['class'] if detected_items else "item"
                items_classifications.append({
                    'item': item_name,
                    'bin_type': bin_type,
                    'explanation': response_text
                })
            
            return items_classifications
        except Exception as e:
            print(f"Error classifying from image: {e}")
            item_name = detected_items[0]['class'] if detected_items else "item"
            # Return as list for consistency
            return [{
                'bin_type': 'landfill',
                'explanation': f"Unable to classify from image. Please check local recycling guidelines.",
                'item': item_name
            }]
    
    def _rule_based_classification(self, detected_items):
        """
        Rule-based classification when API is unavailable
        Provides basic classification based on item names
        """
        if not detected_items:
            return "No items detected."
        
        classifications = []
        for item in detected_items:
            item_name = item['class'].lower()
            bin_type = "landfill"  # default
            explanation = ""
            
            # Food items -> compost (GREEN BIN)
            if any(food in item_name for food in ['apple', 'banana', 'orange', 'pizza', 'donut', 'cake', 
                                                   'sandwich', 'hot dog', 'broccoli', 'carrot']):
                bin_type = "compost"
                explanation = "This goes in the green compost bin. Food waste belongs in compost."
            
            # Containers -> recycling (BLUE BIN) if clean
            elif any(container in item_name for container in ['bottle', 'cup', 'bowl', 'can']):
                bin_type = "recycling"
                explanation = "This goes in the blue recycling bin. Clean containers are recyclable."
            
            # Paper items -> recycling (BLUE) or compost (GREEN) depending on condition
            elif 'paper' in item_name or 'napkin' in item_name:
                if 'napkin' in item_name:
                    bin_type = "compost"
                    explanation = "This goes in the green compost bin. Paper napkins with food residue belong in compost."
                else:
                    bin_type = "recycling"
                    explanation = "This goes in the blue recycling bin. Clean paper can be recycled."
            
            # Electronics -> landfill (BLACK/GREY BIN) - e-waste needs special handling
            elif any(elec in item_name for elec in ['phone', 'cell', 'remote', 'laptop', 'tv', 'monitor']):
                bin_type = "landfill"
                explanation = "This goes in the black/grey landfill bin. Electronics should be taken to e-waste recycling centers, not regular trash."
            
            # Utensils -> recycling (BLUE BIN) if metal
            elif any(utensil in item_name for utensil in ['fork', 'knife', 'spoon']):
                bin_type = "recycling"
                explanation = "This goes in the blue recycling bin. Metal utensils can be recycled if clean."
            
            # Styrofoam, plastic wrappers -> landfill (BLACK/GREY BIN)
            elif 'styrofoam' in item_name or 'wrapper' in item_name:
                bin_type = "landfill"
                explanation = "This goes in the black/grey landfill bin. Styrofoam and plastic wrappers are not recyclable."
            
            else:
                bin_type = "landfill"
                explanation = "This goes in the black/grey landfill bin. Please check local recycling guidelines for specific items."
            
            classifications.append(f"- {item['class']}: {bin_type.upper()} - {explanation}")
        
        return "\n".join(classifications)
    
    def _parse_rule_based_response(self, response_text, detected_items):
        """
        Parse rule-based classification response into the expected format
        Format: "- Item: BIN_TYPE - explanation"
        """
        items_classifications = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('-'):
                continue
            
            # Format: "- Item: BIN_TYPE - explanation"
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    item_name = parts[0].replace('-', '').strip()
                    rest = parts[1].strip()
                    
                    # Extract bin type - it's the first word after the colon, before the dash
                    bin_type = "landfill"  # default
                    rest_lower = rest.lower()
                    
                    # Check for bin type keywords - look for them as standalone words or at start
                    if rest_lower.startswith("recycling"):
                        bin_type = "recycling"
                    elif rest_lower.startswith("compost"):
                        bin_type = "compost"
                    elif rest_lower.startswith("landfill"):
                        bin_type = "landfill"
                    else:
                        # Fallback: check if bin type appears before the explanation dash
                        if ' - ' in rest:
                            before_dash = rest.split(' - ', 1)[0].strip().lower()
                            if "recycling" in before_dash:
                                bin_type = "recycling"
                            elif "compost" in before_dash:
                                bin_type = "compost"
                            elif "landfill" in before_dash:
                                bin_type = "landfill"
                    
                    # Extract explanation (everything after the dash)
                    explanation = rest
                    if ' - ' in rest:
                        explanation = rest.split(' - ', 1)[1].strip()
                    elif rest_lower.startswith(bin_type):
                        # If bin type is at start, explanation is after it
                        explanation = rest[len(bin_type):].strip()
                        if explanation.startswith('-'):
                            explanation = explanation[1:].strip()
                    
                    if item_name:
                        items_classifications.append({
                            'item': item_name,
                            'bin_type': bin_type,
                            'explanation': explanation
                        })
        
        return items_classifications
    
    def update_bin_context(self, bin_layout_result):
        """
        Store bin layout metadata so future classifications can reference the physical bins.
        """
        if not bin_layout_result:
            self.bin_context = ""
            self.bin_layout_metadata = None
            return
        
        bins = []
        if isinstance(bin_layout_result, dict):
            bins = bin_layout_result.get("bins", []) or []
        elif isinstance(bin_layout_result, list):
            bins = bin_layout_result
        
        summaries = []
        for idx, bin_info in enumerate(bins, 1):
            if not isinstance(bin_info, dict):
                continue
            label = bin_info.get("bin_label") or f"Bin {idx}"
            bin_type = bin_info.get("bin_type_guess") or "unknown"
            color = bin_info.get("bin_color") or ""
            signage = bin_info.get("signage_text") or ""
            notes = bin_info.get("additional_notes") or ""
            
            summary = f"{label}: {bin_type.upper()}"
            if color:
                summary += f" (Color: {color})"
            if signage:
                summary += f" Signage: {signage}"
            if notes:
                summary += f" Notes: {notes}"
            summaries.append(summary)
        
        self.bin_context = " | ".join(summaries)
        self.bin_layout_metadata = bin_layout_result
    
    def answer_question(self, question):
        """
        Answer user questions about recycling/waste disposal
        
        Args:
            question: User's question
            
        Returns:
            Answer from Gemini
        """
        prompt = f"""You are a helpful recycling and waste disposal assistant. 
Answer the following question about waste disposal, recycling, or composting:
{question}

Provide a clear, concise, and helpful answer."""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            # Store for logging
            self.last_raw_response = response_text
            return response_text
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error: {str(e)}"
            self.last_raw_response = error_msg
            return error_msg
