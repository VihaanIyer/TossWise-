"""
Gemini API Integration for Trash Classification
Determines which bin (recycling, compost, landfill) an item should go into
"""

import google.generativeai as genai
import os
import re
import json
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
        
        # Load bin layout metadata from JSON
        self.bin_layout = self._load_bin_layout()
        
        # Build system prompt based on actual bin layout
        self.system_prompt = self._build_system_prompt()
        
        # Initialize bin_context for backward compatibility
        self.bin_context = ""
        self.bin_layout_metadata = None
    
    def _load_bin_layout(self):
        """
        Load bin layout metadata from JSON file
        """
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'bin_layout_metadata.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    return data.get('bins', [])
            else:
                print(f"⚠️ bin_layout_metadata.json not found at {json_path}, using default bins")
                return []
        except Exception as e:
            print(f"⚠️ Error loading bin layout: {e}, using default bins")
            return []
    
    def _build_system_prompt(self):
        """
        Build system prompt based on actual bin layout from JSON
        """
        base_prompt = """You are a waste classification expert. Analyze the IMAGE carefully to see the ACTUAL material, condition, and contamination level of each item.

IMPORTANT: ONLY identify items that are TRASH/WASTE meant for disposal. DO NOT classify personal items like:
- Phones, electronics, devices
- Hats, caps, beanies, clothing
- Wallets, keys, personal belongings
- Bags, backpacks, purses
- Any item that is clearly being held/used by a person and not trash

CRITICAL: Look at the IMAGE to determine if items are CLEAN or CONTAMINATED. Analyze what you actually see.

"""
        
        if not self.bin_layout:
            # Default bins if JSON not loaded
            return base_prompt + """BIN SYSTEM:
- RECYCLING (blue): Clean recyclable materials
  * Clean plastic items (forks, bottles, containers)
  * Clean paper/cardboard
  * Metal items (cans, utensils if clean)
  * Glass
- COMPOST (green): Organic food waste (fruit peels, food scraps, coffee grounds)
- LANDFILL (black/grey): Contaminated items, non-recyclables, greasy items"""
        
        # Build bin system from JSON
        bin_descriptions = []
        for bin_info in self.bin_layout:
            bin_type = bin_info.get('type', '').upper()
            color = bin_info.get('color', '')
            sign = bin_info.get('sign', '')
            label = bin_info.get('label', '')
            
            # Format bin description
            bin_desc = f"- {bin_type} ({color}): {sign}"
            if label:
                bin_desc += f" [Label: {label}]"
            bin_descriptions.append(bin_desc)
        
        bin_system = "BIN SYSTEM:\n" + "\n".join(bin_descriptions)
        return base_prompt + bin_system
    
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
    
    def detect_bags_in_image(self, image):
        """
        Detect trash bags in the image using Gemini Vision
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            List of detected bags with their positions/descriptions
        """
        # Convert numpy array to PIL Image if needed
        if not isinstance(image, Image.Image):
            if hasattr(image, 'shape'):  # numpy array
                import numpy as np
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = image[:, :, ::-1]  # BGR to RGB
                    image = Image.fromarray(image_rgb)
                else:
                    image = Image.fromarray(image)
        
        prompt = """EXAMINE THIS IMAGE. Look for TRASH BAGS or GARBAGE BAGS that are ready to be disposed of.

CRITICAL: ONLY identify bags that are:
- Trash bags, garbage bags, or bags filled with waste
- Bags that are clearly meant for disposal (not personal bags like backpacks, purses, shopping bags being used)
- Bags that appear to contain trash/waste

IGNORE:
- Personal bags (backpacks, purses, handbags, shopping bags being carried)
- Empty bags
- Bags that are clearly not trash

If you see trash bags, count how many distinct bags you see and describe their position (left, center, right, etc.).

Respond in this format:
- If bags found: "BAG_1: [position/description], BAG_2: [position/description], ..."
- If no bags found: "NO_BAGS"

Examples:
- "BAG_1: left side, BAG_2: right side"
- "BAG_1: center"
- "NO_BAGS"

Respond:"""
        
        try:
            if self.supports_vision:
                response = self.model.generate_content([prompt, image])
                response_text = response.text.strip()
                self.last_raw_response = response_text
                
                # Parse response
                if "NO_BAGS" in response_text.upper() or "no bag" in response_text.lower():
                    return []
                
                # Extract bag information
                bags = []
                bag_pattern = r'BAG_(\d+):\s*(.+)'
                matches = re.findall(bag_pattern, response_text, re.IGNORECASE)
                
                for match in matches:
                    bag_num = int(match[0])
                    description = match[1].strip()
                    bags.append({
                        'number': bag_num,
                        'description': description,
                        'position': description
                    })
                
                return bags
            else:
                return []
        except Exception as e:
            print(f"Error detecting bags: {e}")
            return []
    
    def classify_bag_contents(self, bag_description):
        """
        Classify which bin a bag should go into based on user's description of contents
        
        Args:
            bag_description: User's description of what's mostly in the bag
            
        Returns:
            Dictionary with bin type, bin_name, bin_color, and explanation
        """
        # Build bin descriptions from JSON
        bin_descriptions = []
        if self.bin_layout:
            for bin_info in self.bin_layout:
                bin_type = bin_info.get('type', '').upper()
                color = bin_info.get('color', '')
                sign = bin_info.get('sign', '')
                bin_desc = f"- {bin_type} ({color}): {sign}"
                bin_descriptions.append(bin_desc)
        else:
            bin_descriptions = [
                "- RECYCLING (blue): Clean recyclable materials",
                "- COMPOST (green): Organic food waste",
                "- LANDFILL (black/grey): Contaminated or non-recyclable items"
            ]
        
        prompt = f"""A user has a bag of trash and described what's mostly in it as: "{bag_description}"

Based on this description, determine which bin this bag should go into.

AVAILABLE BINS:
{chr(10).join(bin_descriptions)}

Consider:
- What is the PRIMARY/MOST COMMON material in the bag?
- If it's mixed, what is the majority?
- If it contains food waste, it likely goes to compost
- If it contains recyclables (paper, plastic, metal), it likely goes to recycling
- If it contains contaminated items or non-recyclables, it likely goes to landfill

Respond in this exact format:
[bin_type] bin usually [color]

Examples:
- "recycle bin usually blue" (if mostly recyclables)
- "compost bin usually green" (if mostly food waste)
- "landfill bin usually black or grey" (if mostly non-recyclables or contaminated)

Your answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            self.last_raw_response = response_text
            
            # Parse response
            response_lower = response_text.lower()
            
            # Extract bin type
            bin_type = "landfill"  # default
            if "recycle" in response_lower:
                bin_type = "recycling"
            elif "compost" in response_lower:
                bin_type = "compost"
            elif "landfill" in response_lower or "trash" in response_lower:
                bin_type = "landfill"
            
            # Extract color
            bin_color = self._get_bin_color_for_type(bin_type)
            if "blue" in response_lower:
                bin_color = "blue"
            elif "green" in response_lower:
                bin_color = "green"
            elif "black" in response_lower or "grey" in response_lower or "gray" in response_lower:
                bin_color = "black or grey"
            
            bin_name = self._get_bin_name_for_type(bin_type)
            
            return {
                'bin_type': bin_type,
                'bin_name': bin_name,
                'bin_color': bin_color,
                'explanation': f"Based on the contents ({bag_description}), this bag goes into the {bin_color} {bin_name}."
            }
        except Exception as e:
            print(f"Error classifying bag contents: {e}")
            bin_name = self._get_bin_name_for_type('landfill')
            return {
                'bin_type': 'landfill',
                'bin_name': bin_name,
                'bin_color': self._get_bin_color_for_type('landfill'),
                'explanation': f"Unable to classify. Please check local recycling guidelines."
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
        
        # Build bin descriptions from JSON
        bin_descriptions = []
        if self.bin_layout:
            for bin_info in self.bin_layout:
                bin_type = bin_info.get('type', '').upper()
                color = bin_info.get('color', '')
                sign = bin_info.get('sign', '')
                label = bin_info.get('label', '')
                bin_desc = f"- {bin_type} ({color}): {sign}"
                if label:
                    bin_desc += f" [Label: {label}]"
                bin_descriptions.append(bin_desc)
        else:
            # Fallback descriptions
            bin_descriptions = [
                "- RECYCLING (blue): Clean recyclable materials",
                "- COMPOST (green): Organic food waste",
                "- LANDFILL (black/grey): Contaminated or non-recyclable items"
            ]
        
        prompt = f"""EXAMINE THIS IMAGE VERY CAREFULLY. Look at each item's ACTUAL appearance, material, condition, and CONTEXT. Be precise and accurate.

CRITICAL: ONLY identify items that are TRASH/WASTE meant for disposal. IGNORE personal items like:
- Phones, electronics, devices, chargers
- Hats, caps, beanies, clothing, accessories, GLOVES (work gloves, medical gloves, etc.)
- Wallets, keys, personal belongings
- Bags, backpacks, purses
- Glasses, watches, jewelry
- Any item clearly being held/used by a person that is NOT trash (like gloves on hands)

ONLY classify items that are clearly waste/trash ready to be thrown away (food scraps, empty containers, used utensils, wrappers, etc.).

ACCURACY CHECKLIST:
- Is this item actually trash/waste? (NOT a personal item being used)
- Can you clearly see it's meant for disposal? (NOT something the person is wearing/using)
- Is it clearly visible in the image? (NOT a misidentification)
- Look at the actual material and condition - don't guess based on color alone

{items_text}

AVAILABLE BINS (use these exact rules from the facility):
{chr(10).join(bin_descriptions)}

CRITICAL - Match items to bins based on the rules above:
- Read each bin's signage/rules carefully
- Match items to the correct bin based on what the bin accepts
- Check if items are CLEAN or CONTAMINATED from the image
- Use the specific bin rules (e.g., landfill: "nothing except styrofoam and PPE", compost: "all food, napkins, compostables", recycle: "all paper, metal, plastic, wrappers")
- For paper items, distinguish between mixed paper (newspapers, magazines, folders) and white paper (printer paper, notebook paper)

RESPONSE FORMAT:
- [item name] goes into [bin type] bin usually [color]

Examples:
- plastic fork (if clean in image) → "plastic fork goes into recycle bin usually blue"
- plastic fork (if dirty in image) → "dirty plastic fork goes into landfill usually black"
- pizza → "pizza goes into compost usually green"
- clean paper plate → "paper plate goes into recycle usually blue"
- greasy paper plate → "greasy paper plate goes into landfill usually black"

IMPORTANT: 
- Look at the ACTUAL IMAGE. See if items are clean or dirty. Don't guess - use what you see.
- ONLY list items that are clearly TRASH/WASTE ready for disposal.
- DO NOT list personal items (phones, caps, beanies, wallets, keys, bags, etc.).
- If you see a person holding something, only classify it if it's clearly trash (like a used napkin, empty bottle, food wrapper), NOT if it's a personal item.

List all trash items:"""
        
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
                    
                    # Use text-only classification - fast and natural
                    # Build bin descriptions from JSON
                    bin_desc_text = ""
                    if self.bin_layout:
                        bin_list = []
                        for bin_info in self.bin_layout:
                            bin_type = bin_info.get('type', '').upper()
                            color = bin_info.get('color', '')
                            sign = bin_info.get('sign', '')
                            bin_list.append(f"- {bin_type} ({color}): {sign}")
                        bin_desc_text = "\n".join(bin_list)
                    else:
                        bin_desc_text = """- RECYCLING = blue bin
- COMPOST = green bin  
- LANDFILL = black or grey bin"""
                    
                    text_prompt = f"""Quickly classify these items. Format: [item] goes into [bin_type] bin usually [color]

CRITICAL: ONLY classify items that are TRASH/WASTE. IGNORE personal items (phones, caps, beanies, wallets, keys, bags, etc.).

AVAILABLE BINS (use these exact rules):
{bin_desc_text}

Items: {items_text}

Respond fast in format: [item] goes into [bin_type] bin usually [color]"""
                    
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
                # Text-only model - fast and natural
                print("⚠️ Model doesn't support vision, using text-only classification")
                # Build bin descriptions from JSON
                bin_desc_text = ""
                if self.bin_layout:
                    bin_list = []
                    for bin_info in self.bin_layout:
                        bin_type = bin_info.get('type', '').upper()
                        color = bin_info.get('color', '')
                        sign = bin_info.get('sign', '')
                        bin_list.append(f"- {bin_type} ({color}): {sign}")
                    bin_desc_text = "\n".join(bin_list)
                else:
                    bin_desc_text = """- RECYCLING = blue bin
- COMPOST = green bin
- LANDFILL = black or grey bin"""
                
                text_prompt = f"""Quickly classify these items. Format: [item] goes into [bin_type] bin usually [color]

CRITICAL: ONLY classify items that are TRASH/WASTE. IGNORE personal items (phones, caps, beanies, wallets, keys, bags, etc.).

AVAILABLE BINS (use these exact rules):
{bin_desc_text}

Items: {items_text}

Respond fast in format: [item] goes into [bin_type] bin usually [color]"""
                
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
            
            # Parse response - look for natural format: "item goes into bin_type bin usually color"
            items_classifications = []
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Accept lines with or without dash prefix, or lines with "goes into/goes in"
                has_dash = line.startswith('-')
                has_goes_into = 'goes into' in line.lower() or 'goes in' in line.lower()
                has_colon = ':' in line
                
                if not (has_dash or has_goes_into or has_colon):
                    continue
                
                # Remove leading dash if present
                line = line.lstrip('-').strip()
                
                # Parse natural format: "item goes into bin_type bin usually color"
                # Or: "item: bin_type - explanation"
                line_lower = line.lower()
                
                # Extract bin type first
                bin_type = "landfill"  # default
                if "recycle" in line_lower or "recycling" in line_lower:
                    bin_type = "recycling"
                elif "compost" in line_lower:
                    bin_type = "compost"
                elif "landfill" in line_lower or "trash" in line_lower:
                    bin_type = "landfill"
                
                # Extract item name and create natural explanation
                item_name = ""
                explanation = ""
                
                # Pattern 1: "item goes into bin_type bin usually color"
                if "goes into" in line_lower or "goes in" in line_lower:
                    # Split on "goes into" or "goes in"
                    parts = re.split(r'goes (into|in)', line_lower, 1)
                    if len(parts) >= 2:
                        item_name = parts[0].strip()
                        rest = parts[-1].strip()
                        
                        # Extract color if mentioned
                        color = ""
                        if "blue" in rest:
                            color = "blue"
                        elif "green" in rest:
                            color = "green"
                        elif "black" in rest or "grey" in rest or "gray" in rest:
                            color = "black or grey"
                        
                        # Create natural explanation
                        if color:
                            explanation = f"This goes in the {color} {bin_type} bin."
                        else:
                            explanation = f"This goes in the {bin_type} bin."
                
                # Pattern 2: "item: bin_type - explanation" (fallback)
                elif ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        item_name = parts[0].strip()
                        rest = parts[1].strip()
                        
                        # Extract color
                        color = ""
                        rest_lower = rest.lower()
                        if "blue" in rest_lower:
                            color = "blue"
                        elif "green" in rest_lower:
                            color = "green"
                        elif "black" in rest_lower or "grey" in rest_lower or "gray" in rest_lower:
                            color = "black or grey"
                        
                        # Create explanation
                        if color:
                            explanation = f"This goes in the {color} {bin_type} bin."
                        else:
                            explanation = rest if rest else f"This goes in the {bin_type} bin."
                
                # If we found an item, add it
                if item_name:
                    # Get bin name/label from bin_layout
                    bin_name = self._get_bin_name_for_type(bin_type)
                    items_classifications.append({
                        'item': item_name,
                        'bin_type': bin_type,
                        'bin_name': bin_name,
                        'bin_color': self._get_bin_color_for_type(bin_type),
                        'explanation': explanation if explanation else f"This goes in the {bin_type} bin."
                    })
            
            # If no structured items found, try to extract from whole response
            if not items_classifications:
                response_lower = response_text.lower()
                bin_type = "landfill"
                if "recycle" in response_lower or "recycling" in response_lower:
                    bin_type = "recycling"
                elif "compost" in response_lower:
                    bin_type = "compost"
                
                # Try to extract item name from detected items or response
                item_name = detected_items[0]['class'] if detected_items else "item"
                if detected_items:
                    item_name = detected_items[0]['class']
                else:
                    # Try to find item name in response
                    words = response_text.split()
                    if words:
                        item_name = words[0]
                
                # Extract color
                color = ""
                if "blue" in response_lower:
                    color = "blue"
                elif "green" in response_lower:
                    color = "green"
                elif "black" in response_lower or "grey" in response_lower or "gray" in response_lower:
                    color = "black or grey"
                
                explanation = f"This goes in the {color} {bin_type} bin." if color else f"This goes in the {bin_type} bin."
                
                # Get bin name/label from bin_layout
                bin_name = self._get_bin_name_for_type(bin_type)
                items_classifications.append({
                    'item': item_name,
                    'bin_type': bin_type,
                    'bin_name': bin_name,
                    'bin_color': color if color else self._get_bin_color_for_type(bin_type),
                    'explanation': explanation
                })
            
            return items_classifications
        except Exception as e:
            print(f"Error classifying from image: {e}")
            item_name = detected_items[0]['class'] if detected_items else "item"
            # Return as list for consistency
            bin_name = self._get_bin_name_for_type('landfill')
            return [{
                'bin_type': 'landfill',
                'bin_name': bin_name,
                'bin_color': self._get_bin_color_for_type('landfill'),
                'explanation': f"Unable to classify from image. Please check local recycling guidelines.",
                'item': item_name
            }]
    
    def _get_bin_name_for_type(self, bin_type):
        """
        Get the bin name/label for a given bin type from the JSON layout
        """
        if not self.bin_layout:
            # Fallback to bin_type if no layout
            return bin_type
        
        bin_type_lower = bin_type.lower()
        for bin_info in self.bin_layout:
            if bin_info.get('type', '').lower() == bin_type_lower:
                # Return a human-readable name based on label
                label = bin_info.get('label', '')
                if label:
                    # Convert "blue_recycle" to "recycle bin" or "green_compost" to "compost bin"
                    if '_' in label:
                        parts = label.split('_')
                        if len(parts) > 1:
                            return f"{parts[-1]} bin"
                    return f"{label} bin"
                # Fallback to type
                return f"{bin_type} bin"
        
        # Default fallback
        return f"{bin_type} bin"
    
    def _get_bin_color_for_type(self, bin_type):
        """
        Get the bin color for a given bin type from the JSON layout
        """
        if not self.bin_layout:
            # Fallback colors
            if bin_type.lower() == 'recycling':
                return 'blue'
            elif bin_type.lower() == 'compost':
                return 'green'
            elif bin_type.lower() == 'landfill':
                return 'black or grey'
            return bin_type
        
        bin_type_lower = bin_type.lower()
        for bin_info in self.bin_layout:
            if bin_info.get('type', '').lower() == bin_type_lower:
                return bin_info.get('color', 'blue')
        
        # Default fallback
        return 'blue'
    
    def _rule_based_classification(self, detected_items):
        """
        Intelligent rule-based classification when API is unavailable
        Uses material-based logic rather than hardcoded item lists
        """
        if not detected_items:
            return "No items detected."
        
        classifications = []
        for item in detected_items:
            item_name = item['class'].lower()
            bin_type = "landfill"  # default
            explanation = ""
            
            # Intelligent material-based classification
            # Organic/Food materials -> COMPOST
            organic_keywords = ['apple', 'banana', 'orange', 'pizza', 'donut', 'cake', 
                              'sandwich', 'hot dog', 'broccoli', 'carrot', 'food', 'fruit', 
                              'vegetable', 'bread', 'meat', 'cheese']
            if any(keyword in item_name for keyword in organic_keywords):
                bin_type = "compost"
                explanation = "This goes in the green compost bin. Organic food waste decomposes naturally and belongs in compost."
            
            # Container materials -> RECYCLING (if likely clean)
            # Check for container keywords AND material indicators
            container_keywords = ['bottle', 'cup', 'bowl', 'can', 'jar', 'container']
            material_keywords = ['plastic', 'metal', 'aluminum', 'glass', 'tin']
            is_container = any(keyword in item_name for keyword in container_keywords)
            has_recyclable_material = any(keyword in item_name for keyword in material_keywords)
            
            if is_container or has_recyclable_material:
                # Check for contamination indicators
                contaminated_keywords = ['dirty', 'greasy', 'soiled', 'contaminated']
                is_contaminated = any(keyword in item_name for keyword in contaminated_keywords)
                
                if is_contaminated:
                    bin_type = "landfill"
                    explanation = "This goes in the black or grey landfill bin. While the material is recyclable, contamination makes it unsuitable for recycling."
                else:
                    bin_type = "recycling"
                    explanation = "This goes in the blue recycling bin. Clean containers made of recyclable materials can be processed into new products."
            
            # Paper products -> context-dependent
            elif 'paper' in item_name or 'cardboard' in item_name or 'napkin' in item_name:
                # Check for contamination
                if 'napkin' in item_name or 'tissue' in item_name:
                    bin_type = "compost"
                    explanation = "This goes in the green compost bin. Paper napkins/tissues with potential food residue are better suited for compost."
                else:
                    bin_type = "recycling"
                    explanation = "This goes in the blue recycling bin. Clean paper and cardboard are highly recyclable materials."
            
            # Electronics -> LANDFILL (e-waste needs special handling)
            elif any(elec in item_name for elec in ['phone', 'cell', 'remote', 'laptop', 'tv', 'monitor', 'electronic']):
                bin_type = "landfill"
                explanation = "This goes in the black or grey landfill bin. Electronics contain hazardous materials and should be taken to e-waste recycling centers, not regular trash."
            
            # Utensils/forks - need to check material type
            elif any(utensil in item_name for utensil in ['fork', 'knife', 'spoon', 'utensil']):
                # Plastic utensils: recyclable if clean (default assumption)
                if 'plastic' in item_name:
                    bin_type = "recycling"
                    explanation = "This goes in the blue recycling bin. Clean plastic utensils are recyclable."
                # Metal utensils: recyclable if clean
                elif 'metal' in item_name or any(m in item_name for m in ['steel', 'aluminum']):
                    bin_type = "recycling"
                    explanation = "This goes in the blue recycling bin. Metal utensils can be recycled if clean."
                else:
                    # Default: assume clean plastic, recyclable
                    bin_type = "recycling"
                    explanation = "This goes in the blue recycling bin. Clean utensils are typically recyclable."
            
            # Metal items (not utensils) -> RECYCLING
            elif 'metal' in item_name and 'utensil' not in item_name and 'fork' not in item_name:
                bin_type = "recycling"
                explanation = "This goes in the blue recycling bin. Metal items can be recycled if clean."
            
            # Non-recyclable plastics -> LANDFILL
            elif any(plastic in item_name for plastic in ['styrofoam', 'wrapper', 'bag', 'film', 'wrap']):
                bin_type = "landfill"
                explanation = "This goes in the black or grey landfill bin. These types of plastics are typically not accepted in curbside recycling programs."
            
            # Glass -> RECYCLING
            elif 'glass' in item_name:
                bin_type = "recycling"
                explanation = "This goes in the blue recycling bin. Glass is infinitely recyclable and should be recycled when clean."
            
            # Default: use intelligent fallback
            else:
                # Try to infer from item name characteristics
                if any(word in item_name for word in ['plastic', 'metal', 'aluminum']):
                    bin_type = "recycling"
                    explanation = "This goes in the blue recycling bin. The material appears recyclable, but check local guidelines for specific item types."
                elif any(word in item_name for word in ['food', 'organic', 'biodegradable']):
                    bin_type = "compost"
                    explanation = "This goes in the green compost bin. Organic materials decompose and belong in compost."
                else:
                    bin_type = "landfill"
                    explanation = "This goes in the black or grey landfill bin. When in doubt, check local recycling guidelines for proper disposal."
            
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
    
    def answer_question(self, question, last_classifications=None):
        """
        Answer user questions about recycling/waste disposal
        
        Args:
            question: User's question
            last_classifications: List of recent classifications (for "why" questions)
            
        Returns:
            Answer from Gemini or None if question is not relevant
        """
        question_lower = question.lower()
        
        # Check if question is relevant to trash/recycling/waste disposal
        relevant_keywords = [
            'trash', 'recycle', 'recycling', 'compost', 'landfill', 'garbage', 'waste',
            'bin', 'disposal', 'throw', 'away', 'why', 'how', 'what', 'where',
            'plastic', 'paper', 'metal', 'fork', 'bottle', 'container', 'plate',
            'repeat', 'again', 'say', 'explain', 'clarify'
        ]
        
        # Check if question is clearly NOT relevant
        irrelevant_keywords = [
            'eiffel tower', 'tall', 'height', 'weather', 'time', 'date', 'capital',
            'president', 'sports', 'movie', 'music', 'recipe', 'cooking'
        ]
        
        # Check for irrelevant questions
        if any(keyword in question_lower for keyword in irrelevant_keywords):
            # Check if it's actually about trash (e.g., "how tall is the recycling bin")
            if not any(relevant in question_lower for relevant in ['bin', 'trash', 'recycle', 'waste']):
                return None  # Not relevant
        
        # Check if question is relevant
        if not any(keyword in question_lower for keyword in relevant_keywords):
            return None  # Not relevant
        
        # Handle "repeat" or "say again" requests
        if any(phrase in question_lower for phrase in ['repeat', 'say again', 'what did you say', 'can you repeat']):
            if last_classifications:
                # Reconstruct what was said
                responses = []
                for item in last_classifications:
                    bin_name = self._get_bin_name_for_type(item.get('bin_type', 'bin'))
                    bin_color = item.get('bin_color', self._get_bin_color_for_type(item.get('bin_type', 'bin')))
                    responses.append(f"{item.get('item', 'item')} goes into {bin_name} usually {bin_color}")
                return ". ".join(responses)
            else:
                return "I don't have anything to repeat. Please hold an item in front of the camera first."
        
        # Build context from last classifications for "why" questions
        context = ""
        if last_classifications and any(word in question_lower for word in ['why', 'explain', 'reason']):
            context_items = []
            for item in last_classifications:
                item_name = item.get('item', 'item')
                bin_type = item.get('bin_type', 'bin')
                bin_name = self._get_bin_name_for_type(bin_type)
                explanation = item.get('explanation', '')
                context_items.append(f"{item_name} was classified as {bin_name}. {explanation}")
            if context_items:
                context = f"\n\nRecent classifications for context:\n" + "\n".join(context_items)
        
        # Build prompt with system context
        system_context = f"""You are a smart garbage bin assistant software. Your purpose is to help users correctly dispose of waste items.

AVAILABLE BINS:
"""
        if self.bin_layout:
            for bin_info in self.bin_layout:
                bin_type = bin_info.get('type', '').upper()
                color = bin_info.get('color', '')
                sign = bin_info.get('sign', '')
                system_context += f"- {bin_type} ({color}): {sign}\n"
        else:
            system_context += "- RECYCLING (blue): Clean recyclable materials\n"
            system_context += "- COMPOST (green): Organic food waste\n"
            system_context += "- LANDFILL (black/grey): Contaminated or non-recyclable items\n"
        
        prompt = f"""{system_context}

User Question: {question}{context}

Instructions:
- Answer ONLY questions related to waste disposal, recycling, composting, or the items you just classified
- If the question is about a recent classification, explain WHY based on the item's condition, material, and contamination level
- Be helpful, clear, and concise
- If asked "why" about a classification, explain the reasoning based on the bin rules and item condition

Answer:"""
        
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
