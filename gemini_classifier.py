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
    def __init__(self, language='english'):
        """
        Initialize Gemini API client and pick a fast, vision-capable model.
        IMPORTANT: Create this ONCE and reuse it.
        Do not re-create TrashClassifier on every frame.
        
        Args:
            language: Language to use for prompts and responses ('english' or 'hungarian')
        """
        self.language = language.lower()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Ordered by preference: fast, vision-capable models first, then text fallback
        preferred_models = [
            'gemini-2.0-flash',
            'gemini-2.5-flash',
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro-vision',
            'gemini-pro',  # text-only fallback
        ]
        
        self.model = None
        self.model_name = None
        self.supports_vision = False
        
        for model_name in preferred_models:
            for variant in (model_name, f"models/{model_name}"):
                try:
                    self.model = genai.GenerativeModel(variant)
                    self.model_name = model_name
                    # Basic heuristic for vision support
                    if any(token in model_name.lower() for token in ['vision', 'flash', '1.5', '2.0', '2.5']):
                        self.supports_vision = True
                    print(f"Loaded Gemini model: {self.model_name} (vision support: {self.supports_vision})")
                    break
                except Exception:
                    continue
            if self.model is not None:
                break
        
        if self.model is None:
            raise ValueError("Could not initialize any Gemini model. Check your API key and model availability.")
        
        # Shared config for faster responses - optimized for <1s response time
        self.generation_config_fast = {
            "temperature": 0.0,  # Zero temperature for fastest, most deterministic responses
            "max_output_tokens": 96,  # Further reduced for faster generation
            "top_p": 0.95,  # Nucleus sampling for faster decoding
        }
        
        # Initialize for logging
        self.last_raw_response = ""
        
        # Load bin layout metadata from JSON (with location support)
        self.bin_layout = self._load_bin_layout()
        
        # Build system prompt based on actual bin layout
        self.system_prompt = self._build_system_prompt()
        
        # Initialize bin_context for backward compatibility
        self.bin_context = ""
        self.bin_layout_metadata = None
    
    # ---------------------- helpers ---------------------- #
    
    @staticmethod
    def _to_pil_and_downscale(image, max_dim: int = 384) -> Image.Image:
        """
        Convert numpy/OpenCV or PIL image to PIL and downscale to max_dim.
        Optimized to 384px for faster API calls while maintaining accuracy.
        Uses smart preprocessing for better quality at smaller size.
        """
        if not isinstance(image, Image.Image):
            if hasattr(image, "shape"):
                import numpy as np  # local import
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = image[:, :, ::-1]  # BGR to RGB
                    image = Image.fromarray(image_rgb)
                else:
                    image = Image.fromarray(image)
            else:
                raise TypeError("Unsupported image type for conversion to PIL.")
        
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            # Use LANCZOS resampling for better quality at smaller size
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance image quality for better accuracy
        # Convert to RGB if needed (some images might be RGBA)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def _load_bin_layout(self):
        """
        Load bin layout metadata from JSON file
        Supports location-specific files via BIN_LOCATION environment variable
        """
        try:
            # Try location-specific file first
            location = os.getenv('BIN_LOCATION', None)
            if location:
                location_file = os.path.join(os.path.dirname(__file__), f'bin_layout_{location}.json')
                if os.path.exists(location_file):
                    with open(location_file, 'r') as f:
                        data = json.load(f)
                        bins = data.get('bins', [])
                        if bins:
                            print(f"✅ Loaded {len(bins)} bins from location-specific file: {location_file}")
                            return bins
            
            # Fallback to main metadata file
            json_path = os.path.join(os.path.dirname(__file__), 'bin_layout_metadata.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    bins = data.get('bins', [])
                    if bins:
                        print(f"✅ Loaded {len(bins)} bins from bin_layout_metadata.json")
                        return bins
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
            pos = bin_info.get('pos', '')
            
            # Convert position to readable format
            if self.language == 'hungarian':
                if 'left' in pos.lower() or pos.lower() == 'leftmost':
                    pos_text = 'bal oldalon'
                elif 'center' in pos.lower() or 'middle' in pos.lower():
                    pos_text = 'középen'
                elif 'right' in pos.lower() or pos.lower() == 'rightmost':
                    pos_text = 'jobb oldalon'
                else:
                    pos_text = pos
            else:
                if 'left' in pos.lower() or pos.lower() == 'leftmost':
                    pos_text = 'on the left'
                elif 'center' in pos.lower() or 'middle' in pos.lower():
                    pos_text = 'in the middle'
                elif 'right' in pos.lower() or pos.lower() == 'rightmost':
                    pos_text = 'on the right'
                else:
                    pos_text = pos
            
            # Format bin description
            bin_desc = f"- {bin_type} ({color} bin {pos_text}): {sign}"
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
                response = self.model.generate_content(
                    [prompt, image],
                    generation_config=self.generation_config_fast,
                )
            else:
                # Text-only classification
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config_fast,
                )
            
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
                response = self.model.generate_content(
                    [prompt, image],
                    generation_config=self.generation_config_fast,
                )
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
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config_fast,
            )
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
            bin_position = self._get_bin_position_for_type(bin_type)
            
            return {
                'bin_type': bin_type,
                'bin_name': bin_name,
                'bin_color': bin_color,
                'bin_position': bin_position,
                'explanation': f"Based on the contents ({bag_description}), this bag goes into the {bin_color} {bin_name}."
            }
        except Exception as e:
            print(f"Error classifying bag contents: {e}")
            bin_name = self._get_bin_name_for_type('landfill')
            bin_position = self._get_bin_position_for_type('landfill')
            return {
                'bin_type': 'landfill',
                'bin_name': bin_name,
                'bin_color': self._get_bin_color_for_type('landfill'),
                'bin_position': bin_position,
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
        # Convert and downscale once
        try:
            image = self._to_pil_and_downscale(image)
        except Exception as e:
            print(f"Image conversion failed, continuing without resize: {e}")
        
        # Build items text in appropriate language
        items_text = ""
        if detected_items:
            items_list = [item['class'] for item in detected_items]
            if self.language == 'hungarian':
                items_text = f"Megjegyzés: A YOLOv8 objektumfelismerés ezeket a tárgyakat javasolta: {', '.join(items_list)}. Azonban nézd meg a VALÓDI KÉPET és azonosítsd, hogy milyen szemét/hulladék tárgyak LÁTHATÓK VALÓJÁBAN."
            else:
                items_text = f"Note: YOLOv8 object detection suggested these items might be present: {', '.join(items_list)}. However, please look at the ACTUAL IMAGE and identify what trash/waste items are REALLY visible."
        
        # Build bin descriptions from JSON in appropriate language
        bin_descriptions = []
        if self.bin_layout:
            print(f"📋 Using {len(self.bin_layout)} configured bins for classification:")
            for bin_info in self.bin_layout:
                bin_type = bin_info.get('type', '').upper()
                color = bin_info.get('color', '')
                sign = bin_info.get('sign', '')
                label = bin_info.get('label', '')
                
                if self.language == 'hungarian':
                    # Translate bin types to Hungarian
                    bin_translations = {
                        'RECYCLING': 'ÚJRAHASZNOSÍTÁS',
                        'COMPOST': 'KOMPOSZT',
                        'LANDFILL': 'SZEMÉTLERAKÓ'
                    }
                    bin_type_display = bin_translations.get(bin_type, bin_type)
                    # Translate colors to Hungarian
                    color_translations = {
                        'blue': 'kék',
                        'green': 'zöld',
                        'black': 'fekete',
                        'grey': 'szürke',
                        'gray': 'szürke',
                        'gray and black': 'szürke és fekete'
                    }
                    color_display = color_translations.get(color.lower(), color)
                    bin_desc = f"- {bin_type_display} ({color_display}): {sign}"
                else:
                    bin_desc = f"- {bin_type} ({color}): {sign}"
                
                if label:
                    bin_desc += f" [Label: {label}]"
                bin_descriptions.append(bin_desc)
                print(f"  ✓ {bin_type} bin - Color: {color}, Sign: {sign}")
        else:
            # Fallback descriptions
            print("⚠️ No bin layout configured, using default bins")
            if self.language == 'hungarian':
                bin_descriptions = [
                    "- ÚJRAHASZNOSÍTÁS (kék): Tiszta újrahasznosítható anyagok",
                    "- KOMPOSZT (zöld): Szerves élelmiszer hulladék",
                    "- SZEMÉTLERAKÓ (fekete/szürke): Szennyezett vagy nem újrahasznosítható tárgyak"
                ]
            else:
                bin_descriptions = [
                    "- RECYCLING (blue): Clean recyclable materials",
                    "- COMPOST (green): Organic food waste",
                    "- LANDFILL (black/grey): Contaminated or non-recyclable items"
                ]
        
        # Build language-specific prompt
        if self.language == 'hungarian':
            prompt = f"""Elemezd a képet. Csak SZEMÉT/HULLADÉK azonosítása. FIGYELMEN KÍVÜL: telefonok, sapkák, kesztyűk, pénztárcák, táskák, személyes tárgyak.

{items_text}

KUKÁK:
{chr(10).join(bin_descriptions)}

SZABÁLYOK:
- Vizuálisan ellenőrizd minden tárgy ÁLLAPOTÁT (tiszta vs szennyezett)
- Tiszta újrahasznosítható → újrahasznosítás, Étel/szerves → komposzt, Szennyezett/nem újrahasznosítható → szemétlerakó
- Legyél pontos: ellenőrizd az anyagot és szennyezettségi szintet

FONTOS: VÁLASZOLJ KIZÁRÓLAG MAGYARUL!

FORMÁTUM: [tárgy] megy a [color] [bin_type] kukába [position]

FONTOS: Ha látsz szemét tárgyakat, sorold fel őket egy sorban ebben a formátumban.
Ha NINCS szemét tárgy, válaszolj: "Nincs szemét tárgy"

Példák:
- tiszta műanyag villa → "műanyag villa megy a kék újrahasznosítás kukába bal oldalon"
- szennyezett műanyag villa → "szennyezett műanyag villa megy a fekete szemétlerakó kukába jobb oldalon"
- pizza → "pizza megy a zöld komposzt kukába középen"

Sorold fel az összes talált szemét tárgyat (egy sorban):"""
        else:  # English (default)
            prompt = f"""Analyze image. Identify ONLY trash/waste items. IGNORE: phones, hats, gloves, wallets, bags, personal items.

{items_text}

BINS:
{chr(10).join(bin_descriptions)}

RULES:
- Visually inspect each item's CONDITION (clean vs contaminated)
- Clean recyclables → recycling, Food/organic → compost, Contaminated/non-recyclable → landfill
- Be precise: check actual material and contamination level

FORMAT: [item] goes into [color] [bin_type] bin [position]

IMPORTANT: If you see trash items, list them one per line in this exact format.
If you see NO trash items, respond with: "No trash items found"

Examples:
- clean plastic fork → "plastic fork goes into blue recycling bin on the left"
- dirty plastic fork → "dirty plastic fork goes into black landfill bin on the right"
- pizza → "pizza goes into green compost bin in the middle"

List all trash items found (one per line):"""
        
        # Log the prompt being sent (first 500 chars for brevity)
        prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        print(f"📝 Prompt sent to LLM (preview):\n{prompt_preview}\n")
        
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
                    response = self.model.generate_content(
                        [prompt, image],
                        generation_config=self.generation_config_fast,
                    )
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
                    # Build bin descriptions from JSON in appropriate language
                    bin_desc_text = ""
                    if self.bin_layout:
                        bin_list = []
                        for bin_info in self.bin_layout:
                            bin_type = bin_info.get('type', '').upper()
                            color = bin_info.get('color', '')
                            sign = bin_info.get('sign', '')
                            if self.language == 'hungarian':
                                bin_translations = {
                                    'RECYCLING': 'ÚJRAHASZNOSÍTÁS',
                                    'COMPOST': 'KOMPOSZT',
                                    'LANDFILL': 'SZEMÉTLERAKÓ'
                                }
                                color_translations = {
                                    'blue': 'kék',
                                    'green': 'zöld',
                                    'black': 'fekete',
                                    'grey': 'szürke',
                                    'gray': 'szürke',
                                    'gray and black': 'szürke és fekete'
                                }
                                bin_type_display = bin_translations.get(bin_type, bin_type)
                                color_display = color_translations.get(color.lower(), color)
                                bin_list.append(f"- {bin_type_display} ({color_display}): {sign}")
                            else:
                                bin_list.append(f"- {bin_type} ({color}): {sign}")
                        bin_desc_text = "\n".join(bin_list)
                    else:
                        if self.language == 'hungarian':
                            bin_desc_text = """- ÚJRAHASZNOSÍTÁS = kék kuka
- KOMPOSZT = zöld kuka
- SZEMÉTLERAKÓ = fekete vagy szürke kuka"""
                        else:
                            bin_desc_text = """- RECYCLING = blue bin
- COMPOST = green bin  
- LANDFILL = black or grey bin"""
                    
                    if self.language == 'hungarian':
                        text_prompt = f"""Gyorsan osztályozd ezeket a tárgyakat. Formátum: [tárgy] megy a [bin_type] kukába általában [color]

KRITIKUS: Csak SZEMÉT/HULLADÉK tárgyakat osztályozz. FIGYELMEN KÍVÜL: személyes tárgyak (telefonok, sapkák, kesztyűk, pénztárcák, kulcsok, táskák, stb.).

ELÉRHETŐ KUKÁK (használd ezeket a szabályokat):
{bin_desc_text}

Tárgyak: {items_text}

Válaszolj gyorsan KIZÁRÓLAG MAGYARUL formátumban: [tárgy] megy a [bin_type] kukába általában [color]"""
                    else:
                        text_prompt = f"""Quickly classify these items. Format: [item] goes into [bin_type] bin usually [color]

CRITICAL: ONLY classify items that are TRASH/WASTE. IGNORE personal items (phones, caps, beanies, wallets, keys, bags, etc.).

AVAILABLE BINS (use these exact rules):
{bin_desc_text}

Items: {items_text}

Respond fast in format: [item] goes into [bin_type] bin usually [color]"""
                    
                    try:
                        response = self.model.generate_content(
                            text_prompt,
                            generation_config=self.generation_config_fast,
                        )
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
                # Build bin descriptions from JSON in appropriate language
                bin_desc_text = ""
                if self.bin_layout:
                    bin_list = []
                    for bin_info in self.bin_layout:
                        bin_type = bin_info.get('type', '').upper()
                        color = bin_info.get('color', '')
                        sign = bin_info.get('sign', '')
                        if self.language == 'hungarian':
                            bin_translations = {
                                'RECYCLING': 'ÚJRAHASZNOSÍTÁS',
                                'COMPOST': 'KOMPOSZT',
                                'LANDFILL': 'SZEMÉTLERAKÓ'
                            }
                            color_translations = {
                                'blue': 'kék',
                                'green': 'zöld',
                                'black': 'fekete',
                                'grey': 'szürke',
                                'gray': 'szürke',
                                'gray and black': 'szürke és fekete'
                            }
                            bin_type_display = bin_translations.get(bin_type, bin_type)
                            color_display = color_translations.get(color.lower(), color)
                            bin_list.append(f"- {bin_type_display} ({color_display}): {sign}")
                        else:
                            bin_list.append(f"- {bin_type} ({color}): {sign}")
                    bin_desc_text = "\n".join(bin_list)
                else:
                    if self.language == 'hungarian':
                        bin_desc_text = """- ÚJRAHASZNOSÍTÁS = kék kuka
- KOMPOSZT = zöld kuka
- SZEMÉTLERAKÓ = fekete vagy szürke kuka"""
                    else:
                        bin_desc_text = """- RECYCLING = blue bin
- COMPOST = green bin
- LANDFILL = black or grey bin"""
                
                if self.language == 'hungarian':
                    text_prompt = f"""Gyorsan osztályozd ezeket a tárgyakat. Formátum: [tárgy] megy a [bin_type] kukába általában [color]

KRITIKUS: Csak SZEMÉT/HULLADÉK tárgyakat osztályozz. FIGYELMEN KÍVÜL: személyes tárgyak (telefonok, sapkák, kesztyűk, pénztárcák, kulcsok, táskák, stb.).

ELÉRHETŐ KUKÁK (használd ezeket a szabályokat):
{bin_desc_text}

Tárgyak: {items_text}

Válaszolj gyorsan KIZÁRÓLAG MAGYARUL formátumban: [tárgy] megy a [bin_type] kukába általában [color]"""
                else:
                    text_prompt = f"""Quickly classify these items. Format: [item] goes into [bin_type] bin usually [color]

CRITICAL: ONLY classify items that are TRASH/WASTE. IGNORE personal items (phones, caps, beanies, wallets, keys, bags, etc.).

AVAILABLE BINS (use these exact rules):
{bin_desc_text}

Items: {items_text}

Respond fast in format: [item] goes into [bin_type] bin usually [color]"""
                
                try:
                    response = self.model.generate_content(
                        text_prompt,
                        generation_config=self.generation_config_fast,
                    )
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
            
            # Check if response indicates no trash found (both English and Hungarian)
            # Only check for clear "no trash" statements, not just any occurrence of "no"
            response_lower = response_text.lower()
            no_trash_phrases = [
                'no trash items', 'no waste items', 'no items found', 
                'no trash found', 'no waste found', 'nothing to classify',
                'i see no trash', 'i don\'t see any', 'there is no trash', 'there are no items',
                'cannot see any', 'can\'t see any', 'unable to see any',
                'no trash', 'no items', 'no waste'  # Only at start or as standalone
            ]
            if self.language == 'hungarian':
                no_trash_phrases.extend([
                    'nincs szemét tárgy', 'nincs hulladék tárgy', 'nincs találat',
                    'nem látok szemét', 'nem látom a szemét', 'nincs itt semmi',
                    'nincs szemét', 'nincs tárgy', 'nincs hulladék'  # Only clear statements
                ])
            
            # Check entire response - but only if it's a clear "no trash" statement
            # Don't trigger on responses that contain "no" but also have items
            response_starts_with_no = any(response_lower.strip().startswith(phrase) for phrase in [
                'no trash', 'no items', 'no waste', 'nincs szemét', 'nincs tárgy', 'nincs hulladék'
            ])
            response_contains_clear_no = any(phrase in response_lower for phrase in [
                'no trash items', 'no waste items', 'no items found', 'no trash found',
                'nincs szemét tárgy', 'nincs hulladék tárgy', 'nincs találat'
            ])
            
            # Only return empty if it's clearly a "no trash" response
            if response_starts_with_no or (response_contains_clear_no and len(response_text.split()) < 10):
                # Short response with clear "no trash" - return empty
                print("✅ No trash items detected in response")
                return []
            
            # Helper function to validate item names
            def is_valid_item_name(item_name):
                """
                Validate that an extracted item name is actually a valid trash item,
                not a common word, negative phrase, or explanation text.
                """
                if not item_name or len(item_name.strip()) < 2:
                    return False
                
                item_lower = item_name.lower().strip()
                
                # Reject common words and phrases
                invalid_words = [
                    # English
                    'no', 'not', 'none', 'nothing', 'this', 'that', 'these', 'those',
                    'note', 'note:', 'note that', 'there', 'here', 'where', 'what',
                    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'can', 'cannot', 'can\'t', 'could', 'should', 'would', 'will',
                    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
                    'see', 'seen', 'look', 'found', 'find', 'check', 'analyze',
                    'image', 'picture', 'photo', 'frame', 'item', 'items', 'object',
                    'trash', 'waste', 'garbage', 'bin', 'bins', 'recycling', 'compost',
                    'landfill', 'goes', 'go', 'going', 'into', 'in', 'to', 'the',
                    # Hungarian
                    'nem', 'nincs', 'nincsen', 'semmi', 'ez', 'az', 'ezek', 'azok',
                    'itt', 'ott', 'ahol', 'ami', 'amit', 'amely', 'amelyik',
                    'látok', 'látom', 'lát', 'találok', 'találom', 'található',
                    'kép', 'képben', 'képen', 'tárgy', 'tárgyak', 'szemét', 'hulladék',
                    'kuka', 'kukák', 'újrahasznosítás', 'komposzt', 'szemétlerakó',
                    'megy', 'megy a', 'megyek', 'megyünk'
                ]
                
                # Check if item is a single invalid word
                if item_lower in invalid_words:
                    return False
                
                # Check if item starts with invalid words (e.g., "no trash", "ez nem")
                for invalid in invalid_words:
                    if item_lower.startswith(invalid + ' ') or item_lower == invalid:
                        return False
                
                # Check for negative phrases
                negative_patterns = [
                    r'^(no|not|nem|nincs|nincsen)\s+',
                    r'^(this|that|ez|az)\s+(doesn\'t|does not|nem|nincs)',
                    r'^(i|we|én|mi)\s+(don\'t|do not|nem|nincs)',
                    r'^(cannot|cannot|can\'t|nem)\s+',
                ]
                for pattern in negative_patterns:
                    if re.match(pattern, item_lower):
                        return False
                
                # Item name should be at least 2 characters (allow short names like "cup", "can")
                if len(item_lower.replace('-', '').replace(' ', '')) < 2:
                    return False
                
                # Allow common trash item words even if they're short
                valid_trash_words = [
                    'cup', 'can', 'bag', 'box', 'jar', 'lid', 'cap', 'bottle', 'fork',
                    'knife', 'spoon', 'plate', 'napkin', 'paper', 'cardboard', 'pizza',
                    'apple', 'banana', 'bottle', 'wrapper', 'container', 'bottle',
                    'pohár', 'doboz', 'táska', 'pohár', 'tálca', 'szalvéta', 'papír'
                ]
                if item_lower in valid_trash_words:
                    return True
                
                return True
            
            # Parse response - look for natural format: "item goes into bin_type bin usually color"
            items_classifications = []
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check if this line indicates "no trash" before parsing
                line_lower = line.lower()
                # Only skip if line clearly says "no trash" and doesn't contain item format
                line_says_no_trash = any(phrase in line_lower for phrase in [
                    'no trash items', 'no waste items', 'no items found',
                    'nincs szemét tárgy', 'nincs hulladék tárgy'
                ])
                has_item_format = 'goes into' in line_lower or 'megy a' in line_lower
                
                if line_says_no_trash and not has_item_format:
                    # Skip this line - it's saying no trash
                    continue
                
                # Store original line for dash check (before stripping)
                original_line = line
                
                # Stricter line acceptance: must have explicit format indicators
                # Accept lines with "goes into/goes in" (English) or "megy a" (Hungarian)
                # OR lines with dash + colon format (item: bin_type)
                has_goes_into = 'goes into' in line_lower or 'goes in' in line_lower
                has_megy_a = 'megy a' in line_lower  # Hungarian
                has_dash_colon = original_line.startswith('-') and ':' in original_line
                
                # Only accept lines that match expected formats
                if not (has_goes_into or has_megy_a or has_dash_colon):
                    continue
                
                # Remove leading dash if present
                line = line.lstrip('-').strip()
                line_lower = line.lower()  # Update after stripping
                
                # Parse natural format: "item goes into bin_type bin usually color" (English)
                # Or: "tárgy megy a bin_type kukába általában color" (Hungarian)
                # Or: "item: bin_type - explanation"
                
                # Extract bin type first - check both English and Hungarian
                bin_type = "landfill"  # default
                if self.language == 'hungarian':
                    # Hungarian bin type detection
                    if "újrahasznosítás" in line_lower or "recycle" in line_lower or "recycling" in line_lower:
                        bin_type = "recycling"
                    elif "komposzt" in line_lower or "compost" in line_lower:
                        bin_type = "compost"
                    elif "szemétlerakó" in line_lower or "landfill" in line_lower or "trash" in line_lower:
                        bin_type = "landfill"
                else:
                    # English bin type detection
                    if "recycle" in line_lower or "recycling" in line_lower:
                        bin_type = "recycling"
                    elif "compost" in line_lower:
                        bin_type = "compost"
                    elif "landfill" in line_lower or "trash" in line_lower:
                        bin_type = "landfill"
                
                # Extract item name and create natural explanation
                item_name = ""
                explanation = ""
                
                # Pattern 1: "item goes into bin_type bin usually color" (English)
                # Pattern 1b: "tárgy megy a bin_type kukába általában color" (Hungarian)
                if self.language == 'hungarian' and "megy a" in line_lower:
                    # Hungarian format: "tárgy megy a bin_type kukába általában color"
                    # Check for negative phrases first
                    if any(neg in line_lower for neg in ['nem megy', 'nincs', 'nincsen', 'semmi']):
                        continue  # Skip negative statements
                    
                    parts = re.split(r'megy a', line_lower, 1)
                    if len(parts) >= 2:
                        item_name = parts[0].strip()
                        
                        # Validate item name before proceeding
                        if not is_valid_item_name(item_name):
                            continue
                        
                        rest = parts[-1].strip()
                        
                        # Extract color if mentioned (Hungarian)
                        color = ""
                        if "kék" in rest:
                            color = "kék"
                        elif "zöld" in rest:
                            color = "zöld"
                        elif "fekete" in rest or "szürke" in rest:
                            color = "fekete vagy szürke"
                        
                        # Extract bin type from rest if not already found
                        if bin_type == "landfill" and "újrahasznosítás" in rest:
                            bin_type = "recycling"
                        elif bin_type == "landfill" and "komposzt" in rest:
                            bin_type = "compost"
                        elif bin_type == "landfill" and "szemétlerakó" in rest:
                            bin_type = "landfill"
                        
                        explanation = f"Ez megy a {bin_type} kukába."
                elif "goes into" in line_lower or "goes in" in line_lower:
                    # English format: Split on "goes into" or "goes in"
                    # Check for negative phrases first
                    if any(neg in line_lower for neg in ['doesn\'t go', 'does not go', 'cannot go', 'can\'t go', 'no ']):
                        continue  # Skip negative statements
                    
                    parts = re.split(r'goes (into|in)', line_lower, 1)
                    if len(parts) >= 2:
                        item_name = parts[0].strip()
                        
                        # Validate item name before proceeding
                        if not is_valid_item_name(item_name):
                            continue
                        
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
                
                # Pattern 2: "item: bin_type - explanation" (fallback, but stricter)
                elif has_dash_colon and ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        item_name = parts[0].strip()
                        
                        # Validate item name before proceeding
                        if not is_valid_item_name(item_name):
                            continue
                        
                        rest = parts[1].strip()
                        
                        # Skip if rest contains "no trash" indicators
                        if any(phrase in rest.lower() for phrase in no_trash_phrases):
                            continue
                        
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
                else:
                    # No valid pattern matched, skip this line
                    continue
                
                # If we found an item, validate and add it (only if bin type exists in current layout)
                if item_name and is_valid_item_name(item_name):
                    # Validate that this bin type exists in current bin_layout
                    if self._bin_type_exists(bin_type):
                        # Get bin name/label from bin_layout
                        bin_name = self._get_bin_name_for_type(bin_type)
                        bin_color = self._get_bin_color_for_type(bin_type)
                        bin_position = self._get_bin_position_for_type(bin_type)
                        print(f"✅ Classified '{item_name}' → {bin_type.upper()} bin (Color: {bin_color}, Position: {bin_position})")
                        items_classifications.append({
                            'item': item_name,
                            'bin_type': bin_type,
                            'bin_name': bin_name,
                            'bin_color': bin_color,
                            'bin_position': bin_position,
                            'explanation': explanation if explanation else f"This goes in the {bin_type} bin."
                        })
                    else:
                        # Bin type doesn't exist - check if item can go into any available bin
                        print(f"⚠️ Preferred bin '{bin_type}' not available. Checking compatibility with available bins...")
                        alternative_result = self._check_alternative_bin(item_name, bin_type, image)
                        if alternative_result:
                            items_classifications.append(alternative_result)
                        else:
                            # Item can't go into any available bin
                            preferred_bin_name = self._get_bin_name_for_type(bin_type)
                            if self.language == 'hungarian':
                                explanation = f"Ez nem mehet egyik kukába sem, amit itt találsz. Keresd meg a legközelebbi {preferred_bin_name}."
                            else:
                                explanation = f"This can't go into any of the trash bins you have. You have to find the closest {preferred_bin_name}."
                            
                            items_classifications.append({
                                'item': item_name,
                                'bin_type': None,  # No bin available
                                'bin_name': preferred_bin_name,
                                'bin_color': self._get_bin_color_for_type(bin_type),
                                'bin_position': None,  # No position for unavailable bin
                                'explanation': explanation,
                                'no_bin_available': True
                            })
            
            # If no structured items found, check if response might have items in different format
            if not items_classifications:
                # Log the raw response for debugging
                print(f"⚠️ No items parsed. Raw response: {response_text[:200]}...")
                
                # Check if response contains bin types but no items were parsed
                # This might indicate a parsing issue, not necessarily "no trash"
                has_bin_mentions = any(word in response_lower for word in [
                    'recycling', 'compost', 'landfill', 'recycle',
                    'újrahasznosítás', 'komposzt', 'szemétlerakó'
                ])
                
                if has_bin_mentions:
                    print("⚠️ Response mentions bins but no items parsed - possible parsing issue")
                else:
                    print("✅ No trash items detected in response")
                
                return []
            
            return items_classifications
        except Exception as e:
            print(f"Error classifying from image: {e}")
            item_name = detected_items[0]['class'] if detected_items else "item"
            # Return as list for consistency
            bin_name = self._get_bin_name_for_type('landfill')
            bin_position = self._get_bin_position_for_type('landfill')
            return [{
                'bin_type': 'landfill',
                'bin_name': bin_name,
                'bin_color': self._get_bin_color_for_type('landfill'),
                'bin_position': bin_position,
                'explanation': f"Unable to classify from image. Please check local recycling guidelines.",
                'item': item_name
            }]
    
    def _bin_type_exists(self, bin_type):
        """
        Check if a bin type exists in the current bin_layout
        """
        if not self.bin_layout:
            return True  # If no layout, allow all (fallback mode)
        
        bin_type_lower = bin_type.lower()
        for bin_info in self.bin_layout:
            if bin_info.get('type', '').lower() == bin_type_lower:
                return True
        return False
    
    def _check_alternative_bin(self, item_name, preferred_bin_type, image=None):
        """
        Check if an item can go into any available bin when preferred bin isn't available.
        Returns classification dict if >60% confidence, None otherwise.
        """
        if not self.bin_layout or len(self.bin_layout) == 0:
            return None
        
        # Get available bin types
        available_bins = [bin_info.get('type', '').lower() for bin_info in self.bin_layout]
        if not available_bins:
            return None
        
        # Build prompt to check compatibility
        available_bin_descriptions = []
        for bin_info in self.bin_layout:
            bin_type = bin_info.get('type', '').upper()
            color = bin_info.get('color', '')
            sign = bin_info.get('sign', '')
            if self.language == 'hungarian':
                bin_translations = {
                    'RECYCLING': 'ÚJRAHASZNOSÍTÁS',
                    'COMPOST': 'KOMPOSZT',
                    'LANDFILL': 'SZEMÉTLERAKÓ'
                }
                color_translations = {
                    'blue': 'kék',
                    'green': 'zöld',
                    'black': 'fekete',
                    'grey': 'szürke',
                    'gray': 'szürke',
                    'gray and black': 'szürke és fekete'
                }
                bin_type_display = bin_translations.get(bin_type, bin_type)
                color_display = color_translations.get(color.lower(), color)
                available_bin_descriptions.append(f"- {bin_type_display} ({color_display}): {sign}")
            else:
                available_bin_descriptions.append(f"- {bin_type} ({color}): {sign}")
        
        preferred_bin_name = self._get_bin_name_for_type(preferred_bin_type)
        
        if self.language == 'hungarian':
            prompt = f"""Elemezd: A "{item_name}" tárgy általában a {preferred_bin_name} kukába megy, de ez a kuka nem elérhető.

ELÉRHETŐ KUKÁK:
{chr(10).join(available_bin_descriptions)}

KÉRDÉS: A "{item_name}" mehet-e valamelyik elérhető kukába? Válaszolj egy számot 0-100 között, ahol:
- 0-59: NEM mehet egyik elérhető kukába sem
- 60-100: IGEN, mehet egy elérhető kukába (melyik?)

Válasz formátum: [szám]% - [igen/nem] - [ha igen, melyik kuka?]"""
        else:
            prompt = f"""Analyze: The item "{item_name}" normally goes into {preferred_bin_name}, but that bin is not available.

AVAILABLE BINS:
{chr(10).join(available_bin_descriptions)}

QUESTION: Can "{item_name}" go into any of the available bins? Answer with a number 0-100, where:
- 0-59: NO, cannot go into any available bin
- 60-100: YES, can go into an available bin (which one?)

Answer format: [number]% - [yes/no] - [if yes, which bin?]"""
        
        try:
            if image and self.supports_vision:
                # Use vision if image available - downscale for speed
                processed_image = self._to_pil_and_downscale(image) if image else None
                response = self.model.generate_content(
                    [prompt, processed_image],
                    generation_config=self.generation_config_fast,
                )
            else:
                # Text-only
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config_fast,
                )
            
            response_text = response.text.strip()
            
            # Extract percentage from response
            import re
            percent_match = re.search(r'(\d+)%', response_text)
            if percent_match:
                confidence = int(percent_match.group(1))
                
                if confidence >= 60:
                    # Item can go into an available bin
                    # Find which available bin
                    response_lower = response_text.lower()
                    alternative_bin_type = None
                    
                    for bin_info in self.bin_layout:
                        bin_type_check = bin_info.get('type', '').lower()
                        if self.language == 'hungarian':
                            if bin_type_check == 'recycling' and ('újrahasznosítás' in response_lower or 'recycle' in response_lower):
                                alternative_bin_type = bin_type_check
                                break
                            elif bin_type_check == 'compost' and ('komposzt' in response_lower or 'compost' in response_lower):
                                alternative_bin_type = bin_type_check
                                break
                            elif bin_type_check == 'landfill' and ('szemétlerakó' in response_lower or 'landfill' in response_lower):
                                alternative_bin_type = bin_type_check
                                break
                        else:
                            if bin_type_check in response_lower:
                                alternative_bin_type = bin_type_check
                                break
                    
                    # Default to first available bin if not found
                    if not alternative_bin_type:
                        alternative_bin_type = self.bin_layout[0].get('type', '').lower()
                    
                    bin_name = self._get_bin_name_for_type(alternative_bin_type)
                    bin_color = self._get_bin_color_for_type(alternative_bin_type)
                    bin_position = self._get_bin_position_for_type(alternative_bin_type)
                    preferred_bin_name_display = preferred_bin_name
                    
                    if self.language == 'hungarian':
                        explanation = f"Ez nem a {preferred_bin_name_display} kukába való, de mivel az nem elérhető, mehet a {bin_name} kukába."
                    else:
                        explanation = f"This doesn't belong in {preferred_bin_name_display}, but since it isn't available, it can go into {bin_name}."
                    
                    print(f"✅ Alternative classification: '{item_name}' → {alternative_bin_type.upper()} bin (was {preferred_bin_type.upper()}, {confidence}% confidence)")
                    
                    return {
                        'item': item_name,
                        'bin_type': alternative_bin_type,
                        'bin_name': bin_name,
                        'bin_color': bin_color,
                        'bin_position': bin_position,
                        'explanation': explanation,
                        'alternative': True,
                        'preferred_bin': preferred_bin_type
                    }
                else:
                    # Confidence < 60%
                    print(f"⚠️ Item '{item_name}' cannot go into any available bin ({confidence}% confidence)")
                    return None
            else:
                # Couldn't parse percentage, assume no
                print(f"⚠️ Could not parse confidence from response: {response_text[:100]}")
                return None
                
        except Exception as e:
            print(f"⚠️ Error checking alternative bin: {e}")
            return None
    
    def _get_bin_name_for_type(self, bin_type):
        """
        Get the bin name/label for a given bin type from the JSON layout.
        Returns a human-readable name like "recycling bin", "compost bin", "landfill bin".
        Supports Hungarian translation when language is set to Hungarian.
        """
        # Translate bin type to Hungarian if needed
        if self.language == 'hungarian':
            bin_translations = {
                'recycling': 'újrahasznosítás',
                'compost': 'komposzt',
                'landfill': 'szemétlerakó'
            }
            bin_type_display = bin_translations.get(bin_type.lower(), bin_type)
            bin_name = f"{bin_type_display} kuka"
        else:
            bin_name = f"{bin_type} bin"
        
        if not self.bin_layout:
            # Fallback to bin_type if no layout
            return bin_name
        
        bin_type_lower = bin_type.lower()
        for bin_info in self.bin_layout:
            if bin_info.get('type', '').lower() == bin_type_lower:
                label = bin_info.get('label', '')
                print(f"  📦 Bin name for {bin_type}: '{bin_name}' (from label: {label})")
                return bin_name
        
        # Default fallback
        return bin_name
    
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
                color = bin_info.get('color', 'blue')
                print(f"  🎨 Bin color for {bin_type}: '{color}' (from JSON)")
                return color
        
        # Default fallback
        print(f"  ⚠️ Bin color for {bin_type}: 'blue' (fallback - bin not found in layout)")
        return 'blue'
    
    def _get_bin_position_for_type(self, bin_type):
        """
        Get the bin position (left, middle, right) for a given bin type from the JSON layout.
        Returns human-readable position like "on the left", "in the middle", "on the right".
        """
        if not self.bin_layout:
            return None
        
        bin_type_lower = bin_type.lower()
        for bin_info in self.bin_layout:
            if bin_info.get('type', '').lower() == bin_type_lower:
                pos = bin_info.get('pos', '').lower()
                color = bin_info.get('color', '')
                
                # Convert position to readable format
                if self.language == 'hungarian':
                    if 'left' in pos or pos == 'leftmost':
                        return 'bal oldalon'
                    elif 'center' in pos or 'middle' in pos:
                        return 'középen'
                    elif 'right' in pos or pos == 'rightmost':
                        return 'jobb oldalon'
                    elif 'second' in pos:
                        if 'left' in pos:
                            return 'balról második'
                        elif 'right' in pos:
                            return 'jobbról második'
                else:  # English
                    if 'left' in pos or pos == 'leftmost':
                        return 'on the left'
                    elif 'center' in pos or 'middle' in pos:
                        return 'in the middle'
                    elif 'right' in pos or pos == 'rightmost':
                        return 'on the right'
                    elif 'second' in pos:
                        if 'left' in pos:
                            return 'second from left'
                        elif 'right' in pos:
                            return 'second from right'
                
                # Fallback: return position as-is if not recognized
                return pos
        
        return None
    
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
        
        # Handle "repeat" or "say again" requests (quick check before LLM call)
        if any(phrase in question_lower for phrase in ['repeat', 'say again', 'what did you say', 'can you repeat']):
            if last_classifications:
                # Reconstruct what was said
                responses = []
                for item in last_classifications:
                    bin_name = self._get_bin_name_for_type(item.get('bin_type', 'bin'))
                    bin_color = item.get('bin_color', self._get_bin_color_for_type(item.get('bin_type', 'bin')))
                    bin_position = item.get('bin_position', None)
                    if bin_position:
                        responses.append(f"{item.get('item', 'item')} goes into {bin_color} {bin_name} {bin_position}")
                    else:
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

User Input: {question}{context}

CRITICAL: First, determine if this input was INTENDED for the bin system or is just chit chat/greeting.

Examples of inputs NOT intended for the bin system (ignore these silently):
- Greetings: "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye"
- Small talk: "how are you", "what's up", "how's it going", "nice", "cool", "awesome"
- General questions: "what's the weather", "tell me a joke", "what do you think", "do you like X"
- Unrelated topics: questions about sports, movies, recipes, general knowledge not related to waste disposal

If the input is NOT intended for the bin system, respond with ONLY: "NOT_RELEVANT"

If the input IS intended for the bin system (questions about waste, recycling, bins, or items classified), then:
- Answer questions related to waste disposal, recycling, composting, or the items you just classified
- If the question is about a recent classification, explain WHY based on the item's condition, material, and contamination level
- Be helpful, clear, and concise
- If asked "why" about a classification, explain the reasoning based on the bin rules and item condition

Answer:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config_fast,
            )
            response_text = response.text.strip()
            # Store for logging
            self.last_raw_response = response_text
            
            # Check if Gemini determined the question is not relevant
            if response_text.upper().startswith("NOT_RELEVANT") or response_text.upper() == "NOT_RELEVANT":
                return None  # Not relevant - chit chat, ignore silently
            
            return response_text
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error: {str(e)}"
            self.last_raw_response = error_msg
            return error_msg
