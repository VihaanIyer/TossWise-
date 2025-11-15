"""
Gemini API Integration for Trash Classification
Determines which bin (recycling, compost, landfill) an item should go into
"""

import os
import re
import json
import io

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


class TrashClassifier:
    def __init__(self, language="english", debug=False):
        """
        Initialize Gemini API client and pick a fast, vision-capable model.
        IMPORTANT: Create this ONCE and reuse it.

        Args:
            language: Language to use for prompts and responses ('english' or 'hungarian')
            debug: enable extra logging when True
        """
        self.language = language.lower()
        self.DEBUG = debug

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)

        preferred_models = [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro-vision",
            "gemini-pro",  # text-only fallback
        ]

        self.model = None
        self.model_name = None
        self.supports_vision = False

        for model_name in preferred_models:
            for variant in (model_name, f"models/{model_name}"):
                try:
                    model = genai.GenerativeModel(variant)
                    # Test a tiny call to fail fast if model is not available
                    _ = model.generate_content("ping").text
                    self.model = model
                    self.model_name = model_name
                    if any(token in model_name.lower() for token in ["vision", "flash", "1.5", "2.0", "2.5"]):
                        self.supports_vision = True
                    if self.DEBUG:
                        print(f"[TrashClassifier] Loaded Gemini model: {self.model_name} (vision: {self.supports_vision})")
                    break
                except Exception:
                    continue
            if self.model is not None:
                break

        if self.model is None:
            raise ValueError("Could not initialize any Gemini model. Check your API key and model availability.")

        # Tighter config for speed
        self.generation_config_fast = {
            "temperature": 0.0,        # deterministic
            "max_output_tokens": 64,   # short, enough for our format
            "top_p": 0.9,
        }

        self.last_raw_response = ""

        # Load bin layout metadata from JSON (with location support)
        self.bin_layout = self._load_bin_layout()

        # Build system prompt based on actual bin layout (used in some calls)
        self.system_prompt = self._build_system_prompt()

        self.bin_context = ""
        self.bin_layout_metadata = None

    # ---------------------- helpers ---------------------- #

    @staticmethod
    def _to_pil_and_downscale(image, max_dim: int = 384) -> Image.Image:
        """
        Convert numpy/OpenCV or PIL image to PIL and downscale to max_dim.
        Optimized to 384px for faster API calls while maintaining accuracy.
        """
        if isinstance(image, Image.Image):
            pil_img = image
        else:
            if not hasattr(image, "shape"):
                raise TypeError("Unsupported image type for conversion to PIL.")
            import numpy as np  # local import
            arr = image
            if len(arr.shape) == 3 and arr.shape[2] == 3:
                # BGR -> RGB if this is OpenCV style
                pil_img = Image.fromarray(arr[:, :, ::-1])
            else:
                pil_img = Image.fromarray(arr)

        w, h = pil_img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        return pil_img

    def _load_bin_layout(self):
        """
        Load bin layout metadata from JSON file.
        Supports location-specific files via BIN_LOCATION environment variable.
        """
        try:
            base_dir = os.path.dirname(__file__)
            location = os.getenv("BIN_LOCATION", None)

            if location:
                location_file = os.path.join(base_dir, f"bin_layout_{location}.json")
                if os.path.exists(location_file):
                    with open(location_file, "r") as f:
                        data = json.load(f)
                    bins = data.get("bins", [])
                    if bins and self.DEBUG:
                        print(f"[TrashClassifier] Loaded {len(bins)} bins from {location_file}")
                    return bins

            json_path = os.path.join(base_dir, "bin_layout_metadata.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                bins = data.get("bins", [])
                if bins and self.DEBUG:
                    print(f"[TrashClassifier] Loaded {len(bins)} bins from bin_layout_metadata.json")
                return bins

            if self.DEBUG:
                print("[TrashClassifier] No bin layout JSON found, using defaults")
            return []
        except Exception as e:
            if self.DEBUG:
                print(f"[TrashClassifier] Error loading bin layout: {e}. Using defaults.")
            return []

    def _build_bin_descriptions_for_prompt(self):
        """
        Return a short, language-appropriate description of bins for prompts.
        """
        if not self.bin_layout:
            if self.language == "hungarian":
                return (
                    "- recycling (kék): tiszta újrahasznosítható anyagok\n"
                    "- compost (zöld): szerves/élelmiszer hulladék\n"
                    "- landfill (fekete/szürke): szennyezett vagy nem újrahasznosítható"
                )
            return (
                "- recycling (blue): clean recyclables\n"
                "- compost (green): food/organic waste\n"
                "- landfill (black/grey): contaminated or non-recyclable"
            )

        lines = []
        for bin_info in self.bin_layout:
            btype = bin_info.get("type", "").lower()
            color = bin_info.get("color", "")
            sign = bin_info.get("sign", "")
            label = bin_info.get("label", "")

            if self.language == "hungarian":
                type_map = {
                    "recycling": "recycling",
                    "compost": "compost",
                    "landfill": "landfill",
                }  # keep English keywords for parsing, explain around them
                t_disp = type_map.get(btype, btype)
                lines.append(f"- {t_disp} ({color}): {sign or label}")
            else:
                lines.append(f"- {btype or 'bin'} ({color}): {sign or label}")
        return "\n".join(lines)

    def _build_system_prompt(self):
        """
        Build system prompt based on actual bin layout from JSON.
        Used in some text-only calls.
        """
        base = (
            "You are a waste classification expert. Analyze items and decide which bin "
            "(recycling, compost, landfill) they belong to based on material and contamination.\n\n"
        )
        return base + "BIN SYSTEM:\n" + self._build_bin_descriptions_for_prompt()

    def _get_bin_name_for_type(self, bin_type: str) -> str:
        """
        Human readable bin name.
        """
        b = bin_type.lower()
        if self.language == "hungarian":
            tr = {
                "recycling": "újrahasznosítási kuka",
                "compost": "komposzt kuka",
                "landfill": "kommunális kuka",
            }
            return tr.get(b, f"{b} kuka")
        return f"{b} bin"

    def _get_bin_color_for_type(self, bin_type: str) -> str:
        """
        Get the bin color for a given bin type from the JSON layout or fallback.
        """
        b = bin_type.lower()
        if self.bin_layout:
            for bin_info in self.bin_layout:
                if bin_info.get("type", "").lower() == b:
                    return bin_info.get("color", "") or self._fallback_color(b)

        return self._fallback_color(b)

    @staticmethod
    def _fallback_color(bin_type: str) -> str:
        b = bin_type.lower()
        if b == "recycling":
            return "blue"
        if b == "compost":
            return "green"
        if b == "landfill":
            return "black/grey"
        return ""

    def _get_bin_position_for_type(self, bin_type: str):
        """
        Optional: get approximate bin position (left/middle/right) if encoded in JSON.
        """
        if not self.bin_layout:
            return None

        b = bin_type.lower()
        for bin_info in self.bin_layout:
            if bin_info.get("type", "").lower() == b:
                pos = (bin_info.get("pos") or "").lower()
                if self.language == "hungarian":
                    if "left" in pos or pos == "leftmost":
                        return "bal oldalon"
                    if "center" in pos or "middle" in pos:
                        return "középen"
                    if "right" in pos or pos == "rightmost":
                        return "jobb oldalon"
                else:
                    if "left" in pos or pos == "leftmost":
                        return "on the left"
                    if "center" in pos or "middle" in pos:
                        return "in the middle"
                    if "right" in pos or pos == "rightmost":
                        return "on the right"
                return pos or None
        return None

    def _build_explanation(self, item_name: str, bin_type: str, bin_color: str) -> str:
        b = bin_type.lower()
        if self.language == "hungarian":
            msg = {
                "recycling": "Ez főleg újrahasznosítható anyag, ezért az újrahasznosítási kukába kerül.",
                "compost": "Ez főleg szerves vagy élelmiszer hulladék, ezért a komposzt kukába kerül.",
                "landfill": "Ez nem újrahasznosítható vagy szennyezett, ezért a kommunális szemétbe kerül.",
            }
            return msg.get(b, "Ez ebbe a kukába kerül.")
        msg = {
            "recycling": "This is recyclable so it goes in the recycling bin.",
            "compost": "This is mostly organic/food waste so it goes in the compost bin.",
            "landfill": "This is non-recyclable or contaminated so it goes in the landfill bin.",
        }
        return msg.get(b, "This goes in that bin.")

    # ---------------------- main APIs ---------------------- #

    def classify_item(self, item_name, context="", image=None):
        """
        Classify a single item into the appropriate bin.
        If image is provided and model supports vision, it will be used.
        """
        # Short, text-only fallback. Used mainly when called with just YOLO label.
        if not item_name:
            return {
                "bin_type": "landfill",
                "explanation": "No item specified.",
                "item": item_name or "item",
            }

        bins_desc = self._build_bin_descriptions_for_prompt()

        if self.language == "hungarian":
            prompt = f"""Te egy hulladék-osztályozó asszisztens vagy.

Feladat: Döntsd el, hogy az alábbi tárgy melyik kukába való: recycling / compost / landfill.

KUKÁK:
{bins_desc}

Tárgy: "{item_name}"
Kontekstus (opcionális): "{context}"

VÁLASZ FORMÁTUM:
bin_type: [recycling|compost|landfill]

Csak ezt a sort add vissza."""
        else:
            prompt = f"""You are a waste classification assistant.

Task: Decide which bin this item belongs in: recycling / compost / landfill.

BINS:
{bins_desc}

Item: "{item_name}"
Context (optional): "{context}"

ANSWER FORMAT:
bin_type: [recycling|compost|landfill]

Return just that one line."""
        try:
            if image is not None and self.supports_vision:
                pil_img = self._to_pil_and_downscale(image)
                response = self.model.generate_content(
                    [prompt, pil_img],
                    generation_config=self.generation_config_fast,
                )
            else:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config_fast,
                )

            text = (response.text or "").strip().lower()
            self.last_raw_response = text

            m = re.search(r"(recycling|compost|landfill)", text)
            bin_type = m.group(1) if m else "landfill"
            explanation = self._build_explanation(item_name, bin_type, self._get_bin_color_for_type(bin_type))
            return {
                "bin_type": bin_type,
                "explanation": explanation,
                "item": item_name,
            }
        except Exception as e:
            if self.DEBUG:
                print(f"[TrashClassifier] classify_item error: {e}")
            return {
                "bin_type": "landfill",
                "explanation": f"Unable to classify {item_name}. Please check local recycling guidelines.",
                "item": item_name,
            }

    def detect_bags_in_image(self, image):
        """
        Detect trash bags in the image using Gemini Vision.

        Returns:
            List of detected bags with their positions/descriptions.
        """
        try:
            pil_img = self._to_pil_and_downscale(image)
        except Exception as e:
            if self.DEBUG:
                print(f"[TrashClassifier] detect_bags_in_image conversion error: {e}")
            return []

        prompt = """Look for TRASH BAGS or GARBAGE BAGS that are ready to be disposed of.

Only count bags that clearly contain trash/waste.
Ignore backpacks, purses, shopping bags being carried, and empty bags.

If you see trash bags, respond like:
BAG_1: [position/description], BAG_2: [position/description], ...

If no trash bags: respond exactly "NO_BAGS"."""

        try:
            if not self.supports_vision:
                return []

            response = self.model.generate_content(
                [prompt, pil_img],
                generation_config=self.generation_config_fast,
            )
            text = (response.text or "").strip()
            self.last_raw_response = text

            if text.upper().startswith("NO_BAGS"):
                return []

            bags = []
            for num, desc in re.findall(r"BAG_(\d+):\s*(.+?)(?:,|$)", text, re.IGNORECASE):
                bags.append(
                    {
                        "number": int(num),
                        "description": desc.strip(),
                        "position": desc.strip(),
                    }
                )
            return bags
        except Exception as e:
            if self.DEBUG:
                print(f"[TrashClassifier] detect_bags_in_image error: {e}")
            return []

    def classify_bag_contents(self, bag_description):
        """
        Classify which bin a bag should go into based on user's description of contents.
        """
        bins_desc = self._build_bin_descriptions_for_prompt()

        if self.language == "hungarian":
            prompt = f"""A felhasználó ezt mondta a zsák tartalmáról: "{bag_description}"

Döntsd el, hogy a zsák inkább recycling, compost vagy landfill kategóriába tartozik.

KUKÁK:
{bins_desc}

VÁLASZ FORMÁTUM:
bin_type: [recycling|compost|landfill]"""

        else:
            prompt = f"""User described a trash bag: "{bag_description}"

Decide which bin the bag should go into overall: recycling, compost, or landfill.

BINS:
{bins_desc}

ANSWER FORMAT:
bin_type: [recycling|compost|landfill]"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config_fast,
            )
            text = (response.text or "").strip().lower()
            self.last_raw_response = text

            m = re.search(r"(recycling|compost|landfill)", text)
            bin_type = m.group(1) if m else "landfill"

            bin_color = self._get_bin_color_for_type(bin_type)
            bin_name = self._get_bin_name_for_type(bin_type)
            bin_position = self._get_bin_position_for_type(bin_type)

            explanation = self._build_explanation(bag_description, bin_type, bin_color)

            return {
                "bin_type": bin_type,
                "bin_name": bin_name,
                "bin_color": bin_color,
                "bin_position": bin_position,
                "explanation": explanation,
            }
        except Exception as e:
            if self.DEBUG:
                print(f"[TrashClassifier] classify_bag_contents error: {e}")
            bin_type = "landfill"
            bin_name = self._get_bin_name_for_type(bin_type)
            bin_color = self._get_bin_color_for_type(bin_type)
            bin_position = self._get_bin_position_for_type(bin_type)
            return {
                "bin_type": bin_type,
                "bin_name": bin_name,
                "bin_color": bin_color,
                "bin_position": bin_position,
                "explanation": "Unable to classify. Please check local recycling guidelines.",
            }

    def _rule_based_fallback(self, detected_items):
        """
        Simple rule-based fallback when API fails.
        """
        results = []
        if not detected_items:
            return results

        for item in detected_items:
            name = str(item.get("class", "item")).lower()
            bin_type = "landfill"

            # crude but fast
            if any(k in name for k in ["apple", "banana", "pizza", "food", "fruit", "vegetable", "bread", "meat", "coffee", "grounds"]):
                bin_type = "compost"
            elif any(k in name for k in ["bottle", "can", "jar", "glass", "aluminum", "metal", "plastic cup", "cup", "container"]):
                bin_type = "recycling"
            elif "paper" in name or "cardboard" in name:
                bin_type = "recycling"

            color = self._get_bin_color_for_type(bin_type)
            bin_name = self._get_bin_name_for_type(bin_type)
            pos = self._get_bin_position_for_type(bin_type)
            explanation = self._build_explanation(name, bin_type, color)

            results.append(
                {
                    "item": name,
                    "bin_type": bin_type,
                    "bin_name": bin_name,
                    "bin_color": color,
                    "bin_position": pos,
                    "explanation": explanation,
                }
            )
        return results

    def classify_item_from_image(self, image, detected_items=None):
        """
        Classify items directly from image using vision.

        Returns:
            List of dicts with keys:
            item, bin_type, bin_name, bin_color, bin_position, explanation
        """
        try:
            pil_img = self._to_pil_and_downscale(image)
        except Exception as e:
            if self.DEBUG:
                print(f"[TrashClassifier] classify_item_from_image conversion error: {e}")
            # fall back to rule-based using detected_items only
            return self._rule_based_fallback(detected_items)

        # Build optional YOLO hint text
        items_text = ""
        if detected_items:
            labels = sorted({str(it.get("class", "")).strip() for it in detected_items if it.get("class")})
            if labels:
                items_text = ", ".join(labels)

        bins_desc = self._build_bin_descriptions_for_prompt()

        if self.language == "hungarian":
            prompt = f"""Te egy hulladék-osztályozó asszisztens vagy.

Feladat: A KÉP alapján azonosítsd azokat a tárgyakat, amelyek SZEMÉT/HULLADÉK. Telefon, ruha, táska, kulcs stb. NEM szemét.

KUKÁK (bin_type kulcsszavak ANGOLUL, hogy könnyen feldolgozható legyen):
{bins_desc}

Ha látsz szemét tárgyakat, MINDEN tárgyat egy külön sorban adj vissza pontosan ebben a formátumban:

item_name | bin_type | bin_color | position

ahol:
- bin_type egyike: recycling, compost, landfill  (EZEKET használd, angolul!)
- bin_color a kuka fő színe (pl. blue, green, black/grey)
- position: left, middle, right vagy hagyd üresen

Ha NINCS szemét, csak ezt írd: NONE

(Opcionális, figyelmen kívül hagyható YOLO tipp): {items_text}
"""
        else:
            prompt = f"""You are a waste classification assistant.

Task: Using the IMAGE, identify ONLY trash/waste items. Ignore phones, clothes, bags, keys, and other personal belongings.

BINS:
{bins_desc}

If you see trash items, output ONE LINE PER ITEM in this exact format:

item_name | bin_type | bin_color | position

where:
- bin_type is one of: recycling, compost, landfill
- bin_color is the bin's main color (e.g. blue, green, black/grey)
- position is left, middle, right, or can be left empty

If there is NO trash, respond with exactly:
NONE

(Optional hint, may be ignored): {items_text}
"""

        try:
            if self.supports_vision:
                response = self.model.generate_content(
                    [prompt, pil_img],
                    generation_config=self.generation_config_fast,
                )
            else:
                # text-only fallback; not ideal but still obeys same format
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config_fast,
                )

            text = (response.text or "").strip()
            self.last_raw_response = text

            if not text or text.upper().startswith("NONE"):
                return []

            results = []
            for line in text.splitlines():
                line = line.strip()
                if not line or line.upper() == "NONE":
                    continue

                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 2:
                    continue

                item_name = parts[0]
                bin_type = parts[1].lower()
                if bin_type not in ("recycling", "compost", "landfill"):
                    continue

                bin_color = parts[2] if len(parts) > 2 and parts[2] else self._get_bin_color_for_type(bin_type)
                position = parts[3] if len(parts) > 3 and parts[3] else self._get_bin_position_for_type(bin_type)

                bin_name = self._get_bin_name_for_type(bin_type)
                explanation = self._build_explanation(item_name, bin_type, bin_color)

                results.append(
                    {
                        "item": item_name,
                        "bin_type": bin_type,
                        "bin_name": bin_name,
                        "bin_color": bin_color,
                        "bin_position": position,
                        "explanation": explanation,
                    }
                )

            return results
        except Exception as e:
            if self.DEBUG:
                print(f"[TrashClassifier] classify_item_from_image API error: {e}")
            # last-resort rule-based
            return self._rule_based_fallback(detected_items)

    def update_bin_context(self, bin_layout_result):
        """
        Store bin layout metadata so future classifications can reference the physical bins.
        """
        if not bin_layout_result:
            self.bin_context = ""
            self.bin_layout_metadata = None
            return

        if isinstance(bin_layout_result, dict):
            bins = bin_layout_result.get("bins", []) or []
        elif isinstance(bin_layout_result, list):
            bins = bin_layout_result
        else:
            bins = []

        summaries = []
        for idx, bin_info in enumerate(bins, 1):
            if not isinstance(bin_info, dict):
                continue
            label = bin_info.get("bin_label") or f"Bin {idx}"
            bin_type = bin_info.get("bin_type_guess") or "unknown"
            color = bin_info.get("bin_color", "")
            signage = bin_info.get("signage_text", "")
            notes = bin_info.get("additional_notes", "")

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
        Answer user questions about recycling/waste disposal.
        Returns None if the question is clearly not about waste/bins.
        """
        q = (question or "").strip()
        if not q:
            return None

        q_lower = q.lower()

        # Quick filter for obviously irrelevant chat
        small_talk_keywords = [
            "how are you",
            "what's up",
            "how is it going",
            "joke",
            "weather",
            "movie",
            "song",
        ]
        if any(k in q_lower for k in small_talk_keywords):
            return None

        # Repeat last classifications if user asks to repeat
        if any(k in q_lower for k in ["repeat", "say again", "what did you say", "can you repeat"]):
            if last_classifications:
                msgs = []
                for item in last_classifications:
                    name = item.get("item", "item")
                    btype = item.get("bin_type", "bin")
                    bname = self._get_bin_name_for_type(btype)
                    bcolor = item.get("bin_color", self._get_bin_color_for_type(btype))
                    pos = item.get("bin_position") or ""
                    if pos:
                        msgs.append(f"{name} → {bcolor} {bname} {pos}")
                    else:
                        msgs.append(f"{name} → {bcolor} {bname}")
                return ". ".join(msgs)
            return None

        bins_desc = self._build_bin_descriptions_for_prompt()

        context = ""
        if last_classifications:
            lines = []
            for item in last_classifications:
                name = item.get("item", "item")
                btype = item.get("bin_type", "bin")
                bname = self._get_bin_name_for_type(btype)
                expl = item.get("explanation", "")
                lines.append(f"{name} → {bname}. {expl}")
            if lines:
                context = "\nRecent classifications:\n" + "\n".join(lines)

        if self.language == "hungarian":
            prompt = f"""Te egy hulladék-osztályozó asszisztens vagy.

Feladat: Válaszolj a felhasználó kérdésére, HA a kérdés hulladék, kuka, újrahasznosítás vagy komposzt témához kapcsolódik.

KUKÁK:
{bins_desc}

Felhasználói kérdés: "{q}"
{context}

Ha a kérdés NEM kapcsolódik a hulladék/kuka témához, válaszolj pontosan így:
NOT_RELEVANT

Ha kapcsolódik, adj rövid, érthető magyarázatot magyarul."""
        else:
            prompt = f"""You are a smart trash bin assistant.

Task: Answer the user's question ONLY if it is about waste, bins, recycling, compost, or the items just classified.

BINS:
{bins_desc}

User question: "{q}"
{context}

If the question is NOT about waste/bins, respond with exactly:
NOT_RELEVANT

If it IS relevant, give a short, clear explanation."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config_fast,
            )
            text = (response.text or "").strip()
            self.last_raw_response = text

            if text.upper().startswith("NOT_RELEVANT"):
                return None
            return text
        except Exception as e:
            if self.DEBUG:
                print(f"[TrashClassifier] answer_question error: {e}")
            return None