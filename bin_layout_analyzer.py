"""
Analyze a reference photo of facility waste bins to extract labels for downstream classification.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image


class BinLayoutAnalyzer:
    """
    Runs a one-time Gemini vision pass over a bin lineup photo and parses the response.
    Update BIN_LAYOUT_IMAGE_PATH to point to your actual reference image.
    """

    BIN_LAYOUT_IMAGE_PATH = "./bins.jpeg"
    BIN_LAYOUT_OUTPUT_PATH = Path("./bin_layout_metadata.json")

    def __init__(self, classifier):
        """
        Args:
            classifier: Instance of TrashClassifier (for Gemini model + logging hooks)
        """
        self.classifier = classifier
        if not getattr(self.classifier, "supports_vision", False):
            raise ValueError("Loaded Gemini model does not support vision; cannot analyze bin layout.")

    def _ensure_pil_image(self, source: Any) -> Image.Image:
        if isinstance(source, Image.Image):
            return source
        if hasattr(source, "shape"):
            import numpy as np  # local import to avoid hard dependency elsewhere

            if len(source.shape) == 3 and source.shape[2] == 3:
                image_rgb = source[:, :, ::-1]
                return Image.fromarray(image_rgb)
            return Image.fromarray(source)
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Bin layout image not found at {path}")
        return Image.open(path)

    def _safe_json_loads(self, blob: str):
        if not blob:
            return None
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            start = blob.find("{")
            end = blob.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(blob[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def _analyze_image(self, image: Image.Image, expected_bins: list = None) -> Dict[str, Any]:
        """
        Analyze a PIL Image and return parsed bin metadata.
        Internal method that accepts a PIL Image directly.
        
        Args:
            image: PIL Image to analyze
            expected_bins: Optional list of currently configured bins. If provided, only look for these bins.
        """
        # Build prompt with expected bins context if provided
        expected_bins_context = ""
        if expected_bins and len(expected_bins) > 0:
            bin_descriptions = []
            for bin_info in expected_bins:
                bin_type = bin_info.get('type', 'unknown')
                bin_color = bin_info.get('color', 'unknown')
                bin_descriptions.append(f"- {bin_type} bin (color: {bin_color})")
            expected_bins_context = f"""
IMPORTANT: You should ONLY identify bins that match the currently configured bins:
{chr(10).join(bin_descriptions)}

If a bin in the image doesn't match any of these, DO NOT include it in your response.
Only identify bins that correspond to the configured bins above.
"""
        
        prompt = f"""
You are inspecting a single photo of several waste/garbage bins. Your goal is to read the labels, signage, icons, and color cues so that a downstream system knows what each bin is meant for.
{expected_bins_context}
Instructions:
- Identify every distinct bin you see (usually 2-6). Work roughly left-to-right.
- For each bin, infer:
  - a short machine-friendly label,
  - a waste stream type (landfill, compost, recycling, paper, mixed, e-waste, unknown, etc.),
  - a short color description,
  - a very short summary of the signage (not full text, just the key idea),
  - a coarse position hint (leftmost, second_from_left, center, second_from_right, rightmost),
  - a confidence score between 0 and 1.

Keep the JSON compact. Do NOT include full paragraphs of text.

Respond with STRICT JSON using this compact schema (no markdown, no prose):

{{
  "bins": [
    {{
      "id": 0,                       // integer index, left-to-right
      "label": "left_black",         // short nickname
      "type": "landfill|compost|recycling|paper|mixed|e-waste|unknown",
      "pos": "leftmost|second_from_left|center|second_from_right|rightmost",
      "color": "short color phrase",
      "sign": "very short summary of main signage",
      "conf": 0.0
    }}
  ],
  "scene": "short summary of the overall lineup"
}}

Return valid JSON only.
        """.strip()

        response = self.classifier.model.generate_content(
            [prompt, image],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 512,  # smaller, keeps response compact
            },
        )

        response_text = response.text
        # keep raw text for debugging if needed
        self.classifier.last_raw_response = response_text

        parsed = self._safe_json_loads(response_text)
        result: Dict[str, Any] = {
            "bins": [],
            "scene": "",
            "raw_response": response_text,
        }

        if isinstance(parsed, dict):
            # Expecting keys: "bins" and "scene"
            result["bins"] = parsed.get("bins", []) or []
            result["scene"] = parsed.get("scene", "") or ""
            # Preserve any extra top-level fields under metadata (optional)
            for key, value in parsed.items():
                if key not in ("bins", "scene"):
                    result.setdefault("metadata", {})[key] = value
        elif isinstance(parsed, list):
            # Fallback: treat as a bare bins list
            result["bins"] = parsed
        else:
            result["metadata"] = {"parse_warning": "Gemini response was not valid JSON"}

        self._write_cache(result)
        return result
    
    def analyze_bins(self) -> Dict[str, Any]:
        """
        Run Gemini over the hardcoded image path and return parsed bin metadata.
        This prefers the cached result if it exists, so you don't repeatedly
        re-run the expensive vision call.
        """
        # Fast path: use cached result if available
        cached = self.load_cached_bins()
        if cached is not None:
            print(f"Using cached bin layout from {self.BIN_LAYOUT_OUTPUT_PATH.resolve()}")
            return cached
        
        image = self._ensure_pil_image(self.BIN_LAYOUT_IMAGE_PATH)
        
        # Use full resolution image for maximum accuracy
        # No resizing - send full image to Gemini Vision API for best text/details reading
        print(f"Using full resolution image: {image.width}x{image.height} for analysis.")
        
        return self._analyze_image(image)

    @classmethod
    def load_cached_bins(cls) -> Optional[Dict[str, Any]]:
        """
        Load previously saved bin layout data if available.
        """
        path = Path(cls.BIN_LAYOUT_OUTPUT_PATH)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def _write_cache(self, data: Dict[str, Any]):
        """
        Persist the parsed layout to disk so it can be reused without reprocessing the photo.
        """
        try:
            self.BIN_LAYOUT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.BIN_LAYOUT_OUTPUT_PATH.write_text(json.dumps(data, indent=2))
            print(f"Bin layout cached at {self.BIN_LAYOUT_OUTPUT_PATH.resolve()}")
        except Exception as exc:
            print(f"Warning: Failed to cache bin layout ({exc})")


def preview_bin_layout():
    """
    Convenience helper so you can run this module directly and inspect the parsed bins.
    """
    print("Initializing Gemini classifier...")
    from gemini_classifier import TrashClassifier  # local import to avoid circular dependency

    classifier = TrashClassifier()
    analyzer = BinLayoutAnalyzer(classifier)
    print("Running bin layout analysis...")
    result = analyzer.analyze_bins()
    bins = result.get("bins", [])
    print(f"Found {len(bins)} bins:\n")
    for idx, bin_info in enumerate(bins, 1):
        # New compact keys
        label = bin_info.get("label", f"bin_{bin_info.get('id', idx-1)}")
        bin_type = bin_info.get("type", "unknown")
        pos = bin_info.get("pos", "n/a")
        color = bin_info.get("color", "n/a")
        sign = bin_info.get("sign", "n/a")
        conf = bin_info.get("conf", None)

        line = f"{idx}. {label} -> {bin_type} | Pos: {pos} | Color: {color} | Sign: {sign}"
        if conf is not None:
            line += f" | Conf: {conf:.2f}" if isinstance(conf, (int, float)) else f" | Conf: {conf}"
        print(line)

    print("\nScene:")
    print(result.get("scene", "").strip() or "(none)")
    print(f"\nCached result at: {BinLayoutAnalyzer.BIN_LAYOUT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    preview_bin_layout()
