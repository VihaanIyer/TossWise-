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

    def analyze_bins(self) -> Dict[str, Any]:
        """
        Run Gemini over the hardcoded image path and return parsed bin metadata.
        """
        prompt = """
You are inspecting a single photo of several waste/garbage bins. Your goal is to read the labels, signage, icons, and color cues so that a downstream system knows what each bin is meant for.

Instructions:
- Identify every distinct bin you see (usually 2-6). Work left-to-right when describing positions.
- For each bin, capture the exact text on the bin ("signage_text"), describe the main color(s), list any icons/symbols, and infer its likely waste stream classification ("bin_type_guess": recycling, compost, landfill, e-waste, bottles, paper, mixed, unknown, etc.).
- If text is partially readable, include what you can and note missing parts.
- If multiple labels/icons exist on one bin, include them all.
- When you are unsure about the class, mark bin_type_guess as "unknown" but still provide clues (e.g., "Blue bin with recycling arrows").

Respond with STRICT JSON using this schema (no markdown, no prose):
{
  "bins": [
    {
      "bin_label": "short nickname you assign (e.g., Left Blue Bin)",
      "bin_type_guess": "recycling|compost|landfill|paper|mixed|e-waste|unknown",
      "bin_color": "text description of colors",
      "signage_text": "verbatim text you can read (or null)",
      "icon_description": "recycling triangle, food scraps icon, etc.",
      "position_hint": "leftmost|second from left|center|etc.",
      "additional_notes": "anything else important about this bin",
      "confidence": 0.0-1.0
    }
  ],
  "scene_notes": "overall context or observations about the lineup"
}

Return valid JSON only.
        """.strip()

        image = self._ensure_pil_image(self.BIN_LAYOUT_IMAGE_PATH)
        response = self.classifier.model.generate_content(
            [prompt, image],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 2048,
            },
        )

        response_text = response.text
        self.classifier.last_raw_response = response_text

        parsed = self._safe_json_loads(response_text)
        result: Dict[str, Any] = {
            "bins": [],
            "scene_notes": "",
            "raw_response": response_text,
        }

        if isinstance(parsed, dict):
            result["bins"] = parsed.get("bins", []) or []
            result["scene_notes"] = parsed.get("scene_notes", "")
            for key, value in parsed.items():
                if key not in ("bins", "scene_notes"):
                    result.setdefault("metadata", {})[key] = value
        elif isinstance(parsed, list):
            result["bins"] = parsed
        else:
            result["metadata"] = {"parse_warning": "Gemini response was not valid JSON"}

        self._write_cache(result)
        return result
    
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
        label = bin_info.get("bin_label", f"Bin {idx}")
        bin_type = bin_info.get("bin_type_guess", "unknown")
        color = bin_info.get("bin_color", "n/a")
        signage = bin_info.get("signage_text", "n/a")
        print(f"{idx}. {label} -> {bin_type} | Color: {color} | Signage: {signage}")
        if bin_info.get("additional_notes"):
            print(f"   Notes: {bin_info['additional_notes']}")
    print("\nScene notes:")
    print(result.get("scene_notes", "").strip() or "(none)")
    print(f"\nCached result at: {BinLayoutAnalyzer.BIN_LAYOUT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    preview_bin_layout()
