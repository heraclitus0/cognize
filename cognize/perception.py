"""
Perception Layer for Cognize
============================
Converts raw multi-modal inputs (text, images, sensors) into
normalized evidence vectors usable by EpistemicState instances.
"""

from typing import Any, Dict
import numpy as np

class Perception:
    def __init__(self, text_encoder=None, image_encoder=None, sensor_fusion_fn=None):
        """
        Initialize the perception layer.
        Args:
            text_encoder: Callable that takes text and returns vector.
            image_encoder: Callable that takes image and returns vector.
            sensor_fusion_fn: Callable that takes sensor dict and returns vector.
        """
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.sensor_fusion_fn = sensor_fusion_fn

    def process(self, inputs: Dict[str, Any]) -> np.ndarray:
        """
        Process multi-modal inputs into a single evidence vector.
        Supported keys: 'text', 'image', 'sensor'
        """
        vectors = []

        if "text" in inputs and self.text_encoder:
            vectors.append(self.text_encoder(inputs["text"]))

        if "image" in inputs and self.image_encoder:
            vectors.append(self.image_encoder(inputs["image"]))

        if "sensor" in inputs and self.sensor_fusion_fn:
            vectors.append(self.sensor_fusion_fn(inputs["sensor"]))

        if not vectors:
            raise ValueError("No valid inputs processed.")

        # Combine vectors into one evidence signal
        return np.mean(vectors, axis=0)
