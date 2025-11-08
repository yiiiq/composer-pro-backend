"""
Model loader utilities.

Load and manage ML models from disk or model registry.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Directory to store model files
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


class ModelLoader:
    """Base class for loading ML models."""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
    
    def load_model(self, model_name: str, model_path: Optional[str] = None) -> Any:
        """
        Load a model from disk.
        
        Args:
            model_name: Name/identifier for the model
            model_path: Path to the model file
            
        Returns:
            Loaded model object
        """
        if model_name in self._models:
            logger.info(f"Model '{model_name}' already loaded, returning cached version")
            return self._models[model_name]
        
        if model_path is None:
            model_path = MODEL_DIR / f"{model_name}.pkl"
        
        # Add your model loading logic here
        # Example for sklearn models:
        # import joblib
        # model = joblib.load(model_path)
        
        logger.info(f"Model '{model_name}' loaded from {model_path}")
        # self._models[model_name] = model
        # return model
        
        raise NotImplementedError("Implement model loading logic for your specific use case")
    
    def get_model(self, model_name: str) -> Any:
        """Get a loaded model by name."""
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not loaded")
        return self._models[model_name]
    
    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Model '{model_name}' unloaded")


# Global model loader instance
model_loader = ModelLoader()
