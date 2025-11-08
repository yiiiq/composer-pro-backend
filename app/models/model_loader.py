"""
Model loader utilities.

Load and manage ML models from disk or model registry.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import torch

logger = logging.getLogger(__name__)

# Directory to store model files
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


class MusicGenLoader:
    """Loader for MusicGen model."""
    
    def __init__(self):
        self._model = None
        self._mbd = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_musicgen(
        self, 
        model_name: str = 'facebook/musicgen-melody',
        use_diffusion_decoder: bool = False
    ):
        """
        Load MusicGen model.
        
        Args:
            model_name: MusicGen model to load (facebook/musicgen-melody, facebook/musicgen-medium, etc.)
            use_diffusion_decoder: Whether to use MultiBandDiffusion decoder
            
        Returns:
            Loaded MusicGen model
        """
        if self._model is not None:
            logger.info("MusicGen model already loaded, returning cached version")
            return self._model, self._mbd
        
        try:
            from audiocraft.models import MusicGen
            from audiocraft.models import MultiBandDiffusion
            
            logger.info(f"Loading MusicGen model: {model_name}")
            self._model = MusicGen.get_pretrained(model_name)
            
            # Set default generation parameters
            self._model.set_generation_params(duration=30)
            
            if use_diffusion_decoder:
                logger.info("Loading MultiBandDiffusion decoder")
                self._mbd = MultiBandDiffusion.get_mbd_musicgen()
            
            logger.info(f"MusicGen model loaded successfully on {self.device}")
            return self._model, self._mbd
            
        except Exception as e:
            logger.error(f"Failed to load MusicGen model: {e}")
            raise
    
    def get_model(self):
        """Get the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_musicgen() first.")
        return self._model, self._mbd
    
    def update_generation_params(self, **kwargs):
        """
        Update generation parameters.
        
        Args:
            duration: Generation duration in seconds
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            temperature: Sampling temperature
            cfg_coef: Classifier-free guidance coefficient
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_musicgen() first.")
        
        self._model.set_generation_params(**kwargs)
        logger.info(f"Updated generation params: {kwargs}")


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
