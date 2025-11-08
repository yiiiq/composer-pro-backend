"""
Inference utilities for ML models.

Handle model predictions and post-processing.
"""
from typing import Any, Dict, List, Union, Optional
import logging
import torch
import torchaudio
from pathlib import Path
import tempfile
import numpy as np

logger = logging.getLogger(__name__)


class MusicGenInference:
    """Inference class for MusicGen model."""
    
    def __init__(self, model, mbd=None):
        self.model = model
        self.mbd = mbd
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def generate_with_melody(
        self,
        description: str,
        melody_audio_path: Optional[str] = None,
        melody_waveform: Optional[torch.Tensor] = None,
        melody_sample_rate: Optional[int] = None,
        duration: int = 30,
        use_diffusion: bool = False
    ) -> Dict[str, Any]:
        """
        Generate music with melody conditioning.
        
        Args:
            description: Text description of the music to generate
            melody_audio_path: Path to melody audio file
            melody_waveform: Melody waveform tensor (alternative to audio_path)
            melody_sample_rate: Sample rate of melody
            duration: Duration in seconds
            use_diffusion: Use MultiBandDiffusion decoder
            
        Returns:
            Dictionary containing generated audio and metadata
        """
        try:
            # Update generation duration
            self.model.set_generation_params(duration=duration)
            
            # Load melody if path provided
            if melody_audio_path:
                melody_waveform, melody_sample_rate = torchaudio.load(melody_audio_path)
                logger.info(f"Loaded melody from {melody_audio_path}")
            
            if melody_waveform is None:
                raise ValueError("Either melody_audio_path or melody_waveform must be provided")
            
            # Ensure melody is on the correct device
            melody_waveform = melody_waveform.to(self.device)
            
            logger.info(f"Generating music with description: {description}")
            
            # Generate with melody conditioning
            output = self.model.generate_with_chroma(
                descriptions=[description],
                melody_wavs=melody_waveform,
                melody_sample_rate=melody_sample_rate,
                progress=True,
                return_tokens=True
            )
            
            result = {
                "audio": output[0],  # Generated audio waveforms
                "tokens": output[1] if len(output) > 1 else None,
                "sample_rate": self.model.sample_rate,
                "duration": duration,
                "description": description,
                "use_diffusion": False
            }
            
            # Apply diffusion decoder if requested
            if use_diffusion and self.mbd is not None:
                logger.info("Applying MultiBandDiffusion decoder")
                diffusion_output = self.mbd.tokens_to_wav(output[1])
                result["audio_diffusion"] = diffusion_output
                result["use_diffusion"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error during music generation: {e}")
            raise
    
    def generate_unconditional(
        self,
        description: str,
        duration: int = 30,
        use_diffusion: bool = False
    ) -> Dict[str, Any]:
        """
        Generate music without melody conditioning.
        
        Args:
            description: Text description of the music to generate
            duration: Duration in seconds
            use_diffusion: Use MultiBandDiffusion decoder
            
        Returns:
            Dictionary containing generated audio and metadata
        """
        try:
            # Update generation duration
            self.model.set_generation_params(duration=duration)
            
            logger.info(f"Generating unconditional music: {description}")
            
            # Generate music
            output = self.model.generate(
                descriptions=[description],
                progress=True,
                return_tokens=True
            )
            
            result = {
                "audio": output[0],
                "tokens": output[1] if len(output) > 1 else None,
                "sample_rate": self.model.sample_rate,
                "duration": duration,
                "description": description,
                "use_diffusion": False
            }
            
            # Apply diffusion decoder if requested
            if use_diffusion and self.mbd is not None:
                logger.info("Applying MultiBandDiffusion decoder")
                diffusion_output = self.mbd.tokens_to_wav(output[1])
                result["audio_diffusion"] = diffusion_output
                result["use_diffusion"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error during music generation: {e}")
            raise
    
    def save_audio(self, audio_tensor: torch.Tensor, output_path: str, sample_rate: int = 32000):
        """
        Save audio tensor to file.
        
        Args:
            audio_tensor: Audio waveform tensor
            output_path: Path to save audio
            sample_rate: Audio sample rate
        """
        from audiocraft.data.audio import audio_write
        
        # Remove batch and channel dimensions if needed
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)
        
        # Get stem name (without extension)
        stem_name = Path(output_path).stem
        
        # Save audio
        audio_write(
            stem_name=stem_name,
            wav=audio_tensor.cpu(),
            sample_rate=sample_rate,
            format="wav"
        )
        
        logger.info(f"Audio saved to {output_path}")


class ModelInference:
    """Base class for model inference."""
    
    def __init__(self, model: Any):
        self.model = model
    
    def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess input data before prediction.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Preprocessed data ready for model
        """
        # Add your preprocessing logic here
        return input_data
    
    def predict(self, input_data: Any) -> Any:
        """
        Run model prediction.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Model predictions
        """
        processed_data = self.preprocess(input_data)
        
        # Add your prediction logic here
        # Example:
        # predictions = self.model.predict(processed_data)
        # return predictions
        
        raise NotImplementedError("Implement prediction logic for your specific model")
    
    def postprocess(self, predictions: Any) -> Dict[str, Any]:
        """
        Post-process model predictions.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Formatted prediction results
        """
        # Add your post-processing logic here
        return {"predictions": predictions}
    
    def run_inference(self, input_data: Any) -> Dict[str, Any]:
        """
        Complete inference pipeline: preprocess -> predict -> postprocess.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Final prediction results
        """
        predictions = self.predict(input_data)
        results = self.postprocess(predictions)
        return results
