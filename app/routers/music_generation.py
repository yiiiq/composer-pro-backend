"""
Music generation endpoints using MusicGen.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import Optional
import logging
import tempfile
import os
from pathlib import Path
import torch
import torchaudio
from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.data.audio import audio_write

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model instances
model = None
mbd = None

# Output directory for generated audio
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

USE_DIFFUSION_DECODER = False


@router.post("/generate")
async def generate_music(
    description: str = Form(..., description="Text description of music to generate"),
    duration: int = Form(default=30, description="Duration in seconds"),
    melody_file: UploadFile = File(..., description="Melody audio file for conditioning")
):
    """
    Generate music using MusicGen model with melody conditioning.
    
    - **description**: Text prompt (e.g., "Generate jazz guitar melody with happy emotion")
    - **duration**: Length of generated audio in seconds
    - **melody_file**: Audio file to condition generation (WAV, MP3, etc.)
    """
    global model, mbd
    
    try:
        # Load model if not already loaded
        if model is None:
            logger.info("Loading MusicGen model: facebook/musicgen-melody")
            model = MusicGen.get_pretrained('facebook/musicgen-melody')
            model.set_generation_params(duration=duration)
            
            if USE_DIFFUSION_DECODER:
                logger.info("Loading MultiBandDiffusion decoder")
                mbd = MultiBandDiffusion.get_mbd_musicgen()
            
            logger.info("MusicGen model loaded successfully")
        else:
            # Update duration for this generation
            model.set_generation_params(duration=duration)
        
        # Save uploaded melody to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_melody:
            content = await melody_file.read()
            temp_melody.write(content)
            melody_path = temp_melody.name
            logger.info(f"Saved melody file to {melody_path}")
        
        # Load melody
        melody_waveform, sr = torchaudio.load(melody_path)
        logger.info(f"Loaded melody: shape={melody_waveform.shape}, sr={sr}")
        
        # Generate music with melody conditioning
        logger.info(f"Generating music: {description}")
        output = model.generate_with_chroma(
            descriptions=[description],
            melody_wavs=melody_waveform,
            melody_sample_rate=sr,
            progress=True,
            return_tokens=True
        )
        
        # Save generated audio
        import time
        timestamp = int(time.time())
        output_filename = f"extended_output_{timestamp}"
        
        samples = output[0]  # shape [batch, channels, time]
        
        # Save first sample
        wav = samples[0].squeeze(0)  # shape [channels, time]
        audio_write(
            stem_name=str(OUTPUT_DIR / output_filename),
            wav=wav.cpu(),
            sample_rate=32000,
            format="wav"
        )
        
        # Apply diffusion decoder if enabled
        if USE_DIFFUSION_DECODER and mbd is not None:
            logger.info("Applying MultiBandDiffusion decoder")
            out_diffusion = mbd.tokens_to_wav(output[1])
            audio_write(
                stem_name=str(OUTPUT_DIR / f"{output_filename}_diffusion"),
                wav=out_diffusion[0].squeeze(0).cpu(),
                sample_rate=32000,
                format="wav"
            )
        
        # Clean up temporary melody file
        if os.path.exists(melody_path):
            os.unlink(melody_path)
        
        logger.info(f"Music generation completed: {output_filename}.wav")
        
        return {
            "success": True,
            "message": "Music generated successfully",
            "audio_file": f"{output_filename}.wav",
            "duration": duration,
            "sample_rate": 32000,
            "description": description
        }
        
    except Exception as e:
        logger.error(f"Error generating music: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Music generation failed: {str(e)}")


@router.get("/download/{filename}")
async def download_audio(filename: str):
    """Download a generated audio file."""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )
