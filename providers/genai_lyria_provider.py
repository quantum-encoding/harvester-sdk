"""
Google AI Studio (GenAI) Lyria RealTime Music Provider
Real-time, streaming music generation with interactive control

Experimental model for instrumental music generation with:
- Real-time steering and control
- WebSocket streaming
- Interactive tempo, scale, and mood changes
"""

import logging
import asyncio
import time
import wave
import io
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from pathlib import Path
import os
from dataclasses import dataclass
from enum import Enum

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None
    
from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

# Musical scales enum
class Scale(Enum):
    """Musical scales supported by Lyria"""
    C_MAJOR_A_MINOR = "C_MAJOR_A_MINOR"
    D_FLAT_MAJOR_B_FLAT_MINOR = "D_FLAT_MAJOR_B_FLAT_MINOR"
    D_MAJOR_B_MINOR = "D_MAJOR_B_MINOR"
    E_FLAT_MAJOR_C_MINOR = "E_FLAT_MAJOR_C_MINOR"
    E_MAJOR_D_FLAT_MINOR = "E_MAJOR_D_FLAT_MINOR"
    F_MAJOR_D_MINOR = "F_MAJOR_D_MINOR"
    G_FLAT_MAJOR_E_FLAT_MINOR = "G_FLAT_MAJOR_E_FLAT_MINOR"
    G_MAJOR_E_MINOR = "G_MAJOR_E_MINOR"
    A_FLAT_MAJOR_F_MINOR = "A_FLAT_MAJOR_F_MINOR"
    A_MAJOR_G_FLAT_MINOR = "A_MAJOR_G_FLAT_MINOR"
    B_FLAT_MAJOR_G_MINOR = "B_FLAT_MAJOR_G_MINOR"
    B_MAJOR_A_FLAT_MINOR = "B_MAJOR_A_FLAT_MINOR"
    SCALE_UNSPECIFIED = "SCALE_UNSPECIFIED"

class MusicGenerationMode(Enum):
    """Generation modes for different musical focuses"""
    QUALITY = "QUALITY"
    DIVERSITY = "DIVERSITY"
    VOCALIZATION = "VOCALIZATION"

@dataclass
class MusicPrompt:
    """Weighted prompt for music generation"""
    text: str
    weight: float = 1.0

class GenAILyriaProvider(BaseProvider):
    """
    Google AI Studio Lyria RealTime Music Provider
    
    This provider uses Google's experimental Lyria model for real-time,
    interactive instrumental music generation with WebSocket streaming.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        if genai is None:
            raise ImportError(
                "Google GenAI not installed. Install with: pip install google-generativeai"
            )
        
        # API key from environment or config
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_GENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Google GenAI API key required. Set GOOGLE_GENAI_API_KEY environment variable")
        
        # Initialize client with v1alpha API for experimental features
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={'api_version': 'v1alpha'}
        )
        
        # Model configuration
        self.model = "models/lyria-realtime-exp"
        
        # Audio specifications
        self.audio_specs = {
            "format": "16-bit PCM",
            "sample_rate": 48000,
            "channels": 2,  # Stereo
            "bits_per_sample": 16
        }
        
        # Prompt categories for better organization
        self.prompt_categories = {
            "instruments": [
                "303 Acid Bass", "808 Hip Hop Beat", "Accordion", "Alto Saxophone",
                "Bagpipes", "Banjo", "Bass Clarinet", "Bongos", "Cello", "Djembe",
                "Flamenco Guitar", "Guitar", "Harmonica", "Harp", "Kalimba", "Koto",
                "Marimba", "Piano", "Rhodes Piano", "Sitar", "Synth Pads", "Tabla",
                "TR-909 Drum Machine", "Trumpet", "Vibraphone", "Violin"
            ],
            "genres": [
                "Acid Jazz", "Afrobeat", "Blues Rock", "Bossa Nova", "Chillout",
                "Deep House", "Disco Funk", "Drum & Bass", "Dubstep", "EDM",
                "Funk", "Hip Hop", "Indie Electronic", "Jazz Fusion", "Latin Jazz",
                "Lo-Fi Hip Hop", "Minimal Techno", "Neo-Soul", "Orchestral Score",
                "Piano Ballad", "Psychedelic Rock", "R&B", "Reggae", "Salsa",
                "Synthpop", "Techno", "Trance", "Trap Beat", "Trip Hop"
            ],
            "moods": [
                "Ambient", "Bright Tones", "Chill", "Danceable", "Dreamy",
                "Emotional", "Ethereal", "Experimental", "Funky", "Live Performance",
                "Lo-fi", "Psychedelic", "Upbeat", "Virtuoso", "Weird Noises"
            ]
        }
        
        logger.info(f"GenAI Lyria provider initialized for real-time music generation")
    
    async def generate_music_stream(
        self,
        prompts: List[MusicPrompt],
        duration_seconds: int = 30,
        output_path: Optional[str] = None,
        **config_params
    ) -> Dict[str, Any]:
        """
        Generate music with real-time streaming
        
        Args:
            prompts: List of weighted prompts for music generation
            duration_seconds: How long to generate music for
            output_path: Where to save the generated audio file
            **config_params: Music generation configuration:
                - bpm: Beats per minute (60-200)
                - temperature: Creativity (0.0-3.0, default 1.1)
                - guidance: Prompt adherence (0.0-6.0, default 4.0)
                - density: Note density (0.0-1.0)
                - brightness: Tonal quality (0.0-1.0)
                - scale: Musical scale (Scale enum)
                - mute_bass: Reduce bass (bool)
                - mute_drums: Reduce drums (bool)
                - only_bass_and_drums: Focus on rhythm section (bool)
                - music_generation_mode: QUALITY, DIVERSITY, or VOCALIZATION
        
        Returns:
            Dict with audio file path and metadata
        """
        audio_buffer = io.BytesIO()
        start_time = time.time()
        
        try:
            async with self.client.aio.live.music.connect(model=self.model) as session:
                # Set up audio receiver task
                audio_chunks = []
                
                async def receive_audio():
                    """Background task to receive and buffer audio chunks"""
                    try:
                        async for message in session.receive():
                            if hasattr(message, 'server_content'):
                                for chunk in message.server_content.audio_chunks:
                                    audio_chunks.append(chunk.data)
                                    
                                    # Check if we've reached duration
                                    elapsed = time.time() - start_time
                                    if elapsed >= duration_seconds:
                                        return
                    except Exception as e:
                        logger.error(f"Error receiving audio: {e}")
                
                # Start audio receiver in background
                receiver_task = asyncio.create_task(receive_audio())
                
                # Convert prompts to WeightedPrompt objects
                weighted_prompts = [
                    types.WeightedPrompt(text=p.text, weight=p.weight)
                    for p in prompts
                ]
                
                # Send initial prompts
                await session.set_weighted_prompts(prompts=weighted_prompts)
                
                # Build and send music generation config
                config = self._build_music_config(config_params)
                await session.set_music_generation_config(config=config)
                
                # Start music generation
                logger.info(f"Starting music generation for {duration_seconds} seconds")
                await session.play()
                
                # Wait for duration or task completion
                try:
                    await asyncio.wait_for(receiver_task, timeout=duration_seconds + 5)
                except asyncio.TimeoutError:
                    logger.info("Generation duration reached")
                
                # Stop generation
                await session.stop()
                
                # Process audio chunks into file
                if audio_chunks:
                    audio_data = b''.join(audio_chunks)
                    
                    # Save as WAV file
                    if not output_path:
                        output_path = f"lyria_music_{int(time.time())}.wav"
                    
                    self._save_audio_as_wav(audio_data, output_path)
                    
                    generation_time = time.time() - start_time
                    logger.info(f"Music generated successfully in {generation_time:.1f} seconds")
                    
                    return {
                        "success": True,
                        "audio_path": output_path,
                        "duration": duration_seconds,
                        "prompts": [p.text for p in prompts],
                        "config": config_params,
                        "generation_time": generation_time,
                        "audio_specs": self.audio_specs
                    }
                else:
                    raise Exception("No audio data received")
                    
        except Exception as e:
            logger.error(f"Music generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompts": [p.text for p in prompts]
            }
    
    async def generate_interactive_session(
        self,
        initial_prompts: List[MusicPrompt],
        session_callback: callable,
        **config_params
    ) -> Dict[str, Any]:
        """
        Create an interactive music generation session
        
        Args:
            initial_prompts: Starting prompts for music
            session_callback: Async function that receives the session for interaction
            **config_params: Initial music generation configuration
        
        Returns:
            Dict with session results
        
        Example:
            async def interact_with_music(session):
                # Let initial music play for 10 seconds
                await asyncio.sleep(10)
                
                # Change to jazz
                await session.set_weighted_prompts([
                    types.WeightedPrompt(text="Smooth Jazz", weight=2.0)
                ])
                
                # Let jazz play for 10 seconds
                await asyncio.sleep(10)
                
                # Add piano
                await session.set_weighted_prompts([
                    types.WeightedPrompt(text="Smooth Jazz", weight=1.0),
                    types.WeightedPrompt(text="Piano Solo", weight=1.5)
                ])
        """
        try:
            async with self.client.aio.live.music.connect(model=self.model) as session:
                # Convert prompts
                weighted_prompts = [
                    types.WeightedPrompt(text=p.text, weight=p.weight)
                    for p in prompts
                ]
                
                # Set initial configuration
                await session.set_weighted_prompts(prompts=weighted_prompts)
                config = self._build_music_config(config_params)
                await session.set_music_generation_config(config=config)
                
                # Start playing
                await session.play()
                
                # Hand over control to callback
                result = await session_callback(session)
                
                # Stop playing
                await session.stop()
                
                return {
                    "success": True,
                    "session_result": result,
                    "initial_prompts": [p.text for p in initial_prompts]
                }
                
        except Exception as e:
            logger.error(f"Interactive session failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_with_transitions(
        self,
        prompt_sequence: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate music with smooth transitions between different prompts
        
        Args:
            prompt_sequence: List of prompt configurations with timing:
                [
                    {
                        "prompts": [MusicPrompt("Ambient", 1.0)],
                        "duration": 10,
                        "config": {"bpm": 90}
                    },
                    {
                        "prompts": [MusicPrompt("Techno", 1.0)],
                        "duration": 10,
                        "config": {"bpm": 128},
                        "transition_time": 2  # Crossfade duration
                    }
                ]
            output_path: Where to save the generated audio
        
        Returns:
            Dict with audio file path and metadata
        """
        audio_chunks = []
        total_duration = sum(step["duration"] for step in prompt_sequence)
        
        try:
            async with self.client.aio.live.music.connect(model=self.model) as session:
                # Audio receiver
                async def receive_audio():
                    try:
                        async for message in session.receive():
                            if hasattr(message, 'server_content'):
                                for chunk in message.server_content.audio_chunks:
                                    audio_chunks.append(chunk.data)
                    except Exception as e:
                        logger.error(f"Error receiving audio: {e}")
                
                # Start receiver
                receiver_task = asyncio.create_task(receive_audio())
                
                # Process each step in sequence
                for i, step in enumerate(prompt_sequence):
                    logger.info(f"Processing step {i+1}/{len(prompt_sequence)}")
                    
                    # Convert prompts
                    weighted_prompts = [
                        types.WeightedPrompt(text=p.text, weight=p.weight)
                        for p in step["prompts"]
                    ]
                    
                    # Handle transitions with weight interpolation
                    if i > 0 and "transition_time" in step:
                        transition_time = step["transition_time"]
                        steps = 5  # Number of interpolation steps
                        
                        for t in range(steps):
                            weight_old = 1.0 - (t / steps)
                            weight_new = t / steps
                            
                            # Mix old and new prompts
                            mixed_prompts = []
                            
                            # Add fading old prompts
                            if i > 0:
                                old_prompts = prompt_sequence[i-1]["prompts"]
                                for p in old_prompts:
                                    mixed_prompts.append(
                                        types.WeightedPrompt(
                                            text=p.text,
                                            weight=p.weight * weight_old
                                        )
                                    )
                            
                            # Add fading in new prompts
                            for p in step["prompts"]:
                                mixed_prompts.append(
                                    types.WeightedPrompt(
                                        text=p.text,
                                        weight=p.weight * weight_new
                                    )
                                )
                            
                            await session.set_weighted_prompts(prompts=mixed_prompts)
                            await asyncio.sleep(transition_time / steps)
                    
                    # Set final prompts for this step
                    await session.set_weighted_prompts(prompts=weighted_prompts)
                    
                    # Update config if provided
                    if "config" in step:
                        config = self._build_music_config(step["config"])
                        await session.set_music_generation_config(config=config)
                        
                        # Reset context for BPM or scale changes
                        if "bpm" in step["config"] or "scale" in step["config"]:
                            await session.reset_context()
                    
                    # Start playing if first step
                    if i == 0:
                        await session.play()
                    
                    # Wait for step duration
                    await asyncio.sleep(step["duration"])
                
                # Stop generation
                await session.stop()
                
                # Cancel receiver
                receiver_task.cancel()
                
                # Save audio
                if audio_chunks:
                    audio_data = b''.join(audio_chunks)
                    
                    if not output_path:
                        output_path = f"lyria_sequence_{int(time.time())}.wav"
                    
                    self._save_audio_as_wav(audio_data, output_path)
                    
                    return {
                        "success": True,
                        "audio_path": output_path,
                        "total_duration": total_duration,
                        "steps": len(prompt_sequence),
                        "audio_specs": self.audio_specs
                    }
                    
        except Exception as e:
            logger.error(f"Transition generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_music_config(self, params: Dict[str, Any]) -> Any:
        """Build LiveMusicGenerationConfig from parameters"""
        config_dict = {}
        
        # Map parameters with defaults
        param_mapping = {
            "bpm": ("bpm", None),  # 60-200
            "temperature": ("temperature", 1.1),  # 0.0-3.0
            "guidance": ("guidance", 4.0),  # 0.0-6.0
            "density": ("density", None),  # 0.0-1.0
            "brightness": ("brightness", None),  # 0.0-1.0
            "mute_bass": ("mute_bass", False),
            "mute_drums": ("mute_drums", False),
            "only_bass_and_drums": ("only_bass_and_drums", False),
            "top_k": ("top_k", 40),  # 1-1000
            "seed": ("seed", None)  # 0-2147483647
        }
        
        for param_key, (config_key, default) in param_mapping.items():
            if param_key in params:
                config_dict[config_key] = params[param_key]
            elif default is not None:
                config_dict[config_key] = default
        
        # Handle scale enum
        if "scale" in params:
            scale_value = params["scale"]
            if isinstance(scale_value, str):
                config_dict["scale"] = getattr(types.Scale, scale_value, types.Scale.SCALE_UNSPECIFIED)
            else:
                config_dict["scale"] = scale_value
        
        # Handle music generation mode enum
        if "music_generation_mode" in params:
            mode_value = params["music_generation_mode"]
            if isinstance(mode_value, str):
                config_dict["music_generation_mode"] = getattr(
                    types.MusicGenerationMode,
                    mode_value,
                    types.MusicGenerationMode.QUALITY
                )
            else:
                config_dict["music_generation_mode"] = mode_value
        
        return types.LiveMusicGenerationConfig(**config_dict)
    
    def _save_audio_as_wav(self, audio_data: bytes, output_path: str):
        """Save raw PCM audio data as WAV file"""
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(self.audio_specs["channels"])
            wav_file.setsampwidth(self.audio_specs["bits_per_sample"] // 8)
            wav_file.setframerate(self.audio_specs["sample_rate"])
            wav_file.writeframes(audio_data)
    
    def get_prompt_suggestions(self, category: str = None) -> Dict[str, List[str]]:
        """Get suggested prompts by category"""
        if category:
            return {category: self.prompt_categories.get(category, [])}
        return self.prompt_categories
    
    def create_prompt(self, text: str, weight: float = 1.0) -> MusicPrompt:
        """Helper to create a weighted music prompt"""
        return MusicPrompt(text=text, weight=weight)
    
    async def complete(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Compatibility method for BaseProvider interface
        Generates music and returns the file path
        """
        # Parse simple prompt into weighted prompts
        prompts = [self.create_prompt(prompt)]
        
        result = await self.generate_music_stream(
            prompts=prompts,
            duration_seconds=kwargs.get("duration", 30),
            **kwargs
        )
        
        if result["success"]:
            return result["audio_path"]
        else:
            raise Exception(f"Music generation failed: {result.get('error', 'Unknown error')}")


# Example usage
if __name__ == "__main__":
    async def test_lyria():
        config = {
            "api_key": "YOUR_API_KEY"  # Or set GOOGLE_GENAI_API_KEY env var
        }
        
        provider = GenAILyriaProvider(config)
        
        # Test 1: Simple music generation
        result = await provider.generate_music_stream(
            prompts=[
                provider.create_prompt("Minimal Techno", 1.0),
                provider.create_prompt("303 Acid Bass", 0.5)
            ],
            duration_seconds=20,
            bpm=128,
            temperature=1.0,
            output_path="techno_test.wav"
        )
        
        if result["success"]:
            print(f"✅ Generated music: {result['audio_path']}")
            print(f"   Duration: {result['duration']} seconds")
            print(f"   Format: {result['audio_specs']}")
        
        # Test 2: Music with transitions
        sequence = [
            {
                "prompts": [provider.create_prompt("Ambient", 1.0)],
                "duration": 10,
                "config": {"bpm": 90, "brightness": 0.3}
            },
            {
                "prompts": [
                    provider.create_prompt("Deep House", 1.0),
                    provider.create_prompt("Synth Pads", 0.5)
                ],
                "duration": 10,
                "config": {"bpm": 120, "brightness": 0.7},
                "transition_time": 3
            },
            {
                "prompts": [
                    provider.create_prompt("Techno", 1.0),
                    provider.create_prompt("TR-909 Drum Machine", 0.8)
                ],
                "duration": 10,
                "config": {"bpm": 128, "brightness": 0.9},
                "transition_time": 2
            }
        ]
        
        result2 = await provider.generate_with_transitions(
            prompt_sequence=sequence,
            output_path="music_journey.wav"
        )
        
        if result2["success"]:
            print(f"✅ Generated music journey: {result2['audio_path']}")
            print(f"   Total duration: {result2['total_duration']} seconds")
            print(f"   Steps: {result2['steps']}")
    
    # Run test
    import asyncio
    asyncio.run(test_lyria())