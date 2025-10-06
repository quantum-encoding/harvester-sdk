"""
Google AI Studio (GenAI) Text-to-Speech Provider
Native TTS generation with single and multi-speaker capabilities

Models available:
- gemini-2.5-flash-preview-tts
- gemini-2.5-pro-preview-tts
"""

import logging
import asyncio
import wave
import io
from typing import Any, Dict, List, Optional, Union
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

# Voice options with characteristics
VOICE_OPTIONS = {
    # Bright voices
    "Zephyr": "Bright",
    "Puck": "Upbeat",
    "Autonoe": "Bright",
    
    # Firm voices
    "Kore": "Firm",
    "Orus": "Firm",
    "Alnilam": "Firm",
    
    # Informative voices
    "Charon": "Informative",
    "Rasalgethi": "Informative",
    
    # Easy-going voices
    "Callirrhoe": "Easy-going",
    "Umbriel": "Easy-going",
    
    # Smooth voices
    "Algieba": "Smooth",
    "Despina": "Smooth",
    
    # Clear voices
    "Iapetus": "Clear",
    "Erinome": "Clear",
    
    # Youthful/Energetic
    "Leda": "Youthful",
    "Fenrir": "Excitable",
    "Laomedeia": "Upbeat",
    "Sadachbia": "Lively",
    
    # Mature/Professional
    "Gacrux": "Mature",
    "Sadaltager": "Knowledgeable",
    "Pulcherrima": "Forward",
    "Schedar": "Even",
    
    # Gentle/Soft
    "Achernar": "Soft",
    "Vindemiatrix": "Gentle",
    "Aoede": "Breezy",
    
    # Unique characteristics
    "Enceladus": "Breathy",
    "Algenib": "Gravelly",
    "Achird": "Friendly",
    "Zubenelgenubi": "Casual",
    "Sulafat": "Warm"
}

@dataclass
class Speaker:
    """Speaker configuration for multi-speaker TTS"""
    name: str
    voice: str
    style: Optional[str] = None  # e.g., "tired", "excited", "whisper"

class GenAITTSProvider(BaseProvider):
    """
    Google AI Studio Text-to-Speech Provider
    
    This provider uses Gemini 2.5 models for high-quality,
    controllable text-to-speech generation with style control.
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
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        
        # Model configurations
        self.model_configs = {
            "gemini-2.5-flash-preview-tts": {
                "name": "Gemini 2.5 Flash TTS (Preview)",
                "supports_single_speaker": True,
                "supports_multi_speaker": True,
                "max_speakers": 2,
                "context_window": 32000,  # 32k tokens
                "languages": 24,
                "version": "preview"
            },
            "gemini-2.5-pro-preview-tts": {
                "name": "Gemini 2.5 Pro TTS (Preview)",
                "supports_single_speaker": True,
                "supports_multi_speaker": True,
                "max_speakers": 2,
                "context_window": 32000,
                "languages": 24,
                "version": "preview"
            }
        }
        
        self.default_model = config.get("default_model", "gemini-2.5-flash-preview-tts")
        
        # Audio specifications
        self.audio_specs = {
            "format": "PCM",
            "sample_rate": 24000,  # 24kHz
            "channels": 1,  # Mono
            "sample_width": 2  # 16-bit
        }
        
        # Supported languages
        self.supported_languages = {
            "ar-EG": "Arabic (Egyptian)",
            "en-US": "English (US)",
            "de-DE": "German",
            "es-US": "Spanish (US)",
            "fr-FR": "French",
            "hi-IN": "Hindi",
            "id-ID": "Indonesian",
            "it-IT": "Italian",
            "ja-JP": "Japanese",
            "ko-KR": "Korean",
            "pt-BR": "Portuguese (Brazil)",
            "ru-RU": "Russian",
            "nl-NL": "Dutch",
            "pl-PL": "Polish",
            "th-TH": "Thai",
            "tr-TR": "Turkish",
            "vi-VN": "Vietnamese",
            "ro-RO": "Romanian",
            "uk-UA": "Ukrainian",
            "bn-BD": "Bengali",
            "en-IN": "English (India)",
            "mr-IN": "Marathi",
            "ta-IN": "Tamil",
            "te-IN": "Telugu"
        }
        
        logger.info(f"GenAI TTS provider initialized with {len(self.model_configs)} models")
    
    async def generate_speech(
        self,
        text: str,
        voice: str = "Kore",
        model: str = None,
        style_prompt: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate single-speaker speech from text
        
        Args:
            text: Text to convert to speech
            voice: Voice name from VOICE_OPTIONS
            model: Model to use
            style_prompt: Natural language style control
            output_path: Where to save the audio file
            **kwargs: Additional parameters
        
        Returns:
            Dict with audio file path and metadata
        """
        model = model or self.default_model
        
        if model not in self.model_configs:
            logger.warning(f"Unknown model {model}, using default")
            model = self.default_model
        
        # Validate voice
        if voice not in VOICE_OPTIONS:
            logger.warning(f"Unknown voice {voice}, using Kore")
            voice = "Kore"
        
        # Build prompt with style control
        if style_prompt:
            prompt = f"{style_prompt}: {text}"
        else:
            prompt = text
        
        try:
            # Generate speech
            logger.info(f"Generating speech with voice '{voice}' ({VOICE_OPTIONS[voice]})")
            
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    )
                )
            )
            
            # Extract audio data
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Save to file
            if not output_path:
                output_path = f"tts_{voice.lower()}_{int(asyncio.get_event_loop().time())}.wav"
            
            self._save_audio_as_wav(audio_data, output_path)
            
            logger.info(f"Speech generated successfully: {output_path}")
            
            return {
                "success": True,
                "audio_path": output_path,
                "voice": voice,
                "voice_style": VOICE_OPTIONS[voice],
                "text_length": len(text),
                "model": model,
                "audio_specs": self.audio_specs
            }
            
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "voice": voice,
                "model": model
            }
    
    async def generate_multi_speaker(
        self,
        dialogue: Union[str, List[Dict[str, str]]],
        speakers: List[Speaker],
        model: str = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate multi-speaker speech (up to 2 speakers)
        
        Args:
            dialogue: Either formatted text or list of speaker/text pairs
            speakers: List of Speaker configurations (max 2)
            model: Model to use
            output_path: Where to save the audio file
            **kwargs: Additional parameters
        
        Returns:
            Dict with audio file path and metadata
        """
        model = model or self.default_model
        
        if model not in self.model_configs:
            logger.warning(f"Unknown model {model}, using default")
            model = self.default_model
        
        # Validate speakers
        if len(speakers) > 2:
            logger.warning("Maximum 2 speakers supported, using first 2")
            speakers = speakers[:2]
        
        # Format dialogue if needed
        if isinstance(dialogue, list):
            dialogue_text = self._format_dialogue(dialogue, speakers)
        else:
            dialogue_text = dialogue
        
        try:
            # Build speaker voice configs
            speaker_configs = []
            for speaker in speakers:
                if speaker.voice not in VOICE_OPTIONS:
                    logger.warning(f"Unknown voice {speaker.voice}, using Kore")
                    speaker.voice = "Kore"
                
                speaker_configs.append(
                    types.SpeakerVoiceConfig(
                        speaker=speaker.name,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=speaker.voice
                            )
                        )
                    )
                )
            
            logger.info(f"Generating multi-speaker dialogue with {len(speakers)} speakers")
            
            # Generate speech
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=dialogue_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=speaker_configs
                        )
                    )
                )
            )
            
            # Extract audio data
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Save to file
            if not output_path:
                output_path = f"tts_dialogue_{int(asyncio.get_event_loop().time())}.wav"
            
            self._save_audio_as_wav(audio_data, output_path)
            
            logger.info(f"Multi-speaker dialogue generated: {output_path}")
            
            return {
                "success": True,
                "audio_path": output_path,
                "speakers": [
                    {
                        "name": s.name,
                        "voice": s.voice,
                        "style": VOICE_OPTIONS[s.voice]
                    } for s in speakers
                ],
                "model": model,
                "audio_specs": self.audio_specs
            }
            
        except Exception as e:
            logger.error(f"Multi-speaker generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model
            }
    
    async def generate_styled_speech(
        self,
        text: str,
        voice: str = "Kore",
        emotion: Optional[str] = None,
        pace: Optional[str] = None,
        tone: Optional[str] = None,
        model: str = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate speech with specific style controls
        
        Args:
            text: Text to convert to speech
            voice: Voice name from VOICE_OPTIONS
            emotion: Emotional style (e.g., "happy", "sad", "excited")
            pace: Speaking pace (e.g., "slow", "fast", "normal")
            tone: Voice tone (e.g., "whisper", "shout", "formal")
            model: Model to use
            output_path: Where to save the audio file
            **kwargs: Additional parameters
        
        Returns:
            Dict with audio file path and metadata
        """
        # Build style prompt
        style_parts = []
        
        if emotion:
            style_parts.append(f"in a {emotion} voice")
        
        if pace:
            style_parts.append(f"speaking {pace}")
        
        if tone:
            if tone == "whisper":
                style_parts.append("in a whisper")
            elif tone == "shout":
                style_parts.append("shouting")
            else:
                style_parts.append(f"in a {tone} tone")
        
        if style_parts:
            style_prompt = f"Say {', '.join(style_parts)}"
        else:
            style_prompt = "Say"
        
        return await self.generate_speech(
            text=text,
            voice=voice,
            model=model,
            style_prompt=style_prompt,
            output_path=output_path,
            **kwargs
        )
    
    async def generate_podcast(
        self,
        topic: str,
        hosts: List[Speaker],
        duration_words: int = 200,
        model: str = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a podcast-style conversation
        
        Args:
            topic: Topic for the podcast
            hosts: List of podcast hosts (max 2)
            duration_words: Approximate word count
            model: Model to use
            output_path: Where to save the audio file
            **kwargs: Additional parameters
        
        Returns:
            Dict with audio file path and metadata
        """
        # First, generate the transcript
        transcript_prompt = f"""Generate a short transcript around {duration_words} words that reads
like it was clipped from a podcast about {topic}.
The hosts names are {' and '.join([h.name for h in hosts])}.
Make it conversational and engaging."""
        
        try:
            # Generate transcript with a text model
            logger.info(f"Generating podcast transcript about {topic}")
            
            transcript_response = await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-2.0-flash",  # Use text model for transcript
                contents=transcript_prompt
            )
            
            transcript = transcript_response.text
            
            # Generate audio from transcript
            result = await self.generate_multi_speaker(
                dialogue=transcript,
                speakers=hosts,
                model=model,
                output_path=output_path or f"podcast_{topic.replace(' ', '_').lower()}.wav",
                **kwargs
            )
            
            if result["success"]:
                result["transcript"] = transcript
                result["topic"] = topic
                result["word_count"] = len(transcript.split())
            
            return result
            
        except Exception as e:
            logger.error(f"Podcast generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "topic": topic
            }
    
    async def generate_audiobook_chapter(
        self,
        text: str,
        narrator_voice: str = "Schedar",
        character_voices: Optional[Dict[str, str]] = None,
        model: str = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate audiobook-style narration with optional character voices
        
        Args:
            text: Chapter text to narrate
            narrator_voice: Voice for narration
            character_voices: Optional mapping of character names to voices
            model: Model to use
            output_path: Where to save the audio file
            **kwargs: Additional parameters
        
        Returns:
            Dict with audio file path and metadata
        """
        # If no character voices, use single speaker
        if not character_voices:
            return await self.generate_speech(
                text=text,
                voice=narrator_voice,
                model=model,
                style_prompt="Narrate in a clear, engaging audiobook style",
                output_path=output_path,
                **kwargs
            )
        
        # Format for multi-speaker with narrator and one character
        # (Limited to 2 speakers total)
        if len(character_voices) > 1:
            logger.warning("Max 2 speakers supported. Using narrator and first character.")
            character_voices = {list(character_voices.keys())[0]: list(character_voices.values())[0]}
        
        # Create speakers
        speakers = [
            Speaker(name="Narrator", voice=narrator_voice),
            Speaker(
                name=list(character_voices.keys())[0],
                voice=list(character_voices.values())[0]
            )
        ]
        
        # Format text for multi-speaker
        formatted_text = f"TTS the following audiobook chapter:\nNarrator: {text}"
        
        return await self.generate_multi_speaker(
            dialogue=formatted_text,
            speakers=speakers,
            model=model,
            output_path=output_path,
            **kwargs
        )
    
    def _format_dialogue(self, dialogue: List[Dict[str, str]], speakers: List[Speaker]) -> str:
        """Format dialogue list into TTS prompt"""
        speaker_names = {s.name for s in speakers}
        
        lines = ["TTS the following conversation:"]
        for entry in dialogue:
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            
            # Add style hints if provided
            if speaker in speaker_names:
                speaker_obj = next((s for s in speakers if s.name == speaker), None)
                if speaker_obj and speaker_obj.style:
                    lines.append(f"Make {speaker} sound {speaker_obj.style}:")
            
            lines.append(f"{speaker}: {text}")
        
        return "\n".join(lines)
    
    def _save_audio_as_wav(self, audio_data: bytes, output_path: str):
        """Save raw PCM audio data as WAV file"""
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(self.audio_specs["channels"])
            wav_file.setsampwidth(self.audio_specs["sample_width"])
            wav_file.setframerate(self.audio_specs["sample_rate"])
            wav_file.writeframes(audio_data)
    
    def get_voices(self, style: Optional[str] = None) -> Dict[str, str]:
        """Get available voices, optionally filtered by style"""
        if not style:
            return VOICE_OPTIONS
        
        return {
            voice: char for voice, char in VOICE_OPTIONS.items()
            if style.lower() in char.lower()
        }
    
    def get_model_info(self, model: str = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        model = model or self.default_model
        
        if model in self.model_configs:
            return self.model_configs[model]
        else:
            return {
                "error": f"Unknown model: {model}",
                "available_models": list(self.model_configs.keys())
            }
    
    async def complete(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Compatibility method for BaseProvider interface
        Generates speech and returns the file path
        """
        result = await self.generate_speech(prompt, model=model, **kwargs)
        
        if result["success"]:
            return result["audio_path"]
        else:
            raise Exception(f"TTS generation failed: {result.get('error', 'Unknown error')}")


# Example usage
if __name__ == "__main__":
    async def test_tts():
        config = {
            "api_key": "YOUR_API_KEY"  # Or set GOOGLE_GENAI_API_KEY env var
        }
        
        provider = GenAITTSProvider(config)
        
        # Test 1: Simple speech generation
        result = await provider.generate_speech(
            text="Hello! This is a test of the text-to-speech system.",
            voice="Kore",
            output_path="test_simple.wav"
        )
        
        if result["success"]:
            print(f"✅ Generated speech: {result['audio_path']}")
            print(f"   Voice: {result['voice']} ({result['voice_style']})")
        
        # Test 2: Styled speech
        result2 = await provider.generate_styled_speech(
            text="This is amazing! I can't believe how well this works!",
            voice="Puck",
            emotion="excited",
            pace="fast",
            output_path="test_excited.wav"
        )
        
        if result2["success"]:
            print(f"✅ Generated styled speech: {result2['audio_path']}")
        
        # Test 3: Multi-speaker dialogue
        speakers = [
            Speaker(name="Alice", voice="Leda", style="cheerful"),
            Speaker(name="Bob", voice="Orus", style="thoughtful")
        ]
        
        dialogue = [
            {"speaker": "Alice", "text": "Have you tried the new AI models?"},
            {"speaker": "Bob", "text": "Yes, they're quite impressive!"},
            {"speaker": "Alice", "text": "I especially like the voice quality."},
            {"speaker": "Bob", "text": "Agreed, it sounds very natural."}
        ]
        
        result3 = await provider.generate_multi_speaker(
            dialogue=dialogue,
            speakers=speakers,
            output_path="test_dialogue.wav"
        )
        
        if result3["success"]:
            print(f"✅ Generated dialogue: {result3['audio_path']}")
            print(f"   Speakers: {', '.join([s['name'] for s in result3['speakers']])}")
        
        # Test 4: Podcast generation
        hosts = [
            Speaker(name="Dr. Smith", voice="Sadaltager"),
            Speaker(name="Jane", voice="Callirrhoe")
        ]
        
        result4 = await provider.generate_podcast(
            topic="artificial intelligence",
            hosts=hosts,
            duration_words=100,
            output_path="test_podcast.wav"
        )
        
        if result4["success"]:
            print(f"✅ Generated podcast: {result4['audio_path']}")
            print(f"   Word count: {result4.get('word_count', 'N/A')}")
    
    # Run test
    import asyncio
    asyncio.run(test_tts())