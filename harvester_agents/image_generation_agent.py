"""
Image Generation Agent - AI-powered image creation and editing

Generate and edit images using GPT Image model with automatic prompt optimization.
"""

from typing import Optional, Dict, Any, List, Literal
from pathlib import Path
import os
import base64
import logging

logger = logging.getLogger(__name__)

# Import OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI SDK not installed. Install with: "
        "pip install 'harvester-sdk[computer]'"
    )


class ImageGenerationAgent:
    """
    Agent that can generate and edit images using AI.

    Features:
    - Text-to-image generation
    - Image editing and refinement
    - Multi-turn iterative editing
    - Automatic prompt optimization
    - Configurable size, quality, format
    """

    def __init__(
        self,
        model: str = "gpt-5",
        size: str = "auto",
        quality: Literal["low", "medium", "high", "auto"] = "auto",
        format: Literal["png", "jpeg", "webp"] = "png",
        background: Literal["transparent", "opaque", "auto"] = "auto"
    ):
        """
        Initialize Image Generation Agent.

        Args:
            model: Model to use (gpt-4o, gpt-4.1, gpt-5, o3)
            size: Image dimensions (e.g., 1024x1024, auto)
            quality: Rendering quality (low, medium, high, auto)
            format: Output format (png, jpeg, webp)
            background: Background mode (transparent, opaque, auto)
        """
        self.client = OpenAI()
        self.model = model
        self.size = size
        self.quality = quality
        self.format = format
        self.background = background

        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable")

        # Track previous response for multi-turn editing
        self.previous_response_id = None

    def generate(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        show_progress: bool = True,
        previous_response_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            output_path: Path to save the generated image
            show_progress: Whether to show progress messages
            previous_response_id: Previous response ID for multi-turn editing

        Returns:
            Dict with status, image_base64, revised_prompt, and output_path
        """
        if show_progress:
            print(f"🎨 Starting Image Generation (model: {self.model})...")
            print(f"📝 Prompt: {prompt}")
            print()

        # Configure tool
        tool_config = {
            "type": "image_generation"
        }

        # Add optional parameters
        if self.size != "auto":
            tool_config["size"] = self.size
        if self.quality != "auto":
            tool_config["quality"] = self.quality
        if self.format:
            tool_config["format"] = self.format
        if self.background != "auto":
            tool_config["background"] = self.background

        # Create request
        request_params = {
            "model": self.model,
            "input": prompt,
            "tools": [tool_config]
        }

        # Add previous response if doing multi-turn editing
        if previous_response_id or self.previous_response_id:
            request_params["previous_response_id"] = previous_response_id or self.previous_response_id

        response = self.client.responses.create(**request_params)

        # Store for future multi-turn edits
        self.previous_response_id = response.id

        # Extract generated images
        image_data = None
        revised_prompt = None

        for output in response.output:
            if output.type == "image_generation_call":
                image_data = output.result
                if hasattr(output, 'revised_prompt'):
                    revised_prompt = output.revised_prompt

                if show_progress:
                    print(f"{'='*60}")
                    print("🖼️  Image Generated")
                    print(f"{'='*60}")
                    if revised_prompt:
                        print(f"✨ Revised prompt: {revised_prompt}")
                    print()

        if not image_data:
            return {
                'status': 'failed',
                'error': 'No image generated'
            }

        # Save image if output path provided
        saved_path = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
            saved_path = output_path

            if show_progress:
                print(f"💾 Saved to: {output_path}")

        if show_progress:
            print("✓ Image generation complete")

        return {
            'status': 'completed',
            'image_base64': image_data,
            'revised_prompt': revised_prompt,
            'output_path': saved_path,
            'response_id': response.id
        }

    def edit(
        self,
        edit_prompt: str,
        output_path: Optional[str] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Edit the previously generated image.

        Args:
            edit_prompt: Description of how to edit the image
            output_path: Path to save the edited image
            show_progress: Whether to show progress messages

        Returns:
            Dict with status, image_base64, revised_prompt, and output_path
        """
        if not self.previous_response_id:
            return {
                'status': 'failed',
                'error': 'No previous image to edit. Generate an image first.'
            }

        if show_progress:
            print(f"🎨 Editing previous image...")
            print(f"📝 Edit: {edit_prompt}")
            print()

        return self.generate(
            prompt=edit_prompt,
            output_path=output_path,
            show_progress=show_progress,
            previous_response_id=self.previous_response_id
        )

    def generate_from_file(
        self,
        prompt: str,
        input_image_path: str,
        output_path: Optional[str] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Generate/edit an image using an input image file.

        Args:
            prompt: Text description of what to generate/edit
            input_image_path: Path to input image
            output_path: Path to save the result
            show_progress: Whether to show progress messages

        Returns:
            Dict with status, image_base64, revised_prompt, and output_path
        """
        if not Path(input_image_path).exists():
            return {
                'status': 'failed',
                'error': f'Input image not found: {input_image_path}'
            }

        # Read and encode input image
        with open(input_image_path, 'rb') as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Determine image type
        ext = Path(input_image_path).suffix.lower()
        mime_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp'
        }.get(ext, 'image/png')

        if show_progress:
            print(f"🎨 Generating with input image: {input_image_path}")
            print(f"📝 Prompt: {prompt}")
            print()

        # Configure tool
        tool_config = {
            "type": "image_generation"
        }

        if self.size != "auto":
            tool_config["size"] = self.size
        if self.quality != "auto":
            tool_config["quality"] = self.quality
        if self.format:
            tool_config["format"] = self.format
        if self.background != "auto":
            tool_config["background"] = self.background

        # Create request with input image
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{image_base64}"
                },
                {
                    "type": "input_text",
                    "text": prompt
                }
            ],
            tools=[tool_config]
        )

        # Store for future edits
        self.previous_response_id = response.id

        # Extract generated image
        image_data = None
        revised_prompt = None

        for output in response.output:
            if output.type == "image_generation_call":
                image_data = output.result
                if hasattr(output, 'revised_prompt'):
                    revised_prompt = output.revised_prompt

                if show_progress:
                    print(f"{'='*60}")
                    print("🖼️  Image Generated")
                    print(f"{'='*60}")
                    if revised_prompt:
                        print(f"✨ Revised prompt: {revised_prompt}")
                    print()

        if not image_data:
            return {
                'status': 'failed',
                'error': 'No image generated'
            }

        # Save image if output path provided
        saved_path = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
            saved_path = output_path

            if show_progress:
                print(f"💾 Saved to: {output_path}")

        if show_progress:
            print("✓ Image generation complete")

        return {
            'status': 'completed',
            'image_base64': image_data,
            'revised_prompt': revised_prompt,
            'output_path': saved_path,
            'response_id': response.id
        }
