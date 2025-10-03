# Image Generation Guide

## Quick Start

### DALL-E 3 (OpenAI)
```bash
# Basic generation (HD quality default)
python harvester.py image "duck having space battle with cats" --model dalle-3

# With template
python harvester.py image "cosmic nebula scene" --model dalle-3 --template cosmic_duck

# Different sizes
python harvester.py image "portrait of a sage" --model dalle-3 --size 1024x1792
```

### Nano Banana (Gemini Flash Image - Fastest!)
```bash
# Requires: export GEMINI_API_KEY=your_key
python harvester.py image "futuristic cityscape" --model nano-banana

# With aspect ratio
python harvester.py image "mountain landscape" --model nano-banana --aspect-ratio 16:9
```

### Grok Image (xAI)
```bash
# Requires: export XAI_API_KEY=your_key
python harvester.py image "abstract art piece" --model grok-image
```

### Imagen 4 (Google Vertex AI - Requires GCP Setup)
```bash
# Ultra quality (best, slower)
python harvester.py image "detailed portrait" --model imagen-4-ultra --aspect-ratio 9:16

# Fast generation
python harvester.py image "quick sketch" --model imagen-4-fast --aspect-ratio 16:9

# Standard quality
python harvester.py image "landscape scene" --model imagen-4 --aspect-ratio 16:9
```

### GPT Image (OpenAI)
```bash
python harvester.py image "creative composition" --model gpt-image
```

## Available Models

| Model | Provider | Speed | Quality | API Key Required | Notes |
|-------|----------|-------|---------|------------------|-------|
| `nano-banana` | Google GenAI | ⚡ Fastest | Good | `GEMINI_API_KEY` | Default, no GCP needed |
| `dalle-3` | OpenAI | Fast | Excellent | `OPENAI_API_KEY` | Popular choice |
| `gpt-image` | OpenAI | Fast | Good | `OPENAI_API_KEY` | Alternative format |
| `grok-image` | xAI | Fast | Good | `XAI_API_KEY` | Grok 2 Image |
| `imagen-4-ultra` | Vertex AI | Slower | ⭐ Best | GCP Setup | Imagen 4 Ultra |
| `imagen-4-fast` | Vertex AI | Fast | Excellent | GCP Setup | Imagen 4 Fast |
| `imagen-4` | Vertex AI | Medium | Excellent | GCP Setup | Imagen 4 Standard |

## Templates

### Cosmic Duck Style
Adds ethereal space theme to any prompt:
```bash
python harvester.py image "duck" --template cosmic_duck
# Becomes: "duck, cosmic nebula background, ethereal lighting, space fantasy art style"
```

### Professional Style
Clean, professional photography:
```bash
python harvester.py image "headshot" --template professional
# Becomes: "headshot, professional photography, clean background, high quality"
```

### Artistic Style
Vibrant digital art:
```bash
python harvester.py image "sunset" --template artistic
# Becomes: "sunset, digital art, vibrant colors, artistic composition"
```

## Output Location

All images are saved to: `~/harvester-sdk/images/`

Each generation creates a timestamped folder:
```
~/harvester-sdk/images/
  └── duck_having_space_battle_with_20251003_214528/
      └── image_20251003_214543.png
```

## Size & Aspect Ratio

### OpenAI Models (DALL-E 3, GPT Image)
Use `--size`:
- `1024x1024` (square, default)
- `1792x1024` (landscape)
- `1024x1792` (portrait)

### Google Models (GenAI, Imagen)
Use `--aspect-ratio`:
- `1:1` (square, default)
- `16:9` (landscape)
- `9:16` (portrait)
- `4:3` (standard)
- `3:4` (vertical)

## Batch Generation

Create a CSV file with prompts:

```csv
prompt,model,size
"duck in space",dalle-3,1024x1024
"cat warrior",grok-image,1024x1024
"cosmic scene",genai-flash-img,
```

Then run:
```bash
python harvester.py image --batch my_prompts.csv
```

## API Key Setup

```bash
# OpenAI (DALL-E 3, GPT Image)
export OPENAI_API_KEY=sk-...

# Google AI Studio (Gemini Flash Image)
export GEMINI_API_KEY=...

# xAI (Grok Image)
export XAI_API_KEY=xai-...

# Google Vertex AI (Imagen 4)
export GOOGLE_CLOUD_PROJECT=your-project-id
# Authenticate: gcloud auth application-default login
```

## Troubleshooting

### "No API key found"
Set the appropriate environment variable for your chosen model (see API Key Setup above).

### "project ID required" (Vertex AI)
```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
```

### "aspect_ratio error" (Vertex AI)
Vertex models require aspect ratio. Always use `--aspect-ratio` with Imagen models:
```bash
python harvester.py image "prompt" --model goo-4-img --aspect-ratio 1:1
```

## Examples Gallery

```bash
# Space battle scene
python harvester.py image "epic space battle with colorful nebulas" --model dalle-3

# Quick concept art (Nano Banana is fastest!)
python harvester.py image "steampunk airship" --model nano-banana --aspect-ratio 16:9

# High quality portrait
python harvester.py image "wise old wizard portrait" --model imagen-4-ultra --aspect-ratio 9:16

# Abstract art
python harvester.py image "abstract geometric patterns" --model grok-image
```
