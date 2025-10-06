# Image Generation - 60 Second Quickstart

Generate your first AI image in under 60 seconds!

## Prerequisites

Choose one (fastest to slowest):
- `GEMINI_API_KEY` - Nano Banana (fastest, free tier available)
- `OPENAI_API_KEY` - DALL-E 3 (popular, high quality)
- `XAI_API_KEY` - Grok Image (creative)

## Install (10 seconds)

```bash
git clone https://github.com/quantum-encoding/harvester-sdk.git
cd harvester-sdk
pip install -r requirements.txt
```

## Your First Image (30 seconds)

### Option 1: Nano Banana (Fastest!)

```bash
export GEMINI_API_KEY=your_key_here
python harvester.py image "cosmic duck warriors in space" --model nano-banana
```

**Output:** `~/harvester-sdk/images/cosmic_duck_warriors_in_spac_TIMESTAMP/image_TIMESTAMP.png`

### Option 2: DALL-E 3 (Most Popular)

```bash
export OPENAI_API_KEY=your_key_here
python harvester.py image "cosmic duck warriors in space" --model dalle-3
```

**Output:** `~/harvester-sdk/images/cosmic_duck_warriors_in_spac_TIMESTAMP/image_TIMESTAMP.png`

### Option 3: Grok Image (xAI)

```bash
export XAI_API_KEY=your_key_here
python harvester.py image "cosmic duck warriors in space" --model grok-image
```

**Output:** `~/harvester-sdk/images/cosmic_duck_warriors_in_spac_TIMESTAMP/image_TIMESTAMP.png`

## ✅ Success!

You should see:
```
🎨 Generating image with nano-banana: cosmic duck warriors in space...
✅ Image saved: ~/harvester-sdk/images/.../image_20251003_221358.png
```

## Next Steps (20 seconds)

### Try different models:

```bash
# Highest quality (needs GCP)
python harvester.py image "majestic dragon" --model imagen-4-ultra --aspect-ratio 16:9

# With style template
python harvester.py image "sunset over mountains" --model dalle-3 --template professional

# Portrait orientation
python harvester.py image "wise wizard portrait" --model nano-banana --aspect-ratio 9:16
```

### See all options:

```bash
python harvester.py image --help
```

## Troubleshooting

**"No API key found"** - Set your environment variable:
```bash
export GEMINI_API_KEY=your_key
```

**"Module not found"** - Install requirements:
```bash
pip install -r requirements.txt
```

## Full Guide

See [IMAGE_GENERATION.md](../IMAGE_GENERATION.md) for all models, templates, and advanced options.

---

**Time to success: ~60 seconds** ⚡
