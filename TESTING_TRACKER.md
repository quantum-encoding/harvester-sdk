# Endpoint Testing Tracker

Track testing progress for all SDK endpoints to ensure 2-minute time-to-success.

## Testing Workflow

1. **Pick an endpoint** from the list below
2. **Test it** - Run the command, time yourself
3. **Copy the template** - `docs/quickstart/TEMPLATE_QUICKSTART.md`
4. **Fill it out** - Include actual commands and output
5. **Save it** - `docs/quickstart/[FEATURE]_QUICKSTART.md`
6. **Mark complete** - Check the box below

## Image Generation

- [x] ✅ **Nano Banana** (Gemini Flash Image) - `IMAGE_QUICKSTART.md`
- [x] ✅ **DALL-E 3** - `IMAGE_QUICKSTART.md`
- [x] ✅ **Grok Image** - `IMAGE_QUICKSTART.md`
- [ ] 🔲 **Imagen 4 Ultra** (Vertex) - Needs GCP testing
- [ ] 🔲 **Imagen 4 Fast** (Vertex) - Needs GCP testing
- [ ] 🔲 **GPT Image** - Needs testing

## Text Generation

- [ ] 🔲 **GPT-5** - OpenAI latest
- [ ] 🔲 **GPT-5 Mini** - Fast/cheap
- [ ] 🔲 **GPT-5 Nano** - Fastest
- [ ] 🔲 **Claude Opus 4.1** - Anthropic flagship
- [ ] 🔲 **Claude Sonnet 4.5** - Balanced
- [ ] 🔲 **Claude Haiku** - Fast
- [ ] 🔲 **Gemini 2.5 Pro** - Google flagship
- [ ] 🔲 **Gemini 2.5 Flash** - Fast
- [ ] 🔲 **Gemini Flash Lite** - Fastest
- [ ] 🔲 **Grok 4** - xAI flagship
- [ ] 🔲 **Grok 3** - Standard
- [ ] 🔲 **Grok 3 Mini** - Fast
- [ ] 🔲 **DeepSeek Chat** - V3.2
- [ ] 🔲 **DeepSeek Reasoner** - R1

## Chat Interface

- [x] ✅ **Turn-based chat** - `CHAT_GUIDE.md`
- [x] ✅ **Model switching** (`/model`) - `CHAT_GUIDE.md`
- [ ] 🔲 **Chat commands** (`/help`, `/temp`, etc.)
- [ ] 🔲 **System prompts** (`/system`)

## Batch Processing

- [ ] 🔲 **CSV batch** - Process rows in parallel
- [ ] 🔲 **Directory batch** - Process files
- [ ] 🔲 **JSON batch** - Template-based processing
- [ ] 🔲 **Image batch** - Multiple images from CSV

## Templates

- [ ] 🔲 **Blog post generation**
- [ ] 🔲 **Code documentation**
- [ ] 🔲 **Code review**
- [ ] 🔲 **Translation**
- [ ] 🔲 **SEO content**
- [ ] 🔲 **Product descriptions**

## Advanced Features

- [ ] 🔲 **Function calling** - Structured output
- [ ] 🔲 **Embeddings** - Vector generation
- [ ] 🔲 **Video generation** (Veo 3)
- [ ] 🔲 **Music generation** (Lyria)
- [ ] 🔲 **Code execution** (Claude)

## Provider-Specific

### OpenAI
- [ ] 🔲 Batch API submission
- [ ] 🔲 Batch status checking
- [ ] 🔲 Results extraction

### Anthropic
- [ ] 🔲 Code execution mode
- [ ] 🔲 Extended context

### Google Vertex
- [ ] 🔲 Service account auth
- [ ] 🔲 Model Garden access

### xAI
- [ ] 🔲 Grok search integration
- [ ] 🔲 Real-time data access

## Testing Checklist (for each endpoint)

When testing, verify:
- [ ] Command works on first try
- [ ] Takes less than 2 minutes total
- [ ] Error messages are clear
- [ ] Output location is obvious
- [ ] Help text is useful
- [ ] Works with minimal API key setup

## Priority Queue

**High Priority (Most Common Use Cases):**
1. GPT-5 text generation
2. Claude Sonnet 4.5 text generation
3. Gemini 2.5 Flash text generation
4. CSV batch processing
5. Blog post template

**Medium Priority:**
1. All other text models
2. Directory batch processing
3. Code templates
4. Embeddings

**Low Priority:**
1. Video/Music generation
2. Advanced function calling
3. Provider-specific batch APIs

## Notes

- Time yourself for each test
- Record actual terminal output
- Note any confusing error messages
- Save working commands exactly as-is
- Include troubleshooting for common issues

## Progress

- **Completed:** 6 / 50+
- **In Progress:** 0
- **Not Started:** 44+

---

**Goal:** Every endpoint should have a working quickstart guide with <2 min time-to-success!
