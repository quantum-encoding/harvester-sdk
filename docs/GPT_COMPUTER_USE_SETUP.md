# GPT Computer Use Setup Guide

Complete guide for setting up and using the GPT Computer Use agent in the Harvester SDK.

---

## Overview

The GPT Computer Use feature allows GPT-5's `computer-use-preview` model to control a browser or computer environment to perform tasks autonomously. It works by:

1. Taking screenshots of the environment
2. Sending them to the model along with your task
3. Receiving action instructions (click, type, scroll, etc.)
4. Executing those actions
5. Repeating until the task is complete

---

## Prerequisites

### Required

- Python 3.8+
- OpenAI API key with access to `computer-use-preview` model
- `pip install openai` (already included in SDK requirements)

### Environment-Specific Requirements

Choose one or both:

#### Browser Environment (Recommended for beginners)
```bash
pip install playwright
playwright install
```

#### Docker Environment (Advanced)
- Docker installed and running
- Docker image built (see Docker setup below)

---

## Quick Start - Browser Environment

This is the easiest way to get started.

### 1. Install Playwright

```bash
pip install playwright
playwright install chromium
```

### 2. Set Your API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 3. Run a Task

```bash
# Simple search task
python cli/chat/gpt_computer_use.py "Search for OpenAI news on bing.com"

# Or via harvester CLI
harvester computer "Search for OpenAI news on bing.com"
```

### 4. Watch It Work

A Chrome browser window will open and you'll see the agent:
- Navigate to websites
- Click on elements
- Type in search boxes
- Scroll through pages
- Complete your task autonomously

---

## Advanced Setup - Docker Environment

Use Docker for full OS-level control beyond just browser automation.

### 1. Create a Dockerfile

Create a file named `Dockerfile` in your project directory:

```dockerfile
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Xfce, x11vnc, Xvfb, xdotool, etc.
RUN apt-get update && apt-get install -y \
    xfce4 \
    xfce4-goodies \
    x11vnc \
    xvfb \
    xdotool \
    imagemagick \
    x11-apps \
    sudo \
    software-properties-common \
    imagemagick \
 && apt-get remove -y light-locker xfce4-screensaver xfce4-power-manager || true \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Firefox ESR
RUN add-apt-repository ppa:mozillateam/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends firefox-esr \
 && update-alternatives --set x-www-browser /usr/bin/firefox-esr \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -ms /bin/bash myuser \
    && echo "myuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER myuser
WORKDIR /home/myuser

# Set x11vnc password
RUN x11vnc -storepasswd secret /home/myuser/.vncpass

# Expose VNC port
EXPOSE 5900

CMD ["/bin/sh", "-c", " \
    Xvfb :99 -screen 0 1280x800x24 >/dev/null 2>&1 & \
    x11vnc -display :99 -forever -rfbauth /home/myuser/.vncpass -listen 0.0.0.0 -rfbport 5900 >/dev/null 2>&1 & \
    export DISPLAY=:99 && \
    startxfce4 >/dev/null 2>&1 & \
    sleep 2 && echo 'Container running!' && \
    tail -f /dev/null \
"]
```

### 2. Build the Docker Image

```bash
docker build -t cua-image .
```

### 3. Run the Container

```bash
docker run --rm -it --name cua-container -p 5900:5900 -e DISPLAY=:99 cua-image
```

Keep this terminal running. Open a new terminal for the next steps.

### 4. (Optional) Connect with VNC Viewer

To watch what's happening inside the container:

```bash
# Install a VNC viewer (e.g., TigerVNC, RealVNC, etc.)
# Connect to: localhost:5900
# Password: secret
```

### 5. Run Tasks with Docker Environment

```bash
python cli/chat/gpt_computer_use.py --environment docker "Open Firefox and search for AI news"

# Or via harvester CLI
harvester computer --environment docker "Open Firefox and browse to github.com"
```

---

## Usage Examples

### Browser Environment

```bash
# Basic search
harvester computer "Search for OpenAI news on bing.com"

# Start at specific URL
harvester computer --url https://github.com "Search for trending AI repositories"

# Book travel
harvester computer "Find flights from NYC to SF on kayak.com"

# Research task
harvester computer "Search for Python tutorials on youtube.com and summarize the top 3"

# Shopping
harvester computer "Find the best price for iPhone 15 on amazon.com"
```

### Docker Environment

```bash
# Open applications
harvester computer --environment docker "Open Firefox and navigate to wikipedia.org"

# Multi-step task
harvester computer --environment docker "Open a text editor and write a hello world program"

# System tasks
harvester computer --environment docker "Take a screenshot and save it"
```

### Custom Display Size

```bash
# Larger display
harvester computer --width 1920 --height 1080 "Browse the web"

# Smaller display (faster)
harvester computer --width 800 --height 600 "Quick search task"
```

---

## Direct Script Usage

You can also run the script directly without the harvester CLI:

```bash
# Browser
python cli/chat/gpt_computer_use.py "Your task here"

# With options
python cli/chat/gpt_computer_use.py \
    --environment browser \
    --url https://bing.com \
    --width 1024 \
    --height 768 \
    "Search for AI news"

# Docker
python cli/chat/gpt_computer_use.py \
    --environment docker \
    --container cua-container \
    --display :99 \
    "Open Firefox"
```

---

## Command Line Options

### Harvester CLI

```
harvester computer [OPTIONS] TASK

Options:
  -e, --environment [browser|docker]  Environment to use (default: browser)
  -u, --url TEXT                      Initial URL to navigate to (browser only)
  --width INTEGER                     Display width (default: 1024)
  --height INTEGER                    Display height (default: 768)
```

### Direct Script

```
python cli/chat/gpt_computer_use.py [OPTIONS] TASK

Options:
  --environment, -e [browser|docker]  Environment to use (default: browser)
  --url, -u TEXT                      Initial URL to navigate to
  --width INTEGER                     Display width (default: 1024)
  --height INTEGER                    Display height (default: 768)
  --container TEXT                    Docker container name (default: cua-container)
  --display TEXT                      Docker display (default: :99)
```

---

## Safety Features

The agent includes built-in safety checks:

### 1. Malicious Instructions Detection
Detects if a screenshot contains adversarial content trying to change the model's behavior.

### 2. Irrelevant Domain Detection
Warns if the current domain seems unrelated to the task.

### 3. Sensitive Domain Detection
Warns when on sensitive domains (banking, authentication, etc.).

### Important Safety Notes

⚠️ **DO NOT use on:**
- Fully authenticated environments
- Banking or financial sites
- Sites with sensitive personal data
- High-stakes production systems

⚠️ **ALWAYS:**
- Use in sandboxed environments
- Monitor the agent's actions
- Implement user confirmation for safety checks
- Use blocklists/allowlists for domains

The current implementation **auto-acknowledges** safety checks with warnings. In production, you should modify the code to require explicit user confirmation.

---

## Troubleshooting

### "Playwright not installed"
```bash
pip install playwright
playwright install chromium
```

### "OpenAI API key required"
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### "Docker container not running"
Make sure your Docker container is running:
```bash
docker ps | grep cua-container
```

If not, start it:
```bash
docker run --rm -it --name cua-container -p 5900:5900 -e DISPLAY=:99 cua-image
```

### "Maximum iterations reached"
The task took more than 50 iterations. This could mean:
- Task is too complex - try breaking it into smaller tasks
- Agent got stuck - try rephrasing the task
- Website is slow - the agent might need more wait time

### Browser closes immediately
Make sure the script is running with `headless=False` (default). Check the code or logs for errors.

### Can't see what's happening in Docker
Connect with a VNC viewer to `localhost:5900` (password: `secret`)

---

## How It Works

### The Computer Use Loop

```
1. User provides task + initial screenshot
   ↓
2. Model analyzes and suggests action
   ↓
3. Code executes action (click, type, etc.)
   ↓
4. Screenshot captured of new state
   ↓
5. Screenshot sent back to model
   ↓
6. Repeat until task complete
```

### Supported Actions

- **click**: Click at (x, y) with left/middle/right button
- **scroll**: Scroll at (x, y) by offset
- **keypress**: Press specific keys (Enter, Space, etc.)
- **type**: Type text
- **wait**: Wait for changes to take effect
- **screenshot**: Capture current state

---

## Architecture

```
GPTComputerUse
├── Environment Setup
│   ├── Browser (Playwright)
│   │   ├── Chromium sandbox
│   │   ├── No extensions
│   │   └── No file system access
│   └── Docker (VM)
│       ├── Xvfb (virtual display)
│       ├── xdotool (mouse/keyboard)
│       └── ImageMagick (screenshots)
│
├── Action Handlers
│   ├── handle_browser_action()
│   └── handle_docker_action()
│
├── Screenshot Capture
│   ├── get_browser_screenshot()
│   └── get_docker_screenshot()
│
└── Main Loop
    ├── Send request with screenshot
    ├── Receive action from model
    ├── Check safety warnings
    ├── Execute action
    ├── Capture new screenshot
    └── Repeat
```

---

## Model Information

- **Model**: `computer-use-preview`
- **API**: OpenAI Responses API (not Chat Completions)
- **Context**: Vision + Reasoning capabilities
- **Rate Limits**: Check OpenAI documentation for current limits

---

## Best Practices

### 1. Start Simple
Begin with straightforward tasks like "Search for X on Y website" before attempting complex multi-step workflows.

### 2. Be Specific
```
✅ Good: "Go to bing.com and search for 'OpenAI GPT-5', then summarize the first 3 results"
❌ Bad: "Find some info about AI"
```

### 3. Use Initial URLs
```bash
# Faster - starts at the right place
harvester computer --url https://github.com "Search for trending repos"

# Slower - has to navigate from blank page
harvester computer "Go to github and search for trending repos"
```

### 4. Monitor First Runs
Watch what the agent does on first runs to understand its behavior and adjust your task descriptions.

### 5. Implement Domain Filtering
For production use, add allowlists/blocklists:

```python
ALLOWED_DOMAINS = ['bing.com', 'github.com', 'stackoverflow.com']
# Check current_url against allowlist before proceeding
```

---

## Production Checklist

Before deploying to production:

- [ ] Implement user confirmation for safety checks
- [ ] Add domain allowlists/blocklists
- [ ] Set up proper error handling and logging
- [ ] Limit task complexity and iteration counts
- [ ] Monitor API usage and costs
- [ ] Add session timeouts
- [ ] Implement task approval workflows
- [ ] Use separate sandboxed environments per user
- [ ] Add rate limiting for tasks
- [ ] Implement audit logging of all actions

---

## Known Limitations

1. **Model Reliability**: ~38% success rate on OSWorld benchmark - not yet production-ready for complex OS tasks
2. **Rate Limits**: `computer-use-preview` has constrained rate limits
3. **Best for Browsers**: More reliable in browser environments than full OS
4. **No Parallel Tasks**: One task at a time per environment
5. **Cost**: Vision API calls can be expensive for long tasks
6. **Authentication**: Not recommended for authenticated sessions
7. **Iteration Limit**: Hard cap at 50 iterations to prevent runaway costs

---

## Cost Considerations

Computer Use can be expensive because:
- Each iteration requires vision API call (screenshot analysis)
- Complex tasks may require 10-50+ iterations
- Screenshots are large images

**Cost-saving tips:**
- Use smaller display sizes (800x600 vs 1920x1080)
- Break complex tasks into smaller subtasks
- Set lower iteration limits for testing
- Use specific starting URLs to reduce navigation steps

---

## Support & Resources

- **OpenAI Docs**: https://platform.openai.com/docs/guides/tools-computer-use
- **Sample App**: https://github.com/openai/openai-cua-sample-app
- **System Card**: https://openai.com/index/operator-system-card/
- **Harvester SDK**: Contact info@quantumencoding.io

---

## License & Terms

Use of the Computer Use feature must comply with:
- OpenAI [Usage Policy](https://openai.com/policies/usage-policies/)
- OpenAI [Business Terms](https://openai.com/policies/business-terms/)
- Harvester SDK license

---

**© 2025 QUANTUM ENCODING LTD**
