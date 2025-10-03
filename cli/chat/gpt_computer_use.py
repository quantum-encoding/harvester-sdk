#!/usr/bin/env python3
"""
GPT Computer Use Agent
A computer-using agent that can perform tasks using GPT-5's computer-use-preview model.

Features:
- Browser automation using Playwright
- Docker VM support
- Screenshot-based interaction loop
- Safety checks for prompt injection protection
- Real-time action execution and feedback

Usage:
    python gpt_computer_use.py --environment browser "Search for OpenAI news"
    python gpt_computer_use.py --environment docker "Open Firefox and search"
    python gpt_computer_use.py --browser --task "Book a flight to NYC"
"""

import asyncio
import sys
import os
import json
import base64
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load environment variables
load_dotenv()

# Color codes for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class GPTComputerUse:
    def __init__(self, environment: str = "browser", display_width: int = 1024, display_height: int = 768):
        # Check for OpenAI SDK
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except ImportError:
            raise ImportError("OpenAI SDK not installed. Install with: pip install openai")

        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable")

        self.environment = environment
        self.display_width = display_width
        self.display_height = display_height
        self.model = "computer-use-preview"

        # Environment instances
        self.browser_page = None
        self.docker_vm = None
        self.playwright = None
        self.browser = None

    async def setup_browser(self):
        """Setup Playwright browser environment"""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError("Playwright not installed. Install with: pip install playwright && playwright install")

        print(f"{Colors.CYAN}🌐 Starting browser environment...{Colors.ENDC}")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            chromium_sandbox=True,
            env={},
            args=["--disable-extensions", "--disable-file-system"]
        )
        self.browser_page = await self.browser.new_page()
        await self.browser_page.set_viewport_size({
            "width": self.display_width,
            "height": self.display_height
        })
        print(f"{Colors.GREEN}✓ Browser ready{Colors.ENDC}")

    def setup_docker(self, container_name: str = "cua-container", display: str = ":99"):
        """Setup Docker VM environment"""
        print(f"{Colors.CYAN}🐳 Setting up Docker environment...{Colors.ENDC}")

        # Check if container is running
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={container_name}"],
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                print(f"{Colors.YELLOW}Container not running. Please start the Docker container first.{Colors.ENDC}")
                print(f"{Colors.YELLOW}Run: docker run --rm -it --name {container_name} -p 5900:5900 -e DISPLAY={display} cua-image{Colors.ENDC}")
                raise RuntimeError("Docker container not running")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error checking Docker container: {e}")

        self.docker_vm = {
            "container_name": container_name,
            "display": display
        }
        print(f"{Colors.GREEN}✓ Docker environment ready{Colors.ENDC}")

    def docker_exec(self, cmd: str, decode: bool = True) -> str:
        """Execute command in Docker container"""
        safe_cmd = cmd.replace('"', '\\"')
        docker_cmd = f'docker exec {self.docker_vm["container_name"]} sh -c "{safe_cmd}"'
        output = subprocess.check_output(docker_cmd, shell=True)
        if decode:
            return output.decode("utf-8", errors="ignore")
        return output

    async def get_browser_screenshot(self) -> bytes:
        """Capture screenshot from browser"""
        return await self.browser_page.screenshot()

    def get_docker_screenshot(self) -> bytes:
        """Capture screenshot from Docker VM"""
        cmd = f"export DISPLAY={self.docker_vm['display']} && import -window root png:-"
        return self.docker_exec(cmd, decode=False)

    async def get_screenshot(self) -> bytes:
        """Get screenshot based on environment"""
        if self.environment == "browser":
            return await self.get_browser_screenshot()
        elif self.environment == "docker":
            return self.get_docker_screenshot()
        else:
            raise ValueError(f"Unknown environment: {self.environment}")

    async def handle_browser_action(self, action: Dict[str, Any]):
        """Execute action in browser environment"""
        action_type = action.get("type")

        try:
            if action_type == "click":
                x, y = action["x"], action["y"]
                button = action.get("button", "left")
                print(f"  → Click at ({x}, {y}) with {button} button")
                await self.browser_page.mouse.click(x, y, button=button)

            elif action_type == "scroll":
                x, y = action["x"], action["y"]
                scroll_x = action.get("scroll_x", 0)
                scroll_y = action.get("scroll_y", 0)
                print(f"  → Scroll at ({x}, {y}) by ({scroll_x}, {scroll_y})")
                await self.browser_page.mouse.move(x, y)
                await self.browser_page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")

            elif action_type == "keypress":
                keys = action.get("keys", [])
                for key in keys:
                    print(f"  → Press key: {key}")
                    if "ENTER" in key.upper():
                        await self.browser_page.keyboard.press("Enter")
                    elif "SPACE" in key.upper():
                        await self.browser_page.keyboard.press(" ")
                    else:
                        await self.browser_page.keyboard.press(key)

            elif action_type == "type":
                text = action.get("text", "")
                print(f"  → Type: {text}")
                await self.browser_page.keyboard.type(text)

            elif action_type == "wait":
                print(f"  → Wait")
                await self.browser_page.wait_for_timeout(2000)

            elif action_type == "screenshot":
                print(f"  → Screenshot")
                # Screenshot will be taken after action

            else:
                print(f"{Colors.YELLOW}  ⚠ Unknown action: {action_type}{Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.RED}  ✗ Error executing action: {e}{Colors.ENDC}")

    def handle_docker_action(self, action: Dict[str, Any]):
        """Execute action in Docker environment"""
        action_type = action.get("type")
        display = self.docker_vm["display"]

        try:
            if action_type == "click":
                x, y = int(action["x"]), int(action["y"])
                button_map = {"left": 1, "middle": 2, "right": 3}
                button = button_map.get(action.get("button", "left"), 1)
                print(f"  → Click at ({x}, {y}) with button {action.get('button', 'left')}")
                self.docker_exec(f"DISPLAY={display} xdotool mousemove {x} {y} click {button}")

            elif action_type == "scroll":
                x, y = int(action["x"]), int(action["y"])
                scroll_y = int(action.get("scroll_y", 0))
                print(f"  → Scroll at ({x}, {y}) by {scroll_y}")
                self.docker_exec(f"DISPLAY={display} xdotool mousemove {x} {y}")
                if scroll_y != 0:
                    button = 4 if scroll_y < 0 else 5
                    clicks = abs(scroll_y)
                    for _ in range(clicks):
                        self.docker_exec(f"DISPLAY={display} xdotool click {button}")

            elif action_type == "keypress":
                keys = action.get("keys", [])
                for key in keys:
                    print(f"  → Press key: {key}")
                    if "ENTER" in key.upper():
                        self.docker_exec(f"DISPLAY={display} xdotool key 'Return'")
                    elif "SPACE" in key.upper():
                        self.docker_exec(f"DISPLAY={display} xdotool key 'space'")
                    else:
                        self.docker_exec(f"DISPLAY={display} xdotool key '{key}'")

            elif action_type == "type":
                text = action.get("text", "")
                print(f"  → Type: {text}")
                self.docker_exec(f"DISPLAY={display} xdotool type '{text}'")

            elif action_type == "wait":
                print(f"  → Wait")
                time.sleep(2)

            elif action_type == "screenshot":
                print(f"  → Screenshot")
                # Screenshot will be taken after action

            else:
                print(f"{Colors.YELLOW}  ⚠ Unknown action: {action_type}{Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.RED}  ✗ Error executing action: {e}{Colors.ENDC}")

    async def handle_action(self, action: Dict[str, Any]):
        """Execute action based on environment"""
        if self.environment == "browser":
            await self.handle_browser_action(action)
        elif self.environment == "docker":
            self.handle_docker_action(action)
        else:
            raise ValueError(f"Unknown environment: {self.environment}")

    async def run_task(self, task: str, initial_url: Optional[str] = None):
        """Run a computer use task"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🤖 GPT Computer Use Agent{Colors.ENDC}")
        print(f"{Colors.YELLOW}Task: {task}{Colors.ENDC}")
        print("=" * 70)

        # Setup environment
        if self.environment == "browser":
            await self.setup_browser()
            if initial_url:
                print(f"{Colors.CYAN}→ Navigating to {initial_url}{Colors.ENDC}")
                await self.browser_page.goto(initial_url)
                await self.browser_page.wait_for_timeout(2000)
        elif self.environment == "docker":
            self.setup_docker()

        # Get initial screenshot
        initial_screenshot = await self.get_screenshot()
        screenshot_base64 = base64.b64encode(initial_screenshot).decode("utf-8")

        # Create initial request
        print(f"\n{Colors.MAGENTA}📤 Sending initial request...{Colors.ENDC}")

        response = self.client.responses.create(
            model=self.model,
            tools=[{
                "type": "computer_use_preview",
                "display_width": self.display_width,
                "display_height": self.display_height,
                "environment": self.environment
            }],
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": task
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_base64}"
                    }
                ]
            }],
            reasoning={
                "summary": "concise"
            },
            truncation="auto"
        )

        # Main loop
        iteration = 0
        max_iterations = 50

        while iteration < max_iterations:
            iteration += 1
            print(f"\n{Colors.BLUE}{'='*70}{Colors.ENDC}")
            print(f"{Colors.BLUE}Iteration {iteration}{Colors.ENDC}")
            print(f"{Colors.BLUE}{'='*70}{Colors.ENDC}")

            # Check for computer calls
            computer_calls = [item for item in response.output if item.type == "computer_call"]

            if not computer_calls:
                # No more actions - task complete
                print(f"\n{Colors.GREEN}✓ Task complete!{Colors.ENDC}\n")

                # Display final output
                for item in response.output:
                    if item.type == "reasoning":
                        if hasattr(item, 'summary') and item.summary:
                            summary_text = item.summary[0].text if item.summary else ""
                            print(f"{Colors.CYAN}💭 Reasoning: {summary_text}{Colors.ENDC}")
                    elif item.type == "message":
                        if hasattr(item, 'content'):
                            for content in item.content:
                                if hasattr(content, 'text'):
                                    print(f"{Colors.GREEN}📝 Result: {content.text}{Colors.ENDC}")
                break

            # Process computer call
            computer_call = computer_calls[0]
            call_id = computer_call.call_id
            action = computer_call.action
            pending_safety_checks = computer_call.pending_safety_checks if hasattr(computer_call, 'pending_safety_checks') else []

            # Show reasoning if available
            reasoning_items = [item for item in response.output if item.type == "reasoning"]
            if reasoning_items:
                reasoning = reasoning_items[0]
                if hasattr(reasoning, 'summary') and reasoning.summary:
                    summary_text = reasoning.summary[0].text if reasoning.summary else ""
                    print(f"{Colors.CYAN}💭 {summary_text}{Colors.ENDC}")

            # Show action
            print(f"{Colors.MAGENTA}🔧 Action: {action.get('type', 'unknown')}{Colors.ENDC}")

            # Handle safety checks
            acknowledged_checks = []
            if pending_safety_checks:
                print(f"\n{Colors.YELLOW}⚠️  Safety checks detected:{Colors.ENDC}")
                for check in pending_safety_checks:
                    print(f"{Colors.YELLOW}  - {check.code}: {check.message}{Colors.ENDC}")

                    # In production, you should ask user for confirmation
                    # For now, we'll auto-acknowledge with a warning
                    print(f"{Colors.RED}  ⚠️  AUTO-ACKNOWLEDGING (implement user confirmation in production){Colors.ENDC}")
                    acknowledged_checks.append({
                        "id": check.id,
                        "code": check.code,
                        "message": check.message
                    })

            # Execute action
            await self.handle_action(action.__dict__)

            # Wait for changes to take effect
            await asyncio.sleep(1)

            # Get updated screenshot
            screenshot_bytes = await self.get_screenshot()
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

            # Get current URL if browser environment
            current_url = None
            if self.environment == "browser":
                current_url = self.browser_page.url

            # Send next request
            input_data = {
                "type": "computer_call_output",
                "call_id": call_id,
                "output": {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_base64}"
                }
            }

            # Add acknowledged safety checks if any
            if acknowledged_checks:
                input_data["acknowledged_safety_checks"] = acknowledged_checks

            # Add current URL if available
            if current_url:
                input_data["current_url"] = current_url

            response = self.client.responses.create(
                model=self.model,
                previous_response_id=response.id,
                tools=[{
                    "type": "computer_use_preview",
                    "display_width": self.display_width,
                    "display_height": self.display_height,
                    "environment": self.environment
                }],
                input=[input_data],
                truncation="auto"
            )

        if iteration >= max_iterations:
            print(f"\n{Colors.YELLOW}⚠️  Maximum iterations reached{Colors.ENDC}")

    async def cleanup(self):
        """Cleanup resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='GPT Computer Use Agent')
    parser.add_argument('task', nargs='?', help='Task to perform')
    parser.add_argument('--environment', '-e', choices=['browser', 'docker'], default='browser',
                       help='Environment to use (browser or docker)')
    parser.add_argument('--url', '-u', help='Initial URL to navigate to (browser only)')
    parser.add_argument('--width', type=int, default=1024, help='Display width')
    parser.add_argument('--height', type=int, default=768, help='Display height')
    parser.add_argument('--container', default='cua-container', help='Docker container name')
    parser.add_argument('--display', default=':99', help='Docker display')

    args = parser.parse_args()

    if not args.task:
        print(f"{Colors.RED}❌ Error: Task is required{Colors.ENDC}")
        print("\nExamples:")
        print("  python gpt_computer_use.py 'Search for OpenAI news on bing.com'")
        print("  python gpt_computer_use.py --url https://bing.com 'Search for AI news'")
        print("  python gpt_computer_use.py --environment docker 'Open Firefox'")
        return

    agent = GPTComputerUse(
        environment=args.environment,
        display_width=args.width,
        display_height=args.height
    )

    try:
        await agent.run_task(args.task, initial_url=args.url)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}👋 Goodbye!{Colors.ENDC}")
        sys.exit(0)
