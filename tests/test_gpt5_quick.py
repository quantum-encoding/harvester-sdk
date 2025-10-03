#!/usr/bin/env python3
"""
Quick GPT-5 test with smaller token counts
"""

import asyncio
import os
from dotenv import load_dotenv
from providers.openai_provider import OpenaiProvider

async def quick_test():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key")
        return
    
    config = {"api_key": api_key, "endpoint": "https://api.openai.com/v1"}
    
    # Test each model with modest token count
    for model in ["gpt-5-nano-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-5-2025-08-07"]:
        print(f"\n🤖 Testing {model} with 1000 tokens")
        
        provider = OpenaiProvider(config)
        try:
            response = await provider.complete(
                prompt="List 10 benefits of microservices architecture",
                model=model,
                max_tokens=1000
            )
            
            print(f"✅ {model}: {len(response)} chars received")
            
            # Check response format
            if "[ResponseReasoningItem" in str(response):
                print("   Format: GPT-5 response object")
            else:
                print(f"   Preview: {str(response)[:100]}...")
                
        except Exception as e:
            print(f"❌ {model}: {e}")
        finally:
            if hasattr(provider, 'client'):
                try:
                    await provider.client.close()
                except:
                    pass

if __name__ == "__main__":
    asyncio.run(quick_test())