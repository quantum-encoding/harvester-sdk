#!/usr/bin/env python3
"""
Test GPT-5 models with simple hello message
"""

import asyncio
import os
from dotenv import load_dotenv
from providers.openai_provider import OpenaiProvider

async def test_gpt5():
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    # Test each GPT-5 model
    models = ["gpt-5-nano-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-5-2025-08-07"]
    
    for model in models:
        print(f"\n🔧 Testing {model}...")
        
        config = {
            "api_key": api_key,
            "endpoint": "https://api.openai.com/v1"
        }
        
        provider = OpenaiProvider(config)
        
        try:
            # Use the complete method
            response = await provider.complete(
                prompt="Say hello",
                model=model,
                max_tokens=100
            )
            
            print(f"✅ {model} response: {response[:200]}...")
            
        except Exception as e:
            print(f"❌ {model} error: {e}")
        finally:
            # Cleanup if method exists
            if hasattr(provider, 'cleanup'):
                try:
                    await provider.cleanup()
                except:
                    pass
    
    print("\n✨ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_gpt5())