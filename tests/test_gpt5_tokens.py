#!/usr/bin/env python3
"""
Test GPT-5 models with new token settings
"""

import asyncio
import os
from dotenv import load_dotenv
from providers.openai_provider import OpenaiProvider

async def test_gpt5_tokens():
    """Test GPT-5 models with 16K+ token outputs"""
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    print("🚀 Testing GPT-5 models with new token limits")
    print("=" * 60)
    
    # Test with each model
    models = [
        ("gpt-5-nano-2025-08-07", 16000),
        ("gpt-5-mini-2025-08-07", 20000),
        ("gpt-5-2025-08-07", 25000)
    ]
    
    config = {
        "api_key": api_key,
        "endpoint": "https://api.openai.com/v1"
    }
    
    for model, tokens in models:
        print(f"\n🤖 Testing {model}")
        print(f"📊 Requesting {tokens:,} tokens")
        print("-" * 40)
        
        provider = OpenaiProvider(config)
        
        try:
            response = await provider.complete(
                prompt="Write a comprehensive guide on building scalable microservices. Include architecture patterns, implementation details, best practices, monitoring strategies, and deployment approaches. Be extremely detailed and thorough.",
                model=model,
                max_tokens=tokens
            )
            
            # Check response
            if isinstance(response, str):
                response_len = len(response)
                print(f"✅ Response received: {response_len:,} characters")
                
                # Show if it's a raw response object or actual text
                if response.startswith('[ResponseReasoningItem'):
                    print("📦 Response format: GPT-5 object (needs parsing)")
                else:
                    print("📄 Response format: Text content")
                    # Show first 200 chars
                    preview = response[:200] + "..." if len(response) > 200 else response
                    print(f"Preview: {preview}")
                
                # Estimate tokens
                estimated_tokens = response_len // 4
                print(f"🎯 Estimated tokens: {estimated_tokens:,}")
                
                if estimated_tokens >= 10000:
                    print("✨ Successfully generated 10K+ tokens!")
                elif estimated_tokens >= 5000:
                    print("⚠️  Generated 5K+ tokens (partial success)")
                else:
                    print("❌ Generated less than 5K tokens")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        finally:
            if hasattr(provider, 'client'):
                try:
                    await provider.client.close()
                except:
                    pass
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("\nToken settings updated:")
    print("• Minimum: 16K tokens")
    print("• Default: 32K tokens")  
    print("• Maximum: 128K tokens")

if __name__ == "__main__":
    asyncio.run(test_gpt5_tokens())