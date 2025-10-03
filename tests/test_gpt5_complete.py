#!/usr/bin/env python3
"""
Comprehensive GPT-5 model testing with proper token settings
Tests all three GPT-5 models with various token configurations
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from providers.openai_provider import OpenaiProvider

async def test_gpt5_models():
    """Test all GPT-5 models with different token settings"""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    # Test configurations
    test_cases = [
        {
            "model": "gpt-5-nano-2025-08-07",
            "prompt": "Write a comprehensive technical analysis of quantum computing's impact on cryptography. Include current state, future implications, and specific algorithms affected.",
            "max_tokens": 20000,  # 20K tokens
            "description": "Technical analysis (20K tokens)"
        },
        {
            "model": "gpt-5-mini-2025-08-07", 
            "prompt": "Create a detailed business plan for an AI-powered healthcare startup. Include market analysis, revenue models, competitive landscape, technical architecture, and 5-year projections.",
            "max_tokens": 30000,  # 30K tokens
            "description": "Business plan (30K tokens)"
        },
        {
            "model": "gpt-5-2025-08-07",
            "prompt": "Write a complete tutorial on building a distributed microservices architecture. Cover design patterns, implementation details, deployment strategies, monitoring, and include code examples.",
            "max_tokens": 40000,  # 40K tokens
            "description": "Architecture tutorial (40K tokens)"
        }
    ]
    
    print("🚀 Starting comprehensive GPT-5 model testing")
    print("=" * 60)
    
    for test in test_cases:
        print(f"\n📝 Test: {test['description']}")
        print(f"🤖 Model: {test['model']}")
        print(f"📊 Max tokens: {test['max_tokens']:,}")
        print("-" * 40)
        
        config = {
            "api_key": api_key,
            "endpoint": "https://api.openai.com/v1"
        }
        
        provider = OpenaiProvider(config)
        
        try:
            # Make the API call
            print("⏳ Generating response...")
            response = await provider.complete(
                prompt=test['prompt'],
                model=test['model'],
                max_tokens=test['max_tokens'],
                verbosity="high",
                reasoning_effort="high"
            )
            
            # Parse the response
            if isinstance(response, str) and response.startswith('['):
                # Response is a string representation of response objects
                print(f"📥 Raw response type: GPT-5 response object")
                response_length = len(response)
            else:
                response_length = len(response)
            
            print(f"✅ Response received!")
            print(f"📏 Response length: {response_length:,} characters")
            
            # Show first 500 chars of response
            preview = response[:500] if len(response) > 500 else response
            print(f"📄 Preview: {preview}...")
            
            # Estimate token count (rough approximation)
            estimated_tokens = response_length // 4
            print(f"🎯 Estimated output tokens: {estimated_tokens:,}")
            
            # Check if we got substantial output
            if estimated_tokens < 1000:
                print("⚠️  Warning: Response shorter than expected")
            else:
                print("✨ Response meets expected length")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
        finally:
            # Cleanup
            if hasattr(provider, 'client') and hasattr(provider.client, 'close'):
                try:
                    await provider.client.close()
                except:
                    pass
        
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("\n📊 Summary:")
    print("• GPT-5 models support 16K-128K output tokens")
    print("• Default is now 32K tokens")
    print("• All models use /v1/responses endpoint")
    print("• No temperature parameter for GPT-5")

if __name__ == "__main__":
    asyncio.run(test_gpt5_models())