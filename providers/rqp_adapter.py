#!/usr/bin/env python3
"""
RQP Provider Adapter - Convert RQP templates to provider-specific formats
Enables any AI provider to process RQP templates through format conversion.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class RQPProviderAdapter(ABC):
    """Base class for RQP to provider format conversion"""
    
    @abstractmethod
    def convert_to_messages(self, rqp_template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert RQP template to provider message format"""
        pass
    
    @abstractmethod
    def convert_to_request(self, rqp_template: Dict[str, Any]) -> Dict[str, Any]:
        """Convert RQP template to complete API request"""
        pass
    
    @abstractmethod
    def parse_response(self, response: Any, rqp_template: Dict[str, Any]) -> Dict[str, Any]:
        """Parse provider response back to RQP format"""
        pass
    
    def extract_system_prompt(self, rqp_template: Dict[str, Any]) -> str:
        """Extract system prompt from RQP template"""
        handshake = rqp_template.get("handshake", {})
        
        system_parts = [
            f"You are a {handshake.get('target_persona', 'Domain Expert')}.",
            f"You are interacting with a {handshake.get('requester_persona', 'Technical Architect')}.",
            f"Purpose: {handshake.get('requester_intent', 'Provide expert knowledge')}",
            "",
            "Response Requirements:",
            f"- Format: {handshake.get('response_modality', {}).get('format', 'structured')}",
            f"- Tone: {handshake.get('response_modality', {}).get('tone', 'technical')}",
            f"- Detail Level: {handshake.get('response_modality', {}).get('content_density', 'comprehensive')}",
            "",
            "Quality Standards:",
            f"- {handshake.get('processing_directives', {}).get('source_requirement', 'Cite sources')}",
            f"- {handshake.get('processing_directives', {}).get('implementation_threshold', 'Complete implementations')}",
            "- Avoid speculation, provide only proven patterns",
            "- Include failure modes and mitigation strategies"
        ]
        
        return "\n".join(system_parts)
    
    def extract_user_prompt(self, rqp_template: Dict[str, Any]) -> str:
        """Extract user prompt from RQP template"""
        payload = rqp_template.get("payload", {})
        context = payload.get("context", {})
        questions = payload.get("questions", [])
        
        prompt_parts = []
        
        # Context section
        prompt_parts.append("Context:")
        prompt_parts.append(f"- Deployment: {context.get('deployment_target', 'Production')}")
        prompt_parts.append(f"- Performance Goal: {context.get('performance_goal', 'Optimal')}")
        
        if "benchmark_baseline" in context:
            baseline = context["benchmark_baseline"]
            prompt_parts.append(f"- Hardware: {baseline.get('hardware', 'Standard')}")
            prompt_parts.append(f"- Current Performance: {baseline.get('throughput', 'baseline')}")
            prompt_parts.append(f"- Measured Latency: {baseline.get('latency', 'baseline')}")
        
        prompt_parts.append("")
        
        # Expected format
        response_format = payload.get("expected_response_format", {})
        prompt_parts.append("Please provide your response as a JSON array with this structure:")
        prompt_parts.append("```json")
        prompt_parts.append(json.dumps(response_format.get("schema_per_answer", {}), indent=2))
        prompt_parts.append("```")
        prompt_parts.append("")
        
        # Questions
        prompt_parts.append("Questions to answer:")
        for i, question in enumerate(questions, 1):
            prompt_parts.append(f"\n{i}. {question['question']}")
            prompt_parts.append(f"   ID: {question.get('id', f'q{i}')}")
            
            if "hardware_specifics" in question:
                prompt_parts.append(f"   Hardware Requirements: {json.dumps(question['hardware_specifics'])}")
            
            if "validation_requirements" in question:
                prompt_parts.append(f"   Validation: {json.dumps(question['validation_requirements'])}")
            
            if "expected_deliverable" in question:
                prompt_parts.append(f"   Expected: {question['expected_deliverable']}")
        
        prompt_parts.append("\nProvide complete, production-ready answers for each question.")
        
        return "\n".join(prompt_parts)


class OpenAIRQPAdapter(RQPProviderAdapter):
    """Convert RQP templates for OpenAI API"""
    
    def convert_to_messages(self, rqp_template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert to OpenAI message format"""
        messages = [
            {
                "role": "system",
                "content": self.extract_system_prompt(rqp_template)
            },
            {
                "role": "user",
                "content": self.extract_user_prompt(rqp_template)
            }
        ]
        
        return messages
    
    def convert_to_request(self, rqp_template: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to complete OpenAI API request"""
        return {
            "model": "gpt-5-mini-2025-08-07",
            "messages": self.convert_to_messages(rqp_template),
            "temperature": 0.2,  # Lower temperature for technical accuracy
            "max_tokens": 4096,
            "response_format": {"type": "json_object"}  # Force JSON response
        }
    
    def parse_response(self, response: Any, rqp_template: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenAI response to RQP format"""
        # Extract content from OpenAI response
        if isinstance(response, dict) and "choices" in response:
            content = response["choices"][0]["message"]["content"]
        else:
            content = str(response)
        
        # Parse JSON if possible
        try:
            if isinstance(content, str):
                answers = json.loads(content)
            else:
                answers = content
        except:
            # Fallback to wrapping content
            answers = [{
                "answer_id": "response_1",
                "response": content,
                "quality_metrics": {
                    "confidence": 0.7,
                    "source_quality": "B",
                    "validation_status": "theoretical"
                }
            }]
        
        # Ensure it's a list
        if not isinstance(answers, list):
            answers = [answers]
        
        return {
            "batch_id": rqp_template["payload"]["batch_id"],
            "answers": answers,
            "metadata": {
                "provider": "openai",
                "model": response.get("model", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        }


class AnthropicRQPAdapter(RQPProviderAdapter):
    """Convert RQP templates for Anthropic Claude API"""
    
    def convert_to_messages(self, rqp_template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert to Claude message format"""
        # Claude combines system and user in a specific way
        system_prompt = self.extract_system_prompt(rqp_template)
        user_prompt = self.extract_user_prompt(rqp_template)
        
        return [
            {
                "role": "user",
                "content": f"{system_prompt}\n\n{user_prompt}"
            }
        ]
    
    def convert_to_request(self, rqp_template: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to complete Anthropic API request"""
        return {
            "model": "claude-3-opus-20240229",
            "messages": self.convert_to_messages(rqp_template),
            "max_tokens": 4096,
            "temperature": 0.2,
            "system": self.extract_system_prompt(rqp_template)  # Anthropic uses separate system
        }
    
    def parse_response(self, response: Any, rqp_template: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Claude response to RQP format"""
        # Extract content from Anthropic response
        if isinstance(response, dict) and "content" in response:
            content = response["content"][0]["text"] if isinstance(response["content"], list) else response["content"]
        else:
            content = str(response)
        
        # Parse JSON from content
        try:
            import re
            # Look for JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                answers = json.loads(json_match.group())
            else:
                raise ValueError("No JSON array found")
        except:
            # Fallback
            answers = [{
                "answer_id": "response_1",
                "response": content,
                "quality_metrics": {
                    "confidence": 0.8,
                    "source_quality": "B",
                    "validation_status": "theoretical"
                }
            }]
        
        return {
            "batch_id": rqp_template["payload"]["batch_id"],
            "answers": answers,
            "metadata": {
                "provider": "anthropic",
                "model": response.get("model", "claude"),
                "timestamp": datetime.now().isoformat()
            }
        }


class GoogleRQPAdapter(RQPProviderAdapter):
    """Convert RQP templates for Google AI (Gemini) API"""
    
    def convert_to_messages(self, rqp_template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert to Gemini message format"""
        system_prompt = self.extract_system_prompt(rqp_template)
        user_prompt = self.extract_user_prompt(rqp_template)
        
        # Gemini uses 'parts' structure
        return [
            {
                "role": "user",
                "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
            }
        ]
    
    def convert_to_request(self, rqp_template: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to complete Google AI API request"""
        return {
            "contents": self.convert_to_messages(rqp_template),
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json"  # Request JSON response
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
        }
    
    def parse_response(self, response: Any, rqp_template: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Gemini response to RQP format"""
        # Extract content from Gemini response
        if isinstance(response, dict) and "candidates" in response:
            content = response["candidates"][0]["content"]["parts"][0]["text"]
        else:
            content = str(response)
        
        # Parse JSON
        try:
            answers = json.loads(content)
            if not isinstance(answers, list):
                answers = [answers]
        except:
            answers = [{
                "answer_id": "response_1",
                "response": content,
                "quality_metrics": {
                    "confidence": 0.75,
                    "source_quality": "B",
                    "validation_status": "theoretical"
                }
            }]
        
        return {
            "batch_id": rqp_template["payload"]["batch_id"],
            "answers": answers,
            "metadata": {
                "provider": "google",
                "model": "gemini-2.5-flash",
                "timestamp": datetime.now().isoformat()
            }
        }


class RQPAdapterFactory:
    """Factory for creating appropriate RQP adapters"""
    
    @staticmethod
    def create_adapter(provider: str) -> RQPProviderAdapter:
        """Create adapter for specified provider"""
        adapters = {
            "openai": OpenAIRQPAdapter,
            "anthropic": AnthropicRQPAdapter,
            "claude": AnthropicRQPAdapter,  # Alias
            "google": GoogleRQPAdapter,
            "gemini": GoogleRQPAdapter,  # Alias
            "vertex": GoogleRQPAdapter,  # Alias
        }
        
        provider_lower = provider.lower()
        if provider_lower in adapters:
            return adapters[provider_lower]()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @staticmethod
    def get_curl_command(provider: str, rqp_template: Dict[str, Any], api_key: str) -> str:
        """Generate CURL command for provider"""
        adapter = RQPAdapterFactory.create_adapter(provider)
        request = adapter.convert_to_request(rqp_template)
        
        if provider.lower() in ["openai"]:
            return f"""curl https://api.openai.com/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {api_key}" \\
  -d '{json.dumps(request, indent=2)}'"""
        
        elif provider.lower() in ["anthropic", "claude"]:
            return f"""curl https://api.anthropic.com/v1/messages \\
  -H "content-type: application/json" \\
  -H "x-api-key: {api_key}" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{json.dumps(request, indent=2)}'"""
        
        elif provider.lower() in ["google", "gemini"]:
            return f"""curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(request, indent=2)}'"""
        
        else:
            return f"# Unsupported provider: {provider}"


# Example usage
if __name__ == "__main__":
    # Example RQP template
    rqp_template = {
        "handshake": {
            "requester_persona": "Performance Architect",
            "target_persona": "Systems Engineer",
            "requester_intent": "Optimize image processing"
        },
        "payload": {
            "batch_id": "test_batch_001",
            "questions": [
                {
                    "id": "q1",
                    "question": "How to implement zero-copy in Rust?",
                    "hardware_specifics": {"cpu": "x86_64"}
                }
            ],
            "context": {
                "deployment_target": "Cloud Run",
                "performance_goal": "30 images/sec"
            }
        }
    }
    
    # Test each adapter
    for provider in ["openai", "anthropic", "google"]:
        print(f"\n{provider.upper()} Adapter Test:")
        print("=" * 50)
        
        adapter = RQPAdapterFactory.create_adapter(provider)
        messages = adapter.convert_to_messages(rqp_template)
        print("Messages:", json.dumps(messages, indent=2))
        
        print("\nCURL Command:")
        print(RQPAdapterFactory.get_curl_command(provider, rqp_template, "YOUR_API_KEY"))