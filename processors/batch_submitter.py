"""
Unified Batch Job Submitter - Industrial Scale API Batch Processing

Supports both OpenAI and Anthropic batch APIs for 50% cost reduction
and massive scale operations (10,000+ requests per batch).

Copyright (c) 2025 Quantum Encoding Ltd.
"""

import json
import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)


class BatchSubmitter(ABC):
    """
    Abstract base class for batch job submission across providers
    """
    
    @abstractmethod
    async def submit_batch(self, requests: List[Dict], **kwargs) -> Dict:
        """Submit a batch job to the provider"""
        pass
    
    @abstractmethod
    async def check_status(self, batch_id: str) -> Dict:
        """Check the status of a batch job"""
        pass
    
    @abstractmethod
    async def retrieve_results(self, batch_id: str) -> List[Dict]:
        """Retrieve results from a completed batch"""
        pass
    
    @abstractmethod
    async def cancel_batch(self, batch_id: str) -> Dict:
        """Cancel a running batch job"""
        pass


class OpenAIBatchSubmitter(BatchSubmitter):
    """
    OpenAI Batch API implementation
    
    Handles file-based batch submission with JSONL format.
    Supports up to 50,000 requests per batch with 24-hour processing.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def submit_batch(self, 
                          requests: List[Dict],
                          completion_window: str = "24h",
                          metadata: Dict = None) -> Dict:
        """
        Submit batch job to OpenAI
        
        Args:
            requests: List of request dictionaries
            completion_window: "24h" (default) 
            metadata: Optional metadata for the batch
            
        Returns:
            Batch submission response with batch_id
        """
        
        # Convert requests to JSONL format
        jsonl_lines = []
        for idx, req in enumerate(requests):
            model = req.get("model", "gpt-5-nano")
            
            # Check if it's a GPT-5 model
            is_gpt5 = model.startswith("gpt-5")
            
            if is_gpt5:
                # GPT-5 uses /v1/responses endpoint with completely different format
                # Input must be an array with role/content structure
                input_text = req.get("prompt", "")
                batch_req = {
                    "custom_id": req.get("custom_id", f"request-{idx}"),
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model,
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": input_text
                                    }
                                ]
                            }
                        ],
                        "max_output_tokens": req.get("max_tokens", 32000),  # 32K default (min 16K)
                        "text": {
                            "format": {
                                "type": "text"
                            },
                            "verbosity": req.get("verbosity", "medium")
                        },
                        "reasoning": {
                            "effort": req.get("reasoning_effort", "medium"),
                            "summary": "auto"
                        },
                        "tools": [],
                        "store": True
                        # NO temperature for GPT-5!
                    }
                }
            else:
                # Legacy models use /v1/chat/completions
                batch_req = {
                    "custom_id": req.get("custom_id", f"request-{idx}"),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": req.get("messages", [{"role": "user", "content": req["prompt"]}]),
                        "temperature": req.get("temperature", 0.7),
                        "max_tokens": req.get("max_tokens", 4000)
                    }
                }
            
            jsonl_lines.append(json.dumps(batch_req))
        
        jsonl_content = "\n".join(jsonl_lines)
        
        # Upload file first
        async with aiohttp.ClientSession() as session:
            # Step 1: Upload the JSONL file
            data = aiohttp.FormData()
            data.add_field('file', jsonl_content, filename='batch_requests.jsonl', content_type='application/jsonl')
            data.add_field('purpose', 'batch')
            
            async with session.post(
                f"{self.base_url}/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=data
            ) as response:
                file_response = await response.json()
                file_id = file_response["id"]
            
            logger.info(f"📤 Uploaded batch file: {file_id}")
            
            # Step 2: Create the batch
            # Determine endpoint based on model types in requests
            first_model = requests[0].get("model", "gpt-5-nano") if requests else "gpt-5-nano"
            endpoint = "/v1/responses" if first_model.startswith("gpt-5") else "/v1/chat/completions"
            
            batch_request = {
                "input_file_id": file_id,
                "endpoint": endpoint,
                "completion_window": completion_window
            }
            
            if metadata:
                batch_request["metadata"] = metadata
            
            async with session.post(
                f"{self.base_url}/batches",
                headers=self.headers,
                json=batch_request
            ) as response:
                batch_response = await response.json()
        
        logger.info(f"✅ OpenAI Batch submitted: {batch_response['id']}")
        logger.info(f"📊 Total requests: {len(requests)}")
        logger.info(f"⏰ Completion window: {completion_window}")
        
        return batch_response
    
    async def check_status(self, batch_id: str) -> Dict:
        """
        Check batch job status
        
        Returns status: validating, in_progress, completed, failed, expired, cancelling, cancelled
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/batches/{batch_id}",
                headers=self.headers
            ) as response:
                status = await response.json()
        
        logger.info(f"📊 Batch {batch_id} status: {status['status']}")
        
        if status.get("request_counts"):
            counts = status["request_counts"]
            logger.info(f"   Completed: {counts.get('completed', 0)}/{counts.get('total', 0)}")
            if counts.get('failed', 0) > 0:
                logger.warning(f"   Failed: {counts['failed']}")
        
        return status
    
    async def retrieve_results(self, batch_id: str) -> List[Dict]:
        """
        Retrieve results from completed batch
        
        Returns list of response dictionaries
        """
        # First check if batch is complete
        status = await self.check_status(batch_id)
        
        if status["status"] != "completed":
            raise ValueError(f"Batch {batch_id} is not complete. Status: {status['status']}")
        
        output_file_id = status.get("output_file_id")
        if not output_file_id:
            raise ValueError(f"No output file for batch {batch_id}")
        
        # Download the output file
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/files/{output_file_id}/content",
                headers=self.headers
            ) as response:
                content = await response.text()
        
        # Parse JSONL results
        results = []
        for line in content.strip().split("\n"):
            if line:
                result = json.loads(line)
                results.append(result)
        
        logger.info(f"✅ Retrieved {len(results)} results from batch {batch_id}")
        
        return results
    
    async def cancel_batch(self, batch_id: str) -> Dict:
        """Cancel a running batch job"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/batches/{batch_id}/cancel",
                headers=self.headers
            ) as response:
                result = await response.json()
        
        logger.info(f"🛑 Batch {batch_id} cancelled")
        return result


class AnthropicBatchSubmitter(BatchSubmitter):
    """
    Anthropic Batch API implementation
    
    Handles direct API batch submission for Claude models.
    Supports up to 10,000 requests per batch with 24-hour processing.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
    
    async def submit_batch(self,
                          requests: List[Dict],
                          processing_priority: str = "default") -> Dict:
        """
        Submit batch job to Anthropic
        
        Args:
            requests: List of request dictionaries
            processing_priority: "default" or "urgent" (2x cost)
            
        Returns:
            Batch submission response with batch_id
        """
        
        # Format requests for Anthropic batch API
        batch_requests = []
        for idx, req in enumerate(requests):
            batch_requests.append({
                "custom_id": req.get("custom_id", f"request-{idx}"),
                "params": {
                    "model": req.get("model", "claude-3-5-sonnet-20241022"),
                    "max_tokens": req.get("max_tokens", 4096),
                    "temperature": req.get("temperature", 0.7),
                    "messages": req.get("messages", [
                        {"role": "user", "content": req["prompt"]}
                    ])
                }
            })
        
        batch_payload = {
            "requests": batch_requests,
            "processing_priority": processing_priority
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/messages/batches",
                headers=self.headers,
                json=batch_payload
            ) as response:
                batch_response = await response.json()
        
        logger.info(f"✅ Anthropic Batch submitted: {batch_response['id']}")
        logger.info(f"📊 Total requests: {len(requests)}")
        logger.info(f"⚡ Priority: {processing_priority}")
        logger.info(f"⏰ Expires at: {batch_response.get('expires_at', 'N/A')}")
        
        return batch_response
    
    async def check_status(self, batch_id: str) -> Dict:
        """
        Check batch job status
        
        Returns status: in_progress, completed, expired, canceled, failed
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/v1/messages/batches/{batch_id}",
                headers=self.headers
            ) as response:
                status = await response.json()
        
        logger.info(f"📊 Batch {batch_id} status: {status['processing_status']}")
        
        request_counts = status.get("request_counts", {})
        logger.info(f"   Processing: {request_counts.get('processing', 0)}")
        logger.info(f"   Succeeded: {request_counts.get('succeeded', 0)}")
        
        if request_counts.get('errored', 0) > 0:
            logger.warning(f"   Errored: {request_counts['errored']}")
        if request_counts.get('expired', 0) > 0:
            logger.warning(f"   Expired: {request_counts['expired']}")
        
        return status
    
    async def retrieve_results(self, batch_id: str) -> List[Dict]:
        """
        Retrieve results from completed batch
        
        Returns list of response dictionaries
        """
        # Check if batch is complete
        status = await self.check_status(batch_id)
        
        if status["processing_status"] != "ended":
            raise ValueError(f"Batch {batch_id} is not complete. Status: {status['processing_status']}")
        
        # Get results URL
        results_url = status.get("results_url")
        if not results_url:
            raise ValueError(f"No results URL for batch {batch_id}")
        
        # Download results
        async with aiohttp.ClientSession() as session:
            async with session.get(
                results_url,
                headers={"x-api-key": self.api_key}
            ) as response:
                content = await response.text()
        
        # Parse JSONL results
        results = []
        for line in content.strip().split("\n"):
            if line:
                result = json.loads(line)
                results.append(result)
        
        logger.info(f"✅ Retrieved {len(results)} results from batch {batch_id}")
        
        return results
    
    async def cancel_batch(self, batch_id: str) -> Dict:
        """Cancel a running batch job"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/messages/batches/{batch_id}/cancel",
                headers=self.headers
            ) as response:
                result = await response.json()
        
        logger.info(f"🛑 Batch {batch_id} cancelled")
        return result


class UnifiedBatchSubmitter:
    """
    Unified interface for submitting batch jobs across all providers
    
    THE COMPLETE PANTHEON: OpenAI, Anthropic, and Google Gemini.
    Automatically routes to the correct provider based on model.
    Provides consistent interface regardless of provider quirks.
    """
    
    def __init__(self, 
                 openai_key: str = None, 
                 anthropic_key: str = None,
                 google_project: str = None):
        """
        Initialize with API keys and project IDs
        
        Keys can also be set via environment variables:
        - OPENAI_API_KEY
        - ANTHROPIC_API_KEY
        - GOOGLE_CLOUD_PROJECT
        """
        import os
        
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
        self.google_project = google_project or os.getenv("GOOGLE_CLOUD_PROJECT")
        
        self.submitters = {}
        
        if self.openai_key:
            self.submitters["openai"] = OpenAIBatchSubmitter(self.openai_key)
        
        if self.anthropic_key:
            self.submitters["anthropic"] = AnthropicBatchSubmitter(self.anthropic_key)
        
        if self.google_project:
            # Import here to avoid dependency issues if not using Gemini
            from .gemini_batch_submitter import GeminiBatchSubmitter
            self.submitters["gemini"] = GeminiBatchSubmitter(project_id=self.google_project)
        
        # Check for XAI API key
        xai_key = os.getenv("XAI_API_KEY")
        if xai_key:
            from .xai_deferred_submitter import XAIDeferredSubmitter
            self.submitters["xai"] = XAIDeferredSubmitter(api_key=xai_key)
    
    def get_provider_for_model(self, model: str) -> str:
        """Determine provider from model name"""
        if "gpt" in model or "dall" in model:
            return "openai"
        elif "claude" in model:
            return "anthropic"
        elif "gemini" in model:
            return "gemini"
        elif "grok" in model:
            return "xai"
        else:
            raise ValueError(f"Unknown provider for model: {model}")
    
    async def submit_batch(self, 
                          requests: List[Dict],
                          provider: str = None,
                          **kwargs) -> Dict:
        """
        Submit batch to appropriate provider
        
        Args:
            requests: List of request dictionaries with 'prompt' and 'model'
            provider: Optional provider override ('openai' or 'anthropic')
            **kwargs: Provider-specific options
            
        Returns:
            Batch submission response
        """
        
        # Auto-detect provider if not specified
        if not provider:
            # Use the first request's model to determine provider
            first_model = requests[0].get("model", "gpt-4o-mini")
            provider = self.get_provider_for_model(first_model)
        
        if provider not in self.submitters:
            raise ValueError(f"No API key configured for provider: {provider}")
        
        submitter = self.submitters[provider]
        
        logger.info(f"🚀 Submitting batch to {provider.upper()}")
        
        # Special handling for XAI deferred completions
        if provider == "xai":
            # XAI uses a different method
            return await submitter.process_batch_with_deferred(
                requests=requests,
                wait_for_results=kwargs.get("wait_for_results", False),
                **kwargs
            )
        else:
            return await submitter.submit_batch(requests, **kwargs)
    
    async def check_status(self, batch_id: str, provider: str) -> Dict:
        """Check status of a batch job"""
        if provider not in self.submitters:
            raise ValueError(f"No API key configured for provider: {provider}")
        
        return await self.submitters[provider].check_status(batch_id)
    
    async def retrieve_results(self, batch_id: str, provider: str) -> List[Dict]:
        """Retrieve results from a completed batch"""
        if provider not in self.submitters:
            raise ValueError(f"No API key configured for provider: {provider}")
        
        return await self.submitters[provider].retrieve_results(batch_id)
    
    async def wait_for_completion(self, 
                                 batch_id: str,
                                 provider: str,
                                 check_interval: int = 60,
                                 max_wait: int = 86400) -> List[Dict]:
        """
        Wait for batch completion and retrieve results
        
        Args:
            batch_id: The batch ID to monitor
            provider: The provider ('openai' or 'anthropic')
            check_interval: Seconds between status checks (default 60)
            max_wait: Maximum seconds to wait (default 24 hours)
            
        Returns:
            List of results when batch completes
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = await self.check_status(batch_id, provider)
            
            # Check completion based on provider
            if provider == "openai":
                if status["status"] == "completed":
                    return await self.retrieve_results(batch_id, provider)
                elif status["status"] in ["failed", "expired", "cancelled"]:
                    raise Exception(f"Batch {batch_id} failed with status: {status['status']}")
            
            elif provider == "anthropic":
                if status["processing_status"] == "ended":
                    return await self.retrieve_results(batch_id, provider)
                elif status["processing_status"] in ["failed", "expired", "canceled"]:
                    raise Exception(f"Batch {batch_id} failed with status: {status['processing_status']}")
            
            elif provider == "gemini":
                if status.get("state") == "JOB_STATE_SUCCEEDED":
                    return await self.retrieve_results(batch_id, provider)
                elif status.get("state") in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
                    raise Exception(f"Batch {batch_id} failed with state: {status.get('state')}")
            
            # Wait before next check
            logger.info(f"⏳ Batch {batch_id} still processing... checking again in {check_interval}s")
            await asyncio.sleep(check_interval)
        
        raise TimeoutError(f"Batch {batch_id} did not complete within {max_wait} seconds")


# Example usage function
async def submit_batch_job(prompts: List[str], model: str = "gpt-4o-mini"):
    """
    Example function showing how to submit a batch job
    """
    
    # Prepare requests
    requests = [
        {
            "prompt": prompt,
            "model": model,
            "temperature": 0.7,
            "max_tokens": 4000
        }
        for prompt in prompts
    ]
    
    # Initialize submitter
    submitter = UnifiedBatchSubmitter()
    
    # Submit batch
    batch_response = await submitter.submit_batch(requests)
    batch_id = batch_response["id"] if "id" in batch_response else batch_response["batch_id"]
    
    logger.info(f"📋 Batch ID: {batch_id}")
    
    # Wait for completion and get results
    provider = submitter.get_provider_for_model(model)
    results = await submitter.wait_for_completion(batch_id, provider)
    
    return results