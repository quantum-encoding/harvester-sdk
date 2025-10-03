"""
xAI Batch Provider using Deferred Chat Completions
Allows submitting multiple requests and retrieving results later
"""
import logging
import json
import asyncio
import aiofiles
import aiohttp
from typing import Any, Dict, List, Union, Optional
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import os

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class XaiBatchProvider(BaseProvider):
    """
    xAI Batch provider using deferred chat completions.
    Submits requests and retrieves results asynchronously.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key: str = config.get("api_key") or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("xAI API key missing. Set XAI_API_KEY environment variable or add to config")
        
        # xAI endpoints
        self.base_url: str = config.get("endpoint", "https://api.x.ai/v1")
        self.chat_endpoint = f"{self.base_url}/chat/completions"
        self.deferred_endpoint = f"{self.base_url}/chat/deferred-completion"
        
        # Batch settings
        self.batch_dir = Path(config.get("batch_dir", "./batch_jobs"))
        self.batch_dir.mkdir(exist_ok=True, parents=True)
        
        # Model settings (same as regular xAI provider)
        self.model_settings: Dict[str, Dict[str, Any]] = {
            "grok-4": {
                "max_tokens": 131072,
                "cost_per_million_input": 2.00,
                "cost_per_million_output": 10.00,
                "context_window": 256000,
                "supports_deferred": True
            },
            "grok-3": {
                "max_tokens": 131072,
                "cost_per_million_input": 15.00,
                "cost_per_million_output": 60.00,
                "context_window": 131072,
                "supports_deferred": True
            },
            "grok-3-mini": {
                "max_tokens": 131072,
                "cost_per_million_input": 1.00,
                "cost_per_million_output": 4.00,
                "context_window": 131072,
                "supports_deferred": True
            },
            "grok-3-mini-fast": {
                "max_tokens": 131072,
                "cost_per_million_input": 1.00,
                "cost_per_million_output": 4.00,
                "context_window": 131072,
                "supports_deferred": True
            }
        }
        
        self.aliases: Dict[str, str] = config.get("aliases", {})
        self.default_model = config.get("default_model", "grok-3-mini")
        
        # Track pending requests
        self.pending_requests: List[Dict] = []
        self.deferred_requests: Dict[str, Dict] = {}  # Maps request_id to request info
        
        logger.info(f"xAI Batch provider initialized with {len(self.model_settings)} models")

    def resolve_model_alias(self, alias: str) -> str:
        """Resolve model alias to actual model name"""
        return self.aliases.get(alias, alias)

    async def add_request(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model: str,
        custom_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Add a request to the batch queue.
        
        Args:
            prompt: The prompt to send
            model: Model to use
            custom_id: Custom identifier for this request
            **kwargs: Additional parameters
            
        Returns:
            The custom_id for tracking this request
        """
        actual_model = self.resolve_model_alias(model)
        
        if actual_model not in self.model_settings:
            logger.warning(f"Unknown model '{actual_model}', falling back to {self.default_model}")
            actual_model = self.default_model
        
        if not self.model_settings[actual_model].get("supports_deferred", False):
            raise ValueError(f"Model {actual_model} does not support deferred completions")
        
        # Generate custom_id if not provided
        if not custom_id:
            content = str(prompt) if isinstance(prompt, str) else json.dumps(prompt)
            custom_id = f"req_{hashlib.md5(content.encode()).hexdigest()[:8]}_{len(self.pending_requests)}"
        
        # Prepare messages
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are Grok, a highly intelligent, helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = prompt
        
        settings = self.model_settings[actual_model]
        
        # Build request
        request = {
            "custom_id": custom_id,
            "model": actual_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", settings["max_tokens"]),
            "stream": False
        }
        
        # Add reasoning_effort for supported models (not grok-4)
        if actual_model != "grok-4" and kwargs.get("reasoning_effort"):
            request["reasoning_effort"] = kwargs["reasoning_effort"]
        
        self.pending_requests.append(request)
        logger.info(f"Added request {custom_id} to batch queue")
        
        return custom_id

    async def submit_batch(
        self,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Submit all pending requests as deferred completions.
        
        Returns:
            Batch ID for tracking
        """
        if not self.pending_requests:
            raise ValueError("No requests to submit")
        
        # Create batch tracking file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = f"xai_batch_{timestamp}"
        batch_file = self.batch_dir / f"{batch_id}_requests.json"
        
        # Save requests to file for tracking
        with open(batch_file, 'w') as f:
            json.dump({
                "batch_id": batch_id,
                "description": description,
                "metadata": metadata,
                "requests": self.pending_requests,
                "created_at": timestamp
            }, f, indent=2)
        
        # Submit each request as a deferred completion
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            for request in self.pending_requests:
                try:
                    # Submit deferred request
                    async with session.post(
                        self.chat_endpoint,
                        headers=headers,
                        json={
                            **request,
                            "defer": True  # Enable deferred mode
                        }
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            request_id = result.get("request_id")
                            
                            if request_id:
                                # Store mapping of request_id to custom_id
                                self.deferred_requests[request_id] = {
                                    "custom_id": request["custom_id"],
                                    "batch_id": batch_id,
                                    "submitted_at": datetime.now().isoformat()
                                }
                                logger.info(f"Submitted deferred request: {request['custom_id']} -> {request_id}")
                            else:
                                logger.error(f"No request_id in response for {request['custom_id']}")
                        else:
                            error_text = await response.text()
                            logger.error(f"Failed to submit {request['custom_id']}: {response.status} - {error_text}")
                            
                except Exception as e:
                    logger.error(f"Error submitting request {request['custom_id']}: {e}")
        
        # Save deferred request mappings
        mappings_file = self.batch_dir / f"{batch_id}_mappings.json"
        with open(mappings_file, 'w') as f:
            json.dump(self.deferred_requests, f, indent=2)
        
        logger.info(f"Submitted batch {batch_id} with {len(self.pending_requests)} requests")
        
        # Clear pending requests
        self.pending_requests = []
        
        return batch_id

    async def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Check the status of a submitted batch.
        
        Args:
            batch_id: The batch ID to check
            
        Returns:
            Status information dict
        """
        mappings_file = self.batch_dir / f"{batch_id}_mappings.json"
        
        if not mappings_file.exists():
            raise ValueError(f"Batch {batch_id} not found")
        
        with open(mappings_file, 'r') as f:
            deferred_requests = json.load(f)
        
        # Check status of each deferred request
        total = len(deferred_requests)
        completed = 0
        failed = 0
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            for request_id in deferred_requests.keys():
                try:
                    # Check if result is ready
                    url = f"{self.deferred_endpoint}/{request_id}"
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            # Result is ready
                            completed += 1
                        elif response.status == 202:
                            # Still processing (Accepted)
                            pass
                        else:
                            # Error or not found
                            failed += 1
                except Exception as e:
                    logger.error(f"Error checking status for {request_id}: {e}")
                    failed += 1
        
        return {
            "batch_id": batch_id,
            "status": "completed" if completed == total else "processing",
            "request_counts": {
                "total": total,
                "completed": completed,
                "failed": failed,
                "pending": total - completed - failed
            }
        }

    async def retrieve_batch_results(
        self, 
        batch_id: str,
        timeout: timedelta = timedelta(minutes=10),
        interval: timedelta = timedelta(seconds=10)
    ) -> Dict[str, str]:
        """
        Retrieve results from a batch of deferred completions.
        
        Args:
            batch_id: The batch ID to retrieve
            timeout: Maximum time to wait for all results
            interval: How often to check for results
            
        Returns:
            Dict mapping custom_id to response content
        """
        mappings_file = self.batch_dir / f"{batch_id}_mappings.json"
        
        if not mappings_file.exists():
            raise ValueError(f"Batch {batch_id} not found")
        
        with open(mappings_file, 'r') as f:
            deferred_requests = json.load(f)
        
        results = {}
        start_time = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Keep checking until all results are retrieved or timeout
            while deferred_requests and datetime.now() - start_time < timeout:
                completed_ids = []
                
                for request_id, info in list(deferred_requests.items()):
                    try:
                        # Try to retrieve result
                        url = f"{self.deferred_endpoint}/{request_id}"
                        async with session.get(url, headers=headers) as response:
                            if response.status == 200:
                                # Result is ready
                                result = await response.json()
                                
                                # Extract content from response
                                if "choices" in result and result["choices"]:
                                    content = result["choices"][0]["message"]["content"]
                                    
                                    # Store with custom_id as key
                                    custom_id = info["custom_id"]
                                    results[custom_id] = content
                                    completed_ids.append(request_id)
                                    
                                    logger.info(f"Retrieved result for {custom_id}")
                                    
                                    # Log reasoning content if available
                                    if "reasoning_content" in result["choices"][0]["message"]:
                                        logger.debug(f"Reasoning available for {custom_id}")
                                        
                            elif response.status == 202:
                                # Still processing
                                logger.debug(f"Request {info['custom_id']} still processing")
                            else:
                                # Error or not found
                                error_text = await response.text()
                                logger.error(f"Failed to retrieve {info['custom_id']}: {response.status} - {error_text}")
                                results[info["custom_id"]] = None
                                completed_ids.append(request_id)
                                
                    except Exception as e:
                        logger.error(f"Error retrieving {info['custom_id']}: {e}")
                        results[info["custom_id"]] = None
                        completed_ids.append(request_id)
                
                # Remove completed requests
                for request_id in completed_ids:
                    del deferred_requests[request_id]
                
                if deferred_requests:
                    # Wait before checking again
                    logger.info(f"Waiting {interval.seconds}s before checking {len(deferred_requests)} pending requests...")
                    await asyncio.sleep(interval.seconds)
        
        # Save results
        results_file = self.batch_dir / f"{batch_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Retrieved {len(results)} results for batch {batch_id}")
        
        return results

    async def complete(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model: str,
        **kwargs
    ) -> str:
        """
        Add to batch and optionally wait for completion.
        For compatibility with base provider interface.
        """
        custom_id = await self.add_request(prompt, model, **kwargs)
        return f"Added to batch with ID: {custom_id}"

    def estimate_tokens(self, prompt: str, response: str = "", model: str = None) -> int:
        """Estimate token count for prompt and response"""
        # Simple estimation: ~4 chars per token
        total_chars = len(prompt) + len(response)
        return int(total_chars / 4)
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage"""
        actual_model = self.resolve_model_alias(model)
        settings = self.model_settings.get(
            actual_model,
            self.model_settings[self.default_model]
        )
        input_tokens = int(tokens * 0.7)
        output_tokens = tokens - input_tokens
        input_cost = (input_tokens / 1_000_000) * settings["cost_per_million_input"]
        output_cost = (output_tokens / 1_000_000) * settings["cost_per_million_output"]
        return input_cost + output_cost

    def get_model_info(self, model: str) -> Dict[str, Any]:
        actual_model = self.resolve_model_alias(model)
        return self.model_settings.get(actual_model, self.model_settings[self.default_model])

    def list_available_models(self) -> List[str]:
        return list(self.model_settings.keys())

    async def cleanup(self):
        """Submit any pending requests before cleanup"""
        if self.pending_requests:
            logger.warning(f"Submitting {len(self.pending_requests)} pending requests before cleanup")
            await self.submit_batch(description="Auto-submitted on cleanup")

    def __repr__(self) -> str:
        return f"XaiBatchProvider(models={len(self.model_settings)}, pending={len(self.pending_requests)})"