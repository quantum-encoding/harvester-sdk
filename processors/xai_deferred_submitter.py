"""
XAI Deferred Completions Submitter - The Fourth Horseman

A hybrid between real-time and batch processing. Submit immediately,
retrieve later. The perfect middle ground for flexible workloads.

Copyright (c) 2025 Quantum Encoding Ltd.
"""

import json
import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import time
import os

logger = logging.getLogger(__name__)


class XAIDeferredSubmitter:
    """
    XAI Deferred Completions implementation
    
    The fourth pillar of our processing empire - neither fully sync nor batch,
    but a powerful hybrid that offers immediate submission with async retrieval.
    """
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.x.ai/v1"):
        """
        Initialize XAI Deferred Submitter
        
        Args:
            api_key: XAI API key (or set XAI_API_KEY env var)
            base_url: XAI API base URL
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI API key required. Set XAI_API_KEY or pass api_key")
        
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Track submitted requests for management
        self.pending_requests = {}
        
        logger.info(f"⚡ XAI Deferred Submitter initialized")
    
    async def submit_deferred(self, 
                             prompt: str,
                             model: str = "grok-4",
                             system: str = None,
                             temperature: float = 0.7,
                             max_tokens: int = 8192,
                             custom_id: str = None) -> Dict:
        """
        Submit a single deferred completion request
        
        Args:
            prompt: The user prompt
            model: XAI model (grok-4, grok-4-turbo, etc.)
            system: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            custom_id: Custom identifier for tracking
            
        Returns:
            Dict with request_id and submission details
        """
        
        # Prepare messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request body
        request_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False  # Deferred completions don't support streaming
        }
        
        # Submit deferred request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers={**self.headers, "X-Deferred": "true"},  # Special header for deferred
                json=request_body
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Deferred submission failed: {error_text}")
                
                result = await response.json()
        
        request_id = result.get("request_id")
        if not request_id:
            raise ValueError("No request_id in response")
        
        # Track the request
        submission_data = {
            "request_id": request_id,
            "custom_id": custom_id or request_id,
            "model": model,
            "submitted_at": datetime.now().isoformat(),
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "status": "pending"
        }
        
        self.pending_requests[request_id] = submission_data
        
        logger.info(f"✅ XAI Deferred request submitted: {request_id}")
        
        return submission_data
    
    async def submit_batch_deferred(self,
                                   requests: List[Dict],
                                   model: str = "grok-4",
                                   max_concurrent: int = 10) -> List[Dict]:
        """
        Submit multiple deferred requests concurrently
        
        This is the key innovation - we can submit many deferred requests
        in parallel, each getting its own request_id for later retrieval.
        
        Args:
            requests: List of request dictionaries with 'prompt' key
            model: Default model for all requests
            max_concurrent: Maximum concurrent submissions
            
        Returns:
            List of submission results with request_ids
        """
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def submit_single(request: Dict, index: int) -> Dict:
            async with semaphore:
                try:
                    result = await self.submit_deferred(
                        prompt=request.get("prompt", ""),
                        model=request.get("model", model),
                        system=request.get("system"),
                        temperature=request.get("temperature", 0.7),
                        max_tokens=request.get("max_tokens", 8192),
                        custom_id=request.get("custom_id", f"request-{index}")
                    )
                    return result
                except Exception as e:
                    logger.error(f"Failed to submit request {index}: {str(e)}")
                    return {
                        "error": str(e),
                        "custom_id": request.get("custom_id", f"request-{index}"),
                        "status": "failed"
                    }
        
        # Submit all requests concurrently
        logger.info(f"🚀 Submitting {len(requests)} deferred requests...")
        
        tasks = [submit_single(req, idx) for idx, req in enumerate(requests)]
        results = await asyncio.gather(*tasks)
        
        successful = sum(1 for r in results if "request_id" in r)
        failed = len(results) - successful
        
        logger.info(f"✅ Submitted {successful} requests successfully")
        if failed > 0:
            logger.warning(f"❌ {failed} requests failed")
        
        return results
    
    async def retrieve_result(self, request_id: str) -> Optional[Dict]:
        """
        Retrieve a single deferred completion result
        
        Returns None if not ready (202 status), or the completion if ready.
        Note: Results can only be retrieved ONCE within 24 hours.
        
        Args:
            request_id: The request ID from submission
            
        Returns:
            Completion result or None if not ready
        """
        
        endpoint = f"{self.base_url}/chat/deferred-completion/{request_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                endpoint,
                headers=self.headers
            ) as response:
                if response.status == 202:
                    # Not ready yet
                    logger.debug(f"Request {request_id} not ready (202)")
                    return None
                elif response.status == 200:
                    # Ready!
                    result = await response.json()
                    
                    # Update tracking
                    if request_id in self.pending_requests:
                        self.pending_requests[request_id]["status"] = "completed"
                        self.pending_requests[request_id]["retrieved_at"] = datetime.now().isoformat()
                    
                    logger.info(f"✅ Retrieved result for {request_id}")
                    return result
                elif response.status == 404:
                    # Either expired or already retrieved
                    error_msg = "Request not found (expired or already retrieved)"
                    logger.error(f"❌ {request_id}: {error_msg}")
                    
                    if request_id in self.pending_requests:
                        self.pending_requests[request_id]["status"] = "expired"
                    
                    raise ValueError(error_msg)
                else:
                    error_text = await response.text()
                    raise Exception(f"Retrieval failed ({response.status}): {error_text}")
    
    async def poll_for_result(self,
                             request_id: str,
                             timeout: timedelta = timedelta(minutes=10),
                             interval: timedelta = timedelta(seconds=10)) -> Dict:
        """
        Poll for a deferred result until ready or timeout
        
        This matches the XAI SDK's defer() method behavior.
        
        Args:
            request_id: The request ID to poll
            timeout: Maximum time to wait
            interval: Time between polls
            
        Returns:
            Completion result when ready
            
        Raises:
            TimeoutError if timeout exceeded
        """
        
        start_time = datetime.now()
        timeout_seconds = timeout.total_seconds()
        interval_seconds = interval.total_seconds()
        
        logger.info(f"⏳ Polling {request_id} (timeout: {timeout}, interval: {interval})")
        
        while (datetime.now() - start_time).total_seconds() < timeout_seconds:
            result = await self.retrieve_result(request_id)
            
            if result is not None:
                return result
            
            # Wait before next poll
            await asyncio.sleep(interval_seconds)
        
        raise TimeoutError(f"Request {request_id} did not complete within {timeout}")
    
    async def retrieve_batch_results(self,
                                    request_ids: List[str],
                                    timeout: timedelta = timedelta(minutes=30),
                                    interval: timedelta = timedelta(seconds=10),
                                    max_concurrent: int = 10) -> List[Dict]:
        """
        Retrieve multiple deferred results concurrently
        
        Args:
            request_ids: List of request IDs to retrieve
            timeout: Maximum time to wait for each request
            interval: Polling interval
            max_concurrent: Maximum concurrent retrievals
            
        Returns:
            List of completion results
        """
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def retrieve_single(request_id: str) -> Dict:
            async with semaphore:
                try:
                    result = await self.poll_for_result(request_id, timeout, interval)
                    return {
                        "request_id": request_id,
                        "status": "success",
                        "result": result
                    }
                except Exception as e:
                    logger.error(f"Failed to retrieve {request_id}: {str(e)}")
                    return {
                        "request_id": request_id,
                        "status": "failed",
                        "error": str(e)
                    }
        
        logger.info(f"📥 Retrieving {len(request_ids)} deferred results...")
        
        tasks = [retrieve_single(rid) for rid in request_ids]
        results = await asyncio.gather(*tasks)
        
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        
        logger.info(f"✅ Retrieved {successful} results successfully")
        if failed > 0:
            logger.warning(f"❌ {failed} retrievals failed")
        
        return results
    
    async def process_batch_with_deferred(self,
                                         requests: List[Dict],
                                         model: str = "grok-4",
                                         wait_for_results: bool = True,
                                         timeout: timedelta = timedelta(minutes=30)) -> Dict:
        """
        Complete batch processing using deferred completions
        
        This is the full workflow: submit all, then retrieve all.
        
        Args:
            requests: List of request dictionaries
            model: Default model
            wait_for_results: Whether to wait and retrieve results
            timeout: Timeout for retrieving results
            
        Returns:
            Complete batch results with statistics
        """
        
        start_time = datetime.now()
        
        # Submit all requests
        submission_results = await self.submit_batch_deferred(requests, model)
        
        # Extract successful request IDs
        request_ids = [r["request_id"] for r in submission_results if "request_id" in r]
        
        batch_data = {
            "batch_id": f"xai_batch_{int(start_time.timestamp())}",
            "submitted_at": start_time.isoformat(),
            "total_requests": len(requests),
            "submitted_successfully": len(request_ids),
            "submission_results": submission_results
        }
        
        if wait_for_results and request_ids:
            # Retrieve all results
            logger.info(f"⏳ Waiting for {len(request_ids)} results...")
            
            retrieval_results = await self.retrieve_batch_results(
                request_ids,
                timeout=timeout,
                interval=timedelta(seconds=10)
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            batch_data.update({
                "completed_at": end_time.isoformat(),
                "duration_seconds": duration,
                "retrieval_results": retrieval_results,
                "successful_retrievals": sum(1 for r in retrieval_results if r["status"] == "success")
            })
            
            logger.info(f"✨ Batch complete in {duration:.1f} seconds")
        
        return batch_data
    
    def get_pending_status(self) -> Dict[str, Any]:
        """
        Get status of all pending requests
        
        Returns:
            Summary of pending, completed, and expired requests
        """
        
        pending = sum(1 for r in self.pending_requests.values() if r["status"] == "pending")
        completed = sum(1 for r in self.pending_requests.values() if r["status"] == "completed")
        expired = sum(1 for r in self.pending_requests.values() if r["status"] == "expired")
        failed = sum(1 for r in self.pending_requests.values() if r["status"] == "failed")
        
        return {
            "total_tracked": len(self.pending_requests),
            "pending": pending,
            "completed": completed,
            "expired": expired,
            "failed": failed,
            "requests": list(self.pending_requests.values())
        }