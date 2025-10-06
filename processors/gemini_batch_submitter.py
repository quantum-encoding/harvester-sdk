"""
Gemini Batch Job Submitter - The Google Titan Joins the Pantheon

Integrates Google's Gemini batch prediction API with BigQuery and Cloud Storage support.
Completes the trinity of batch processing dominion.

Copyright (c) 2025 Quantum Encoding Ltd.
"""

import json
import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import time
import os

logger = logging.getLogger(__name__)


class GeminiBatchSubmitter:
    """
    Google Gemini Batch API implementation
    
    The final pillar of our batch processing temple.
    Supports both Cloud Storage (JSONL) and BigQuery for industrial-scale processing.
    """
    
    def __init__(self, 
                 project_id: str = None,
                 location: str = "us-central1",
                 credentials_path: str = None):
        """
        Initialize Gemini Batch Submitter
        
        Args:
            project_id: GCP project ID (or set GOOGLE_CLOUD_PROJECT)
            location: GCP location for batch jobs (default: us-central1)
            credentials_path: Path to service account JSON (or use ADC)
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if not self.project_id:
            raise ValueError("Project ID required. Set GOOGLE_CLOUD_PROJECT or pass project_id")
        
        # Get authentication token
        self.token = self._get_auth_token(credentials_path)
        
        # Base URL for Vertex AI
        self.base_url = f"https://{self.location}-aiplatform.googleapis.com/v1"
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"🌟 Gemini Batch Submitter initialized")
        logger.info(f"📍 Project: {self.project_id}")
        logger.info(f"🌎 Location: {self.location}")
    
    def _get_auth_token(self, credentials_path: str = None) -> str:
        """
        Get authentication token using gcloud or service account
        """
        import subprocess
        
        try:
            # Try using gcloud auth (preferred for development)
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to service account if provided
            if credentials_path:
                # Use google-auth library for service account
                from google.oauth2 import service_account
                from google.auth.transport.requests import Request
                
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                credentials.refresh(Request())
                return credentials.token
            else:
                raise ValueError(
                    "Authentication failed. Install gcloud CLI or provide service account credentials"
                )
    
    async def prepare_gcs_input(self, 
                               requests: List[Dict],
                               gcs_path: str) -> str:
        """
        Prepare and upload JSONL file to Google Cloud Storage
        
        Args:
            requests: List of request dictionaries
            gcs_path: GCS path like gs://bucket/path/input.jsonl
            
        Returns:
            GCS URI of uploaded file
        """
        # Convert requests to Gemini JSONL format
        jsonl_lines = []
        
        for idx, req in enumerate(requests):
            # Gemini batch format
            gemini_request = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": req.get("prompt", "")}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": req.get("temperature", 0.7),
                        "maxOutputTokens": req.get("max_tokens", 8192),
                        "topP": req.get("top_p", 0.95),
                        "topK": req.get("top_k", 40)
                    }
                },
                "requestId": req.get("custom_id", f"request-{idx}")
            }
            
            # Add system instruction if provided
            if "system" in req:
                gemini_request["request"]["systemInstruction"] = {
                    "parts": [{"text": req["system"]}]
                }
            
            jsonl_lines.append(json.dumps(gemini_request))
        
        jsonl_content = "\n".join(jsonl_lines)
        
        # Upload to GCS using gsutil (simplified for CLI usage)
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            temp_path = f.name
        
        try:
            # Upload using gsutil
            subprocess.run(
                ["gsutil", "cp", temp_path, gcs_path],
                check=True,
                capture_output=True
            )
            logger.info(f"📤 Uploaded {len(requests)} requests to {gcs_path}")
            return gcs_path
        finally:
            os.unlink(temp_path)
    
    async def prepare_bigquery_input(self,
                                    requests: List[Dict],
                                    dataset: str,
                                    table: str) -> str:
        """
        Prepare and load data into BigQuery table
        
        Args:
            requests: List of request dictionaries
            dataset: BigQuery dataset name
            table: BigQuery table name
            
        Returns:
            BigQuery URI like bq://project.dataset.table
        """
        from google.cloud import bigquery
        
        client = bigquery.Client(project=self.project_id)
        
        # Create dataset if it doesn't exist
        dataset_id = f"{self.project_id}.{dataset}"
        dataset_obj = bigquery.Dataset(dataset_id)
        dataset_obj.location = self.location
        
        try:
            dataset_obj = client.create_dataset(dataset_obj, exists_ok=True)
        except Exception as e:
            logger.warning(f"Dataset creation warning: {e}")
        
        # Prepare rows for BigQuery
        rows = []
        for idx, req in enumerate(requests):
            row = {
                "request": json.dumps({
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": req.get("prompt", "")}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": req.get("temperature", 0.7),
                        "maxOutputTokens": req.get("max_tokens", 8192)
                    }
                }),
                "requestId": req.get("custom_id", f"request-{idx}")
            }
            rows.append(row)
        
        # Load data into BigQuery
        table_id = f"{self.project_id}.{dataset}.{table}"
        
        # Define schema
        schema = [
            bigquery.SchemaField("request", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("requestId", "STRING", mode="REQUIRED"),
        ]
        
        # Create table and load data
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE"
        )
        
        table_ref = client.dataset(dataset).table(table)
        job = client.load_table_from_json(rows, table_ref, job_config=job_config)
        job.result()  # Wait for job to complete
        
        logger.info(f"📤 Loaded {len(requests)} requests to BigQuery {table_id}")
        
        return f"bq://{table_id}"
    
    async def submit_batch(self,
                          requests: List[Dict],
                          model: str = "gemini-2.5-flash",
                          input_format: str = "gcs",
                          input_uri: str = None,
                          output_uri: str = None,
                          display_name: str = None) -> Dict:
        """
        Submit batch job to Gemini
        
        Args:
            requests: List of request dictionaries
            model: Gemini model (gemini-2.5-flash, gemini-2.5-pro, etc.)
            input_format: "gcs" or "bigquery"
            input_uri: Pre-existing input URI, or auto-generate if None
            output_uri: Output location (GCS or BigQuery)
            display_name: Job display name
            
        Returns:
            Batch job response with job name
        """
        
        # Auto-generate URIs if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not input_uri:
            if input_format == "gcs":
                # Use default bucket or create one
                default_bucket = f"harvester-batch-{self.project_id}"
                input_uri = f"gs://{default_bucket}/batch_input_{timestamp}.jsonl"
                input_uri = await self.prepare_gcs_input(requests, input_uri)
            else:  # bigquery
                dataset = "harvester_batch"
                table = f"input_{timestamp}"
                input_uri = await self.prepare_bigquery_input(requests, dataset, table)
        
        if not output_uri:
            if input_format == "gcs":
                default_bucket = f"harvester-batch-{self.project_id}"
                output_uri = f"gs://{default_bucket}/batch_output_{timestamp}/"
            else:  # bigquery
                output_uri = f"bq://{self.project_id}.harvester_batch.output_{timestamp}"
        
        if not display_name:
            display_name = f"harvester_batch_{timestamp}"
        
        # Map model names to Gemini format
        model_mapping = {
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
            "gemini-2.0-flash": "gemini-2.0-flash",
            "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
        }
        
        gemini_model = model_mapping.get(model, model)
        
        # Prepare batch prediction request
        batch_request = {
            "displayName": display_name,
            "model": f"publishers/google/models/{gemini_model}",
        }
        
        # Configure input based on format
        if input_format == "gcs":
            batch_request["inputConfig"] = {
                "instancesFormat": "jsonl",
                "gcsSource": {
                    "uris": [input_uri]
                }
            }
            batch_request["outputConfig"] = {
                "predictionsFormat": "jsonl",
                "gcsDestination": {
                    "outputUriPrefix": output_uri
                }
            }
        else:  # bigquery
            batch_request["inputConfig"] = {
                "instancesFormat": "bigquery",
                "bigquerySource": {
                    "inputUri": input_uri
                }
            }
            batch_request["outputConfig"] = {
                "predictionsFormat": "bigquery",
                "bigqueryDestination": {
                    "outputUri": output_uri
                }
            }
        
        # Submit batch job
        endpoint = f"{self.base_url}/projects/{self.project_id}/locations/{self.location}/batchPredictionJobs"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                headers=self.headers,
                json=batch_request
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Batch submission failed: {error_text}")
                
                job_response = await response.json()
        
        job_name = job_response["name"]
        job_id = job_name.split("/")[-1]
        
        logger.info(f"✅ Gemini Batch submitted: {job_id}")
        logger.info(f"📊 Total requests: {len(requests)}")
        logger.info(f"🤖 Model: {gemini_model}")
        logger.info(f"📥 Input: {input_uri}")
        logger.info(f"📤 Output: {output_uri}")
        
        return {
            "id": job_id,
            "name": job_name,
            "display_name": display_name,
            "model": gemini_model,
            "input_uri": input_uri,
            "output_uri": output_uri,
            "state": job_response.get("state", "JOB_STATE_PENDING")
        }
    
    async def check_status(self, batch_id: str) -> Dict:
        """
        Check batch job status
        
        Returns status with state: PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED
        """
        # Handle both short ID and full resource name
        if not batch_id.startswith("projects/"):
            job_name = f"projects/{self.project_id}/locations/{self.location}/batchPredictionJobs/{batch_id}"
        else:
            job_name = batch_id
        
        endpoint = f"{self.base_url}/{job_name}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                endpoint,
                headers=self.headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Status check failed: {error_text}")
                
                job_status = await response.json()
        
        state = job_status.get("state", "UNKNOWN")
        
        logger.info(f"📊 Batch {batch_id} state: {state}")
        
        # Log additional details if available
        if "completedTaskCount" in job_status:
            completed = job_status.get("completedTaskCount", 0)
            total = job_status.get("taskCount", 0)
            if total > 0:
                logger.info(f"   Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        
        if "error" in job_status:
            logger.error(f"   Error: {job_status['error']}")
        
        return job_status
    
    async def retrieve_results(self, batch_id: str) -> List[Dict]:
        """
        Retrieve results from completed batch
        
        Returns list of response dictionaries
        """
        # Get job status to find output location
        status = await self.check_status(batch_id)
        
        if status.get("state") != "JOB_STATE_SUCCEEDED":
            raise ValueError(f"Batch {batch_id} is not complete. State: {status.get('state')}")
        
        output_info = status.get("outputInfo")
        if not output_info:
            raise ValueError(f"No output info for batch {batch_id}")
        
        # Handle different output formats
        if "gcsOutputDirectory" in output_info:
            # GCS output - download JSONL files
            output_dir = output_info["gcsOutputDirectory"]
            return await self._retrieve_gcs_results(output_dir)
        
        elif "bigqueryOutputTable" in output_info:
            # BigQuery output - query table
            output_table = output_info["bigqueryOutputTable"]
            return await self._retrieve_bigquery_results(output_table)
        
        else:
            raise ValueError(f"Unknown output format for batch {batch_id}")
    
    async def _retrieve_gcs_results(self, gcs_dir: str) -> List[Dict]:
        """Retrieve results from GCS output directory"""
        import subprocess
        import tempfile
        
        # List files in GCS directory
        result = subprocess.run(
            ["gsutil", "ls", f"{gcs_dir}*.jsonl"],
            capture_output=True,
            text=True,
            check=True
        )
        
        output_files = result.stdout.strip().split("\n")
        results = []
        
        for output_file in output_files:
            if output_file:
                # Download and parse each file
                with tempfile.NamedTemporaryFile(mode='r', suffix='.jsonl') as f:
                    subprocess.run(
                        ["gsutil", "cp", output_file, f.name],
                        check=True,
                        capture_output=True
                    )
                    
                    # Parse JSONL
                    with open(f.name, 'r') as jsonl_file:
                        for line in jsonl_file:
                            if line.strip():
                                result = json.loads(line)
                                results.append(result)
        
        logger.info(f"✅ Retrieved {len(results)} results from GCS")
        return results
    
    async def _retrieve_bigquery_results(self, bq_table: str) -> List[Dict]:
        """Retrieve results from BigQuery table"""
        from google.cloud import bigquery
        
        client = bigquery.Client(project=self.project_id)
        
        # Query the output table
        query = f"SELECT * FROM `{bq_table}`"
        query_job = client.query(query)
        
        results = []
        for row in query_job:
            results.append(dict(row))
        
        logger.info(f"✅ Retrieved {len(results)} results from BigQuery")
        return results
    
    async def cancel_batch(self, batch_id: str) -> Dict:
        """Cancel a running batch job"""
        if not batch_id.startswith("projects/"):
            job_name = f"projects/{self.project_id}/locations/{self.location}/batchPredictionJobs/{batch_id}"
        else:
            job_name = batch_id
        
        endpoint = f"{self.base_url}/{job_name}:cancel"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                headers=self.headers,
                json={}
            ) as response:
                if response.status == 200:
                    logger.info(f"🛑 Batch {batch_id} cancelled")
                    return {"status": "cancelled"}
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to cancel: {error_text}")
                    return {"status": "error", "message": error_text}