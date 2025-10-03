#!/usr/bin/env python3
"""
AI Council Synthesizer - Combines multiple AI responses into a superior synthesis
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class Synthesizer:
    """
    The Master Architect of the AI Council.
    Takes multiple AI responses and synthesizes them into a single, superior output.
    """
    
    def __init__(self, provider_factory, synthesizer_model: str = 'ant-1'):
        """
        Initialize the Synthesizer.
        
        Args:
            provider_factory: Factory to get providers for API calls
            synthesizer_model: Model to use for synthesis (should be high-reasoning model)
        """
        self.provider_factory = provider_factory
        self.synthesizer_model = synthesizer_model
        logger.info(f"🧙 Synthesizer initialized with model: {synthesizer_model}")
    
    async def synthesize(
        self,
        source_file_path: Path,
        council_responses: Dict[str, str],
        original_code: str,
        template_used: str
    ) -> str:
        """
        Synthesize multiple AI responses into a single, superior output.
        
        Args:
            source_file_path: Path to the original source file
            council_responses: Dictionary mapping model names to their responses
            original_code: The original source code
            template_used: The template that was used for processing
            
        Returns:
            The synthesized response combining the best of all inputs
        """
        logger.info(f"Synthesizing {len(council_responses)} responses for {source_file_path.name}")
        
        # Build the synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(
            source_file_path,
            council_responses,
            original_code,
            template_used
        )
        
        try:
            # Get the synthesizer provider
            provider = self.provider_factory.get_provider(self.synthesizer_model)
            
            # Make the synthesis call
            synthesized_response = await provider.complete(synthesis_prompt, self.synthesizer_model)
            
            # Add synthesis metadata
            synthesis_metadata = self._create_synthesis_metadata(council_responses)
            
            # Combine response with metadata
            final_output = self._format_final_output(
                synthesized_response,
                synthesis_metadata,
                council_responses
            )
            
            logger.info(f"✅ Successfully synthesized response for {source_file_path.name}")
            return final_output
            
        except Exception as e:
            logger.error(f"Synthesis failed for {source_file_path.name}: {e}")
            # Fallback: return the best individual response
            return self._select_best_fallback(council_responses)
    
    async def synthesize_council_responses(
        self,
        original_prompt: str,
        council_responses: Dict[str, str],
        template_used: str
    ) -> str:
        """
        Synthesize AI Council responses for general prompts (not code-specific)
        
        Args:
            original_prompt: The original user prompt
            council_responses: Dictionary mapping model names to their responses
            template_used: The template that was used for processing
            
        Returns:
            The synthesized response combining the best of all council inputs
        """
        logger.info(f"Synthesizing {len(council_responses)} council responses for template: {template_used}")
        
        # Build synthesis prompt
        synthesis_prompt = self._build_council_synthesis_prompt(
            original_prompt, council_responses, template_used
        )
        
        # Get synthesis provider
        provider = self.provider_factory.get_provider(self.synthesizer_model)
        
        # Generate synthesis
        synthesis = await provider.complete(synthesis_prompt, self.synthesizer_model)
        
        logger.info(f"Council synthesis completed using {self.synthesizer_model}")
        return synthesis
    
    def _build_council_synthesis_prompt(
        self,
        original_prompt: str,
        council_responses: Dict[str, str],
        template_used: str
    ) -> str:
        """Build synthesis prompt for AI Council responses"""
        
        # Prepare council responses section
        responses_section = ""
        for model, response in council_responses.items():
            responses_section += f"\n## {model.upper()} RESPONSE:\n{response}\n"
        
        synthesis_prompt = f"""You are the Master Synthesizer of the AI Council. Your role is to create a superior synthesis that combines the best insights from multiple AI responses.

**ORIGINAL REQUEST:**
Template: {template_used}
Query: {original_prompt}

**AI COUNCIL RESPONSES:**{responses_section}

**SYNTHESIS INSTRUCTIONS:**
1. **Analyze all responses** for their unique strengths and insights
2. **Identify the best elements** from each council member's contribution
3. **Resolve any contradictions** using your reasoning
4. **Synthesize a superior response** that:
   - Combines the strongest insights from all responses
   - Maintains coherent structure and flow
   - Eliminates redundancy while preserving valuable details
   - Provides the most comprehensive and accurate answer
   - Improves upon individual responses where possible

**SYNTHESIS REQUIREMENTS:**
- Be more comprehensive than any individual response
- Maintain the tone and approach appropriate for the template used
- Credit insights where multiple models agreed (shows consensus)
- Resolve disagreements with clear reasoning
- Structure the response clearly and professionally

**DELIVER THE MASTER SYNTHESIS:**"""
        
        return synthesis_prompt
    
    def _build_synthesis_prompt(
        self,
        source_file_path: Path,
        council_responses: Dict[str, str],
        original_code: str,
        template_used: str
    ) -> str:
        """Build the prompt for the synthesis model"""
        
        # Format the council responses
        formatted_responses = "\n\n".join([
            f"### Response from {model}:\n{response}"
            for model, response in council_responses.items()
        ])
        
        prompt = f"""You are the Master Architect of the AI Council, tasked with synthesizing multiple AI responses into a single, superior output.

## Original File: {source_file_path.name}

## Processing Template Used: {template_used}

## Original Code:

{original_code[:1000]}... # Truncated for context


## Council Member Responses:
{formatted_responses}

## Your Task:
Analyze all the responses from the AI Council members and create a synthesized output that:

1. **Combines the Best Insights**: Identify the strongest points from each response and integrate them cohesively
2. **Resolves Contradictions**: Where responses disagree, use your judgment to determine the best approach
3. **Enhances Completeness**: Fill any gaps that individual responses might have missed
4. **Maintains Consistency**: Ensure the final output has a consistent style and structure
5. **Adds Meta-Insights**: Provide any additional insights that emerge from seeing all perspectives

## Synthesis Guidelines:
- Preserve the format expected by the template ({template_used})
- Prioritize correctness and best practices
- When in doubt, favor the approach that appears in multiple responses
- Add brief notes about significant synthesis decisions in comments

## Output Format:
Provide the synthesized result that represents the collective wisdom of the AI Council, enhanced by your architectural oversight.

Begin your synthesis:
"""
        
        return prompt
    
    def _create_synthesis_metadata(self, council_responses: Dict[str, str]) -> Dict[str, Any]:
        """Create metadata about the synthesis process"""
        return {
            'synthesis_timestamp': datetime.now().isoformat(),
            'council_members': list(council_responses.keys()),
            'council_size': len(council_responses),
            'response_lengths': {
                model: len(response) 
                for model, response in council_responses.items()
            },
            'synthesizer_model': self.synthesizer_model
        }
    
    def _format_final_output(
        self,
        synthesized_response: str,
        synthesis_metadata: Dict[str, Any],
        council_responses: Dict[str, str]
    ) -> str:
        """Format the final output with metadata"""
        
        # Create a header with synthesis information
        header = f"""# AI Council Synthesis Report
# Generated: {synthesis_metadata['synthesis_timestamp']}
# Council Members: {', '.join(synthesis_metadata['council_members'])}
# Synthesizer: {synthesis_metadata['synthesizer_model']}
# ---

"""
        
        # Add the synthesized content
        output = header + synthesized_response
        
        # Optionally add a footer with response statistics
        footer = f"""

# ---
# Synthesis Metadata:
# - Council Size: {synthesis_metadata['council_size']} models
# - Response Lengths: {json.dumps(synthesis_metadata['response_lengths'], indent=2)}
# - Synthesis Model: {synthesis_metadata['synthesizer_model']}
"""
        
        return output + footer
    
    def _select_best_fallback(self, council_responses: Dict[str, str]) -> str:
        """Select the best response as a fallback if synthesis fails"""
        if not council_responses:
            return "# Error: No council responses available for synthesis"
        
        # Simple heuristic: choose the longest response (most comprehensive)
        best_model = max(council_responses.items(), key=lambda x: len(x[1]))
        
        return f"""# Fallback: Best Individual Response
# Selected Model: {best_model[0]}
# Note: Synthesis failed, showing best individual response
# ---

{best_model[1]}
"""
    
    async def batch_synthesize(
        self,
        synthesis_requests: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Synthesize multiple files in batch.
        
        Args:
            synthesis_requests: List of synthesis request dictionaries
            
        Returns:
            List of synthesized responses
        """
        results = []
        
        for request in synthesis_requests:
            try:
                result = await self.synthesize(
                    source_file_path=request['source_file_path'],
                    council_responses=request['council_responses'],
                    original_code=request['original_code'],
                    template_used=request['template_used']
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch synthesis failed for {request['source_file_path']}: {e}")
                results.append(self._select_best_fallback(request.get('council_responses', {})))
        
        return results