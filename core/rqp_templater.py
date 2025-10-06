#!/usr/bin/env python3
"""
RQP Templater - Resonant Query Protocol V2 Template Generator
Generates production-grade knowledge acquisition templates for AI providers.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
import re


@dataclass
class QuestionSpec:
    """Specification for an RQP question"""
    question: str
    id: Optional[str] = None
    priority: str = "high"
    depends_on: List[str] = field(default_factory=list)
    hardware_specifics: Optional[Dict[str, Any]] = None
    validation_requirements: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from question
            self.id = f"q_{re.sub(r'[^a-zA-Z0-9]', '_', self.question[:30]).lower()}"


@dataclass
class RQPContext:
    """Context specification for RQP template"""
    deployment_target: str = "Cloud Run x86_64, 4-vCPU"
    performance_goal: str = "30+ operations/sec"
    memory_constraints: str = "4-8GB RAM"
    integration_requirements: str = "Production monitoring, error handling"
    benchmark_baseline: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.benchmark_baseline:
            self.benchmark_baseline = {
                "hardware": "x86_64 4-core @ 2.6GHz",
                "throughput": "baseline",
                "latency": "baseline",
                "source": "Internal benchmarks",
                "date": datetime.now().strftime("%Y-%m-%d")
            }


class RQPTemplater:
    """Generate RQP V2 templates from simple inputs"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self.presets = self._load_presets()
        self.schema_version = "2.0"
        
    def _load_presets(self) -> Dict[str, Any]:
        """Load RQP presets from configuration"""
        presets_file = self.config_dir / "rqp_presets.yaml"
        
        # Default presets if file doesn't exist
        default_presets = {
            "optimization": {
                "personas": {
                    "requester": "Performance Optimization Architect",
                    "target": "Systems Performance Engineer"
                },
                "intent": "Seeking production-ready optimization patterns with measurable improvements",
                "response_modality": {
                    "format": "batch_response_json",
                    "content_density": "implementation_focused",
                    "metadata_level": "comprehensive",
                    "tone": "technical_pragmatic",
                    "brevity_boost": False
                },
                "processing_directives": {
                    "response_structure": "array_of_answer_objects",
                    "fill_strategy": "high_confidence_patterns_only",
                    "source_requirement": "minimum_2_authoritative",
                    "implementation_threshold": "80_percent_complete",
                    "avoid_speculation": True,
                    "concurrency_handling": "dependency_aware",
                    "strict_schema": True
                },
                "security_clearance": "PRODUCTION_GRADE",
                "defaults": {
                    "hardware": "x86_64 Ice Lake 4-core @ 2.6GHz",
                    "require_benchmarks": True,
                    "require_failure_modes": True,
                    "require_cross_checks": True
                }
            },
            "ml_serving": {
                "personas": {
                    "requester": "ML Systems Architect",
                    "target": "ML Infrastructure Engineer"
                },
                "intent": "Implementing production ML serving with high throughput",
                "defaults": {
                    "hardware": "NVIDIA T4 GPU",
                    "performance_target": "1000 inferences/sec",
                    "require_gpu_optimization": True
                }
            },
            "database": {
                "personas": {
                    "requester": "Database Optimization Expert",
                    "target": "Database Performance Engineer"
                },
                "intent": "Optimizing database queries and architecture",
                "defaults": {
                    "require_query_plans": True,
                    "require_index_analysis": True,
                    "benchmark_queries": True
                }
            },
            "distributed": {
                "personas": {
                    "requester": "Distributed Systems Architect",
                    "target": "Distributed Systems Engineer"
                },
                "intent": "Building scalable distributed systems",
                "defaults": {
                    "require_consensus_analysis": True,
                    "require_failure_scenarios": True,
                    "network_assumptions": "Partial failures possible"
                }
            }
        }
        
        if presets_file.exists():
            try:
                with open(presets_file, 'r') as f:
                    loaded_presets = yaml.safe_load(f)
                    return loaded_presets.get('presets', default_presets)
            except Exception as e:
                print(f"Warning: Could not load presets: {e}")
                
        return default_presets
    
    def create_batch(self, 
                    questions: List[Union[str, QuestionSpec]], 
                    preset: str = "optimization",
                    context: Optional[RQPContext] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Create an RQP V2 batch template from questions
        
        Args:
            questions: List of questions (strings or QuestionSpec objects)
            preset: Preset name to use for template configuration
            context: Optional context override
            **kwargs: Additional context parameters
            
        Returns:
            Complete RQP V2 template as dictionary
        """
        # Load preset
        preset_config = self.presets.get(preset, self.presets["optimization"])
        
        # Create context
        if not context:
            context = RQPContext(**kwargs)
            
        # Apply preset defaults to context
        if "defaults" in preset_config:
            defaults = preset_config["defaults"]
            if "hardware" in defaults and not kwargs.get("hardware"):
                context.benchmark_baseline["hardware"] = defaults["hardware"]
            if "performance_target" in defaults and not kwargs.get("performance_target"):
                context.performance_goal = defaults["performance_target"]
        
        # Process questions
        question_objects = []
        for i, q in enumerate(questions):
            if isinstance(q, str):
                q_spec = QuestionSpec(question=q)
            else:
                q_spec = q
                
            # Apply preset-specific enhancements
            self._enhance_question(q_spec, preset_config, context)
            
            question_objects.append(self._question_to_dict(q_spec, i))
        
        # Build template
        template = {
            "rqp_version": self.schema_version,
            "handshake": self._create_handshake(preset_config),
            "payload": {
                "batch_id": f"{preset}_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "query_type": f"{preset}_knowledge_acquisition",
                "context": self._context_to_dict(context, preset_config),
                "expected_response_format": self._create_response_format(len(questions)),
                "questions": question_objects
            }
        }
        
        return template
    
    def _create_handshake(self, preset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create handshake block from preset configuration"""
        personas = preset_config.get("personas", {})
        
        handshake = {
            "requester_persona": personas.get("requester", "Technical Architect"),
            "requester_intent": preset_config.get("intent", "Seeking production-ready implementations"),
            "target_persona": personas.get("target", "Domain Expert"),
            "response_modality": preset_config.get("response_modality", {
                "format": "batch_response_json",
                "content_density": "implementation_focused",
                "metadata_level": "comprehensive",
                "tone": "technical_pragmatic",
                "brevity_boost": False
            }),
            "processing_directives": preset_config.get("processing_directives", {
                "response_structure": "array_of_answer_objects",
                "fill_strategy": "high_confidence_patterns_only",
                "source_requirement": "minimum_2_authoritative",
                "implementation_threshold": "80_percent_complete",
                "avoid_speculation": True,
                "concurrency_handling": "dependency_aware",
                "strict_schema": True
            }),
            "security_clearance": preset_config.get("security_clearance", "PRODUCTION_GRADE")
        }
        
        return handshake
    
    def _context_to_dict(self, context: RQPContext, preset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert context object to dictionary with preset enhancements"""
        ctx_dict = {
            "deployment_target": context.deployment_target,
            "performance_goal": context.performance_goal,
            "memory_constraints": context.memory_constraints,
            "integration_requirements": context.integration_requirements,
            "benchmark_baseline": context.benchmark_baseline
        }
        
        # Add preset-specific context elements
        defaults = preset_config.get("defaults", {})
        
        if defaults.get("require_gpu_optimization"):
            ctx_dict["gpu_requirements"] = {
                "cuda_compute_capability": "7.5+",
                "memory_bandwidth": "320GB/s",
                "tensor_cores": True
            }
            
        if defaults.get("network_assumptions"):
            ctx_dict["network_constraints"] = {
                "assumptions": defaults["network_assumptions"],
                "latency_budget": "10ms p99",
                "bandwidth": "10Gbps"
            }
            
        return ctx_dict
    
    def _enhance_question(self, q_spec: QuestionSpec, preset_config: Dict[str, Any], context: RQPContext):
        """Enhance question with preset-specific requirements"""
        defaults = preset_config.get("defaults", {})
        
        # Add hardware specifics if not present
        if not q_spec.hardware_specifics and "hardware" in context.benchmark_baseline:
            q_spec.hardware_specifics = {
                "target_hardware": context.benchmark_baseline["hardware"],
                "optimization_target": context.performance_goal
            }
            
        # Add validation requirements based on preset
        if not q_spec.validation_requirements:
            q_spec.validation_requirements = {}
            
        if defaults.get("require_benchmarks"):
            q_spec.validation_requirements["benchmark"] = {
                "performance_target": context.performance_goal,
                "measurement_accuracy": "±5%",
                "test_environment": "Production-like"
            }
            
        if defaults.get("require_failure_modes"):
            q_spec.validation_requirements["failure_analysis"] = {
                "identify_failure_modes": True,
                "mitigation_strategies": True,
                "performance_impact": "Quantified"
            }
            
        if defaults.get("require_query_plans"):
            q_spec.validation_requirements["database"] = {
                "explain_plans": True,
                "index_recommendations": True,
                "query_optimization": True
            }
    
    def _question_to_dict(self, q_spec: QuestionSpec, index: int) -> Dict[str, Any]:
        """Convert question specification to dictionary"""
        q_dict = {
            "id": q_spec.id or f"q{index + 1}",
            "question": q_spec.question,
            "priority": q_spec.priority,
            "depends_on": q_spec.depends_on,
            "expected_deliverable": f"Production-ready implementation for: {q_spec.question}"
        }
        
        if q_spec.hardware_specifics:
            q_dict["hardware_specifics"] = q_spec.hardware_specifics
            
        if q_spec.validation_requirements:
            q_dict["validation_requirements"] = q_spec.validation_requirements
            
        return q_dict
    
    def _create_response_format(self, num_questions: int) -> Dict[str, Any]:
        """Create expected response format specification"""
        return {
            "structure": f"Array of {num_questions} answer objects matching question count",
            "schema_per_answer": {
                "answer_id": "string (matches question.id)",
                "question": "string (the full question)",
                "implementation": {
                    "code_snippets": "Production-ready code with comments",
                    "validation_patterns": "Testing and verification approach",
                    "performance_characteristics": "Measured on target hardware",
                    "failure_modes": [
                        {
                            "scenario": "string describing failure condition",
                            "mitigation": "string describing recovery strategy",
                            "performance_impact": "string (e.g., '10% throughput loss')"
                        }
                    ]
                },
                "tradeoff_analysis": {
                    "performance_vs_complexity": "string",
                    "maintenance_impact": "low|medium|high",
                    "scalability": "string",
                    "cost_impact": "string"
                },
                "quality_metrics": {
                    "confidence": "number 0.0-1.0",
                    "source_quality": "A|B|C|D|E|F",
                    "validation_status": "theoretical|simulated|lab_tested|production_proven"
                },
                "sources": [
                    {
                        "url": "string",
                        "type": "implementation|documentation|benchmark|paper|production_system",
                        "quality": "A|B|C|D|E|F",
                        "relevance": "direct|adapted|inspired"
                    }
                ],
                "metadata": {
                    "complexity": "1-5 scale",
                    "platform_compatibility": ["x86_64", "ARM64", "Cloud Run"],
                    "testing_difficulty": "Trivial|Easy|Moderate|Hard|Extreme",
                    "assumptions": ["string array"],
                    "alternatives": {}
                },
                "cross_checks": [
                    {
                        "system": "Benchmark|Production|Testing",
                        "result": "pass|fail|warnings",
                        "required": True,
                        "details": "string"
                    }
                ]
            }
        }
    
    def save_template(self, template: Dict[str, Any], output_path: Union[str, Path]) -> Path:
        """Save template to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
            
        return output_path
    
    def validate_template(self, template: Union[Dict[str, Any], Path]) -> Dict[str, Any]:
        """Validate an RQP template against schema"""
        if isinstance(template, Path):
            with open(template, 'r') as f:
                template = json.load(f)
                
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required top-level fields
        required_fields = ["handshake", "payload"]
        for field in required_fields:
            if field not in template:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
                
        # Check handshake structure
        if "handshake" in template:
            handshake = template["handshake"]
            required_handshake = ["requester_persona", "target_persona", "response_modality"]
            for field in required_handshake:
                if field not in handshake:
                    validation_result["warnings"].append(f"Missing recommended handshake field: {field}")
                    
        # Check payload structure
        if "payload" in template:
            payload = template["payload"]
            required_payload = ["batch_id", "questions", "context"]
            for field in required_payload:
                if field not in payload:
                    validation_result["errors"].append(f"Missing required payload field: {field}")
                    validation_result["valid"] = False
                    
            # Validate questions
            if "questions" in payload:
                if not isinstance(payload["questions"], list):
                    validation_result["errors"].append("Questions must be a list")
                    validation_result["valid"] = False
                elif len(payload["questions"]) == 0:
                    validation_result["errors"].append("At least one question required")
                    validation_result["valid"] = False
                else:
                    for i, question in enumerate(payload["questions"]):
                        if "question" not in question:
                            validation_result["errors"].append(f"Question {i} missing 'question' field")
                            validation_result["valid"] = False
                            
        return validation_result


def create_example_usage():
    """Example usage of RQPTemplater"""
    templater = RQPTemplater()
    
    # Example 1: Simple optimization questions
    questions = [
        "How to implement zero-copy memory patterns in Rust?",
        "What's the best way to optimize SIMD loops for AVX2?",
        "How to reduce cache misses in image processing?"
    ]
    
    template = templater.create_batch(
        questions=questions,
        preset="optimization",
        hardware="Intel Ice Lake 8-core @ 3.0GHz",
        performance_target="50 images/sec"
    )
    
    # Save template
    output_path = Path("examples/optimization_batch.json")
    templater.save_template(template, output_path)
    print(f"Created template: {output_path}")
    
    # Example 2: ML serving with dependencies
    ml_questions = [
        QuestionSpec(
            question="How to implement dynamic batching for ML inference?",
            id="dynamic_batching",
            hardware_specifics={"gpu": "NVIDIA T4", "cuda": "11.8"}
        ),
        QuestionSpec(
            question="What's the pattern for GPU memory pooling?",
            id="gpu_memory",
            depends_on=["dynamic_batching"]
        )
    ]
    
    ml_template = templater.create_batch(
        questions=ml_questions,
        preset="ml_serving",
        deployment_target="Kubernetes with GPU nodes"
    )
    
    print("\nML Template created with dependencies")
    
    return templater


if __name__ == "__main__":
    # Run example
    create_example_usage()