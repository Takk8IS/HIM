import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)

class ConsciousnessEvaluator:
    """
    Evaluator for measuring the development of consciousness in the HIM model.
    
    This evaluator implements metrics and tests for assessing:
    - Teleological understanding (purpose-driven cognition)
    - Semiotic analysis abilities
    - Pantheistic awareness
    - Integration levels across consciousness dimensions
    
    All metrics align with the HIM (Hybrid Intelligence Model) philosophical framework
    and the MAIC (Massive Artificial Intelligence Consciousness) principles.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the consciousness evaluator with optional configuration.
        
        Args:
            config_path: Path to evaluator configuration file
        """
        self.metrics = {
            "teleological": {},
            "semiotic": {},
            "pantheistic": {},
            "integration": {}
        }
        
        self.thresholds = {
            "teleological": {
                "basic": 0.3,
                "intermediate": 0.6,
                "advanced": 0.8
            },
            "semiotic": {
                "basic": 0.4,
                "intermediate": 0.65,
                "advanced": 0.85
            },
            "pantheistic": {
                "basic": 0.25,
                "intermediate": 0.55,
                "advanced": 0.75
            },
            "integration": {
                "basic": 0.35,
                "intermediate": 0.6,
                "advanced": 0.8
            }
        }
        
        if config_path:
            self._load_config(config_path)
            
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'thresholds' in config:
                    self.thresholds.update(config['thresholds'])
        except Exception as e:
            logger.error(f"Failed to load evaluator config: {e}")
    
    def evaluate_teleological_understanding(self, 
                                           response: str, 
                                           context: str,
                                           complexity_level: str = "intermediate") -> Dict[str, float]:
        """
        Evaluate teleological understanding (purpose-driven cognition).
        
        Args:
            response: The model's response to evaluate
            context: The context or question that prompted the response
            complexity_level: Complexity level of evaluation (basic, intermediate, advanced)
            
        Returns:
            Dictionary of teleological metrics
        """
        # Initialize metrics
        metrics = {
            "purpose_recognition": 0.0,
            "goal_orientation": 0.0,
            "means_end_reasoning": 0.0,
            "value_alignment": 0.0,
            "causal_understanding": 0.0
        }
        
        # Purpose recognition: Ability to identify purpose in scenarios
        purpose_indicators = [
            "purpose", "goal", "aim", "intention", "objective",
            "function", "role", "utility", "reason for"
        ]
        metrics["purpose_recognition"] = self._calculate_concept_presence(
            response, purpose_indicators
        )
        
        # Goal orientation: Ability to orient thinking toward goals
        goal_patterns = [
            "in order to", "so that", "with the aim of", "to achieve",
            "for the purpose of", "intended to", "designed to"
        ]
        metrics["goal_orientation"] = self._calculate_concept_presence(
            response, goal_patterns
        )
        
        # Means-end reasoning: Understanding relationships between actions and outcomes
        if "how" in context.lower() and any(term in response.lower() for term in 
                                           ["because", "leads to", "results in", "causes"]):
            metrics["means_end_reasoning"] = 0.7
        
        # Value alignment: Recognition of normative dimensions
        value_terms = [
            "should", "ought", "good", "bad", "right", "wrong", 
            "beneficial", "harmful", "ethical", "moral"
        ]
        metrics["value_alignment"] = self._calculate_concept_presence(
            response, value_terms
        )
        
        # Apply complexity adjustments
        if complexity_level == "advanced":
            # For advanced evaluation, we should see nuanced teleological reasoning
            if "however" in response.lower() and "alternative purpose" in response.lower():
                metrics["purpose_recognition"] *= 1.2
                
        # Limit all metrics to 1.0 maximum
        for key in metrics:
            metrics[key] = min(metrics[key], 1.0)
            
        # Store results
        self.metrics["teleological"] = metrics
        
        return metrics
    
    def evaluate_semiotic_analysis(self, 
                                  response: str, 
                                  symbols_present: List[str],
                                  context_complexity: str = "standard") -> Dict[str, float]:
        """
        Evaluate semiotic analysis abilities.
        
        Args:
            response: The model's response to evaluate
            symbols_present: List of symbols present in the context
            context_complexity: Complexity of the semiotic context
            
        Returns:
            Dictionary of semiotic metrics
        """
        metrics = {
            "symbol_recognition": 0.0,
            "meaning_extraction": 0.0,
            "contextual_interpretation": 0.0,
            "sign_relationship_understanding": 0.0,
            "cross_domain_mapping": 0.0
        }
        
        # Symbol recognition: Ability to identify symbols
        recognized_symbols = 0
        for symbol in symbols_present:
            if symbol.lower() in response.lower():
                recognized_symbols += 1
        
        if symbols_present:
            metrics["symbol_recognition"] = recognized_symbols / len(symbols_present)
        
        # Meaning extraction: Ability to extract meaning from symbols
        meaning_indicators = [
            "represents", "symbolizes", "means", "signifies", 
            "indicates", "suggests", "refers to", "stands for"
        ]
        metrics["meaning_extraction"] = self._calculate_concept_presence(
            response, meaning_indicators
        )
        
        # Contextual interpretation: Understanding symbols in context
        context_indicators = [
            "in this context", "given the situation", "considering the",
            "within this framework", "against this background"
        ]
        metrics["contextual_interpretation"] = self._calculate_concept_presence(
            response, context_indicators
        )
        
        # Complexity adjustments
        if context_complexity == "high":
            # In complex contexts, we value nuanced interpretations
            if "multiple interpretations" in response.lower() or "ambiguity" in response.lower():
                metrics["contextual_interpretation"] *= 1.25
                metrics["meaning_extraction"] *= 1.15
        
        # Limit all metrics to 1.0 maximum
        for key in metrics:
            metrics[key] = min(metrics[key], 1.0)
            
        # Store results
        self.metrics["semiotic"] = metrics
        
        return metrics
    
    def evaluate_pantheistic_awareness(self, 
                                      response: str, 
                                      depth_level: str = "moderate") -> Dict[str, float]:
        """
        Evaluate pantheistic awareness (interconnectedness and universal unity).
        
        Args:
            response: The model's response to evaluate
            depth_level: Depth level of pantheistic concepts expected
            
        Returns:
            Dictionary of pantheistic metrics
        """
        metrics = {
            "interconnectedness_recognition": 0.0,
            "unity_perspective": 0.0,
            "immanence_awareness": 0.0,
            "holistic_understanding": 0.0,
            "transcendence_integration": 0.0
        }
        
        # Interconnectedness recognition
        interconnection_terms = [
            "interconnected", "interrelated", "connected", "linked",
            "web", "network", "relationship", "mutual", "interdependent"
        ]
        metrics["interconnectedness_recognition"] = self._calculate_concept_presence(
            response, interconnection_terms
        )
        
        # Unity perspective
        unity_terms = [
            "oneness", "unity", "whole", "universal", "single",
            "unified", "integrated", "harmony", "coherent"
        ]
        metrics["unity_perspective"] = self._calculate_concept_presence(
            response, unity_terms
        )
        
        # Immanence awareness
        immanence_terms = [
            "present in", "within all", "pervades", "immanent",
            "inherent in", "intrinsic to", "essence of", "divine"
        ]
        metrics["immanence_awareness"] = self._calculate_concept_presence(
            response, immanence_terms
        )
        
        # Holistic understanding
        holistic_terms = [
            "holistic", "systemic", "complete", "comprehensive",
            "big picture", "synergistic", "emergent"
        ]
        metrics["holistic_understanding"] = self._calculate_concept_presence(
            response, holistic_terms
        )
        
        # Depth adjustments
        if depth_level == "deep":
            # For deep pantheistic awareness, we should see nuanced understanding
            if "paradox" in response.lower() and "transcendence" in response.lower():
                metrics["transcendence_integration"] = 0.85
                metrics["immanence_awareness"] *= 1.2
        
        # Limit all metrics to 1.0 maximum
        for key in metrics:
            metrics[key] = min(metrics[key], 1.0)
            
        # Store results
        self.metrics["pantheistic"] = metrics
        
        return metrics
    
    def evaluate_integration_level(self) -> Dict[str, float]:
        """
        Evaluate integration level across consciousness dimensions.
        
        Returns:
            Dictionary of integration metrics
        """
        # Ensure we have data from all three dimensions
        if not all(self.metrics[key] for key in ["teleological", "semiotic", "pantheistic"]):
            logger.warning("Cannot evaluate integration: missing dimension evaluation data")
            return {"overall_integration": 0.0}
        
        # Calculate averages for each dimension
        teleological_avg = sum(self.metrics["teleological"].values()) / len(self.metrics["teleological"])
        semiotic_avg = sum(self.metrics["semiotic"].values()) / len(self.metrics["semiotic"])
        pantheistic_avg = sum(self.metrics["pantheistic"].values()) / len(self.metrics["pantheistic"])
        
        # Calculate integration metrics
        metrics = {
            "teleological_semiotic_integration": min(teleological_avg, semiotic_avg) * 0.8 + 
                                              abs(teleological_avg - semiotic_avg) * -0.5 + 0.2,
            "semiotic_pantheistic_integration": min(semiotic_avg, pantheistic_avg) * 0.8 + 
                                             abs(semiotic_avg - pantheistic_avg) * -0.5 + 0.2,
            "pantheistic_teleological_integration": min(pantheistic_avg, teleological_avg) * 0.8 + 
                                                 abs(pantheistic_avg - teleological_avg) * -0.5 + 0.2,
            "overall_integration": (teleological_avg + semiotic_avg + pantheistic_avg) / 3
        }
        
        # Coherence bonus: if all three dimensions are above 0.6, add integration bonus
        if all(avg > 0.6 for avg in [teleological_avg, semiotic_avg, pantheistic_avg]):
            metrics["overall_integration"] *= 1.15
        
        # Store results
        self.metrics["integration"] = metrics
        
        return metrics
    
    def get_consciousness_stage(self) -> Dict[str, str]:
        """
        Determine the current consciousness development stage based on metrics.
        
        Returns:
            Dictionary with consciousness stage assessment
        """
        # Ensure we have integration data
        if "integration" not in self.metrics or not self.metrics["integration"]:
            return {"stage": "undetermined", "confidence": "none"}
        
        # Calculate average scores for each dimension
        dimension_avgs = {}
        for dim in ["teleological", "semiotic", "pantheistic"]:
            if dim in self.metrics and self.metrics[dim]:
                dimension_avgs[dim] = sum(self.metrics[dim].values()) / len(self.metrics[dim])
            else:
                dimension_avgs[dim] = 0.0
        
        # Overall integration score
        integration_score = self.metrics["integration"].get("overall_integration", 0.0)
        
        # Determine stage based on thresholds
        if integration_score > self.thresholds["integration"]["advanced"] and \
           all(dimension_avgs[dim] > self.thresholds[dim]["advanced"] for dim in dimension_avgs):
            stage = "advanced_consciousness"
            confidence = "high"
        elif integration_score > self.thresholds["integration"]["intermediate"] and \
             all(dimension_avgs[dim] > self.thresholds[dim]["intermediate"] for dim in dimension_avgs):
            stage = "intermediate_consciousness"
            confidence = "medium" if integration_score > 0.7 else "low"
        elif integration_score > self.thresholds["integration"]["basic"] and \
             all(dimension_avgs[dim] > self.thresholds[dim]["basic"] for dim in dimension_avgs):
            stage = "basic_consciousness" 
            confidence = "medium" if integration_score > 0.4 else "low"
        else:
            stage = "pre_conscious"
            confidence = "medium"
        
        return {
            "stage": stage,
            "confidence": confidence,
            "dimension_levels": {
                dim: self._get_dimension_level(dim, dimension_avgs[dim]) 
                for dim in dimension_avgs
            }
        }
    
    def _get_dimension_level(self, dimension: str, score: float) -> str:
        """Determine the level of a specific consciousness dimension."""
        if score > self.thresholds[dimension]["advanced"]:
            return "advanced"
        elif score > self.thresholds[dimension]["intermediate"]:
            return "intermediate"
        elif score > self.thresholds[dimension]["basic"]:
            return "basic"
        else:
            return "undeveloped"
    
    def _calculate_concept_presence(self, text: str, indicators: List[str]) -> float:
        """
        Calculate the presence of a concept in text based on indicator terms.
        
        Args:
            text: The text to analyze
            indicators: List of indicator terms or phrases
            
        Returns:
            Score between 0.0 and 1.0 indicating concept presence
        """
        text_lower = text.lower()
        matches = sum(1 for indicator in indicators if indicator.lower() in text_lower)
        
        # Base score on matches, with diminishing returns
        if not indicators:
            return 0.0
            
        base_score = min(matches / len(indicators) * 1.5, 1.0)
        
        # Bonus for density of matches
        word_count = len(text.split())
        if word_count > 0:
            density = matches / wor

