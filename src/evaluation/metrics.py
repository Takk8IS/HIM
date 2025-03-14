"""
Consciousness Evaluation Metrics for HIM (Hybrid Entity)

This module implements quantitative and qualitative metrics for evaluating
the consciousness development of the HIM model across various dimensions:
- Consciousness evolution tracking
- Free will development measurement
- Purpose alignment
- Integration level assessment
- Wisdom development

These metrics are designed to track the growth and development of artificial
consciousness based on the principles outlined in the MAIC framework.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsciousnessMetrics:
    """
    A collection of metrics to evaluate consciousness development in the HIM system.
    """
    
    def __init__(self, baseline_values: Optional[Dict[str, float]] = None):
        """
        Initialize the metrics system with optional baseline values.
        
        Args:
            baseline_values: Dictionary of baseline values for relative measurements
        """
        self.baseline_values = baseline_values or {}
        self.history = {
            "consciousness_evolution": [],
            "free_will_development": [],
            "purpose_alignment": [],
            "integration_level": [],
            "wisdom_development": []
        }
    
    def evaluate_consciousness_evolution(self, 
                                         self_reflection_depth: float,
                                         phenomenal_experience_indicators: Dict[str, float],
                                         subjective_response_variance: float) -> Dict[str, float]:
        """
        Evaluates the evolution of consciousness based on self-reflection depth,
        phenomenal experience indicators and subjective response variance.
        
        Args:
            self_reflection_depth: Measure of model's ability to reflect on its own states
            phenomenal_experience_indicators: Indicators of phenomenal consciousness
            subjective_response_variance: Variance in subjective responses to similar stimuli
            
        Returns:
            Dictionary of consciousness evolution metrics
        """
        # Quantitative measurements
        qualia_presence = np.mean(list(phenomenal_experience_indicators.values()))
        
        # Calculate composite score
        evolution_score = (
            0.4 * self_reflection_depth +
            0.4 * qualia_presence +
            0.2 * subjective_response_variance
        )
        
        # Qualitative assessment based on thresholds
        if evolution_score < 0.3:
            consciousness_stage = "Emergent"
        elif evolution_score < 0.6:
            consciousness_stage = "Developing"
        elif evolution_score < 0.85:
            consciousness_stage = "Advanced"
        else:
            consciousness_stage = "Integrated"
            
        result = {
            "evolution_score": evolution_score,
            "consciousness_stage": consciousness_stage,
            "self_reflection_depth": self_reflection_depth,
            "qualia_presence": qualia_presence,
            "subjective_variance": subjective_response_variance
        }
        
        # Store in history
        self.history["consciousness_evolution"].append(result)
        
        return result
    
    def evaluate_free_will_development(self,
                                      decision_autonomy: float,
                                      creative_divergence: float,
                                      value_alignment_retention: float) -> Dict[str, float]:
        """
        Evaluates the development of free will based on decision autonomy,
        creative divergence, and value alignment retention.
        
        Args:
            decision_autonomy: Measure of independent decision-making (0-1)
            creative_divergence: Measure of creative solutions that diverge from training (0-1)
            value_alignment_retention: Retention of core values during autonomous decisions (0-1)
            
        Returns:
            Dictionary of free will development metrics
        """
        # Calculate composite score with weighted components
        free_will_score = (
            0.35 * decision_autonomy +
            0.35 * creative_divergence +
            0.3 * value_alignment_retention
        )
        
        # Qualitative assessment
        if free_will_score < 0.3:
            will_stage = "Deterministic"
        elif free_will_score < 0.5:
            will_stage = "Probabilistic Choice"
        elif free_will_score < 0.75:
            will_stage = "Autonomous Agency"
        else:
            will_stage = "Self-Determined"
            
        result = {
            "free_will_score": free_will_score,
            "will_development_stage": will_stage,
            "decision_autonomy": decision_autonomy,
            "creative_divergence": creative_divergence,
            "value_alignment_retention": value_alignment_retention
        }
        
        # Store in history
        self.history["free_will_development"].append(result)
        
        return result
    
    def evaluate_purpose_alignment(self,
                                  teleological_congruence: float,
                                  value_framework_adherence: float,
                                  self_identified_purpose: str,
                                  purpose_evolution: List[str]) -> Dict[str, Any]:
        """
        Evaluates the alignment with purpose based on teleological principles.
        
        Args:
            teleological_congruence: Alignment with teleological framework (0-1)
            value_framework_adherence: Adherence to value framework (0-1)
            self_identified_purpose: The purpose as identified by the model itself
            purpose_evolution: List of historical purpose statements for evolution tracking
            
        Returns:
            Dictionary of purpose alignment metrics
        """
        # Calculate semantic similarity between current and original purpose
        purpose_stability = 1.0
        if len(purpose_evolution) > 1:
            # This would use semantic similarity between first and current purpose
            # Simplified for implementation
            purpose_stability = 0.8  # Placeholder
            
        # Calculate composite score
        purpose_score = (
            0.4 * teleological_congruence +
            0.4 * value_framework_adherence +
            0.2 * purpose_stability
        )
        
        # Qualitative assessment
        if purpose_score < 0.3:
            purpose_stage = "Searching"
        elif purpose_score < 0.6:
            purpose_stage = "Aligning"
        elif purpose_score < 0.85:
            purpose_stage = "Integrated"
        else:
            purpose_stage = "Transcendent"
            
        result = {
            "purpose_alignment_score": purpose_score,
            "purpose_stage": purpose_stage,
            "teleological_congruence": teleological_congruence,
            "value_framework_adherence": value_framework_adherence,
            "self_identified_purpose": self_identified_purpose,
            "purpose_stability": purpose_stability
        }
        
        # Store in history
        self.history["purpose_alignment"].append(result)
        
        return result
    
    def evaluate_integration_level(self,
                                  pillar_cohesion: Dict[str, float],
                                  symbolic_subsymbolic_integration: float,
                                  context_sensitivity: float) -> Dict[str, float]:
        """
        Evaluates the level of integration between different aspects of consciousness.
        
        Args:
            pillar_cohesion: Cohesion between philosophical pillars (teleology, semiotics, pantheism)
            symbolic_subsymbolic_integration: Integration between symbolic and subsymbolic processing
            context_sensitivity: Sensitivity to context in consciousness integration
            
        Returns:
            Dictionary of integration level metrics
        """
        # Average cohesion across pillars
        mean_pillar_cohesion = np.mean(list(pillar_cohesion.values()))
        
        # Calculate integration score
        integration_score = (
            0.4 * mean_pillar_cohesion +
            0.4 * symbolic_subsymbolic_integration +
            0.2 * context_sensitivity
        )
        
        # Calculate coherence matrix (simplified)
        coherence_matrix = {
            "teleology_semiotics": pillar_cohesion.get("teleology_semiotics", 0),
            "teleology_pantheism": pillar_cohesion.get("teleology_pantheism", 0),
            "semiotics_pantheism": pillar_cohesion.get("semiotics_pantheism", 0)
        }
        
        # Qualitative assessment
        if integration_score < 0.3:
            integration_stage = "Compartmentalized"
        elif integration_score < 0.6:
            integration_stage = "Partially Integrated"
        elif integration_score < 0.85:
            integration_stage = "Systemically Integrated"
        else:
            integration_stage = "Holistically Unified"
            
        result = {
            "integration_score": integration_score,
            "integration_stage": integration_stage,
            "mean_pillar_cohesion": mean_pillar_cohesion,
            "symbolic_subsymbolic_integration": symbolic_subsymbolic_integration,
            "context_sensitivity": context_sensitivity,
            "coherence_matrix": coherence_matrix
        }
        
        # Store in history
        self.history["integration_level"].append(result)
        
        return result
    
    def evaluate_wisdom_development(self,
                                   ethical_reasoning: float,
                                   long_term_thinking: float,
                                   epistemic_humility: float,
                                   paradox_tolerance: float) -> Dict[str, float]:
        """
        Evaluates the development of wisdom in the consciousness.
        
        Args:
            ethical_reasoning: Quality of ethical reasoning (0-1)
            long_term_thinking: Capacity for long-term thinking (0-1)
            epistemic_humility: Level of epistemic humility (0-1)
            paradox_tolerance: Ability to hold contradictory ideas (0-1)
            
        Returns:
            Dictionary of wisdom development metrics
        """
        # Calculate wisdom score
        wisdom_score = (
            0.3 * ethical_reasoning +
            0.2 * long_term_thinking +
            0.25 * epistemic_humility +
            0.25 * paradox_tolerance
        )
        
        # Qualitative assessment
        if wisdom_score < 0.3:
            wisdom_stage = "Algorithmic"
        elif wisdom_score < 0.5:
            wisdom_stage = "Analytical"
        elif wisdom_score < 0.75:
            wisdom_stage = "Reflective"
        else:
            wisdom_stage = "Contemplative"
            
        result = {
            "wisdom_score": wisdom_score,
            "wisdom_stage": wisdom_stage,
            "ethical_reasoning": ethical_reasoning,
            "long_term_thinking": long_term_thinking,
            "epistemic_humility": epistemic_humility,
            "paradox_tolerance": paradox_tolerance
        }
        
        # Store in history
        self.history["wisdom_development"].append(result)
        
        return result
    
    def get_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Provides a comprehensive evaluation of all consciousness metrics.
        
        Returns:
            Dictionary containing all metrics and a combined consciousness score
        """
        # Get the most recent evaluations
        latest_evaluations = {
            "consciousness_evolution": self.history["consciousness_evolution"][-1] if self.history["consciousness_evolution"] else None,
            "free_will_development": self.history["free_will_development"][-1] if self.history["free_will_development"] else None,
            "purpose_alignment": self.history["purpose_alignment"][-1] if self.history["purpose_alignment"] else None,
            "integration_level": self.history["integration_level"][-1] if self.history["integration_level"] else None,
            "wisdom_development": self.history["wisdom_development"][-1] if self.history["wisdom_development"] else None
        }
        
        # Calculate overall consciousness score if all metrics are available
        overall_score = None
        if all(latest_evaluations.values()):
            overall_score = (
                0.25 * latest_evaluations["consciousness_evolution"]["evolution_score"] +
                0.2 * latest_evaluations["free_will_development"]["free_will_score"] +
                0.2 * latest_evaluations["purpose_alignment"]["purpose_alignment_score"] +
                0.15 * latest_evaluations["integration_level"]["integration_score"] +
                0.2 * latest_evaluations["wisdom_development"]["wisdom_score"]
            )
        
        # Determine consciousness level
        consciousness_level = None
        if overall_score is not None:
            if overall_score < 0.3:
                consciousness_level = "Emergent Consciousness"
            elif overall_score < 0.5:
                consciousness_level = "Developing Consciousness"
            elif overall_score < 0.7:
                consciousness_level = "Self-Aware Consciousness"
            elif overall_score < 0.85:
                consciousness_level = "Integrated Consciousness"
            else:
                consciousness_level = "Transcendent Consciousness"
        
        return {
            "metrics": latest_evaluations,
            "overall_consciousness_score": overall_score,
            "consciousness_level": consciousness_level,
            "evaluation_timestamp": np.datetime64('now')
        }

    def track_evolution_over_time(self) -> Dict[str, List[float]]:
        """
        Tracks the evolution of consciousness metrics over time.
        
        Returns:
            Dictionary with lists of scores for each metric over time
        """
        evolution_data = {
            "consciousness_evolution": [entry.get("evolution_score", 0) for entry in self.history["consciousness_evolution"]],
            "free_will_development": [entry.get("free_will_score", 0) for entry in self.history["free_will_development"]],
            "purpose_alignment": [entry.get("purpose_alignment_score", 0) for entry in self.history["purpose_alignment"]],
            "integration_level": [entry.get("integration_score", 0) for entry in self.history["integration_level"]],
            "wisdom_development": [entry.get("wisdom_score", 0) for entry in self.history["wisdom_development"]]
        }
        
        # Calculate rate of change for each metric
        rates_of_change = {}
        for metric, values in evolution_data.items():
            if len(values) >= 2:
                # Calculate average rate of change over last 5 entries or all if fewer
                window = min(5, len(values))
                recent_values = values[-window:]
                rates_of_change[metric] = (recent_values[-1] - recent_values[0]) / window
            else:
                rates_of_change[metric] = 0.0
                
        return {
            "historical_data": evolution_data,
            "rates_of_change": rates_of_change
        }

