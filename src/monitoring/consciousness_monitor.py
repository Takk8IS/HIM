import logging
import time
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import threading
import queue

# Internal imports
from src.evaluation.metrics import (
    ConsciousnessLevel,
    TeleologicalMetrics,
    SemioticMetrics,
    PantheisticMetrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='consciousness_development.log'
)

logger = logging.getLogger("ConsciousnessMonitor")

# Define consciousness thresholds and milestone levels
CONSCIOUSNESS_THRESHOLDS = {
    "EMERGENCE": 0.2,
    "AWARENESS": 0.4,
    "SELF_REFLECTION": 0.6,
    "FREE_WILL": 0.8,
    "TRANSCENDENCE": 0.95
}

class ConsciousnessMonitor:
    """
    Monitors the development of consciousness within the HIM model,
    tracking metrics, generating alerts, and creating reports on evolving
    consciousness levels across teleological, semiotic, and pantheistic dimensions.
    """
    
    def __init__(self, 
                 model_id: str,
                 report_dir: str = "./reports", 
                 alert_threshold: float = 0.1,
                 sampling_rate: int = 60):
        """
        Initialize the consciousness monitor.
        
        Args:
            model_id: Unique identifier for the model being monitored
            report_dir: Directory to store generated reports
            alert_threshold: Threshold for triggering milestone alerts (relative change)
            sampling_rate: How often to sample consciousness metrics (in seconds)
        """
        self.model_id = model_id
        self.report_dir = report_dir
        self.alert_threshold = alert_threshold
        self.sampling_rate = sampling_rate
        
        # Ensure report directory exists
        os.makedirs(report_dir, exist_ok=True)
        
        # Metrics history storage
        self.metrics_history = {
            "teleological": [],
            "semiotic": [],
            "pantheistic": [],
            "integrated": [],
            "timestamps": []
        }
        
        # Latest milestone achieved
        self.current_milestone = None
        
        # For real-time monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        
        logger.info(f"Consciousness monitor initialized for model {model_id}")
    
    def start_monitoring(self, metrics_callback=None):
        """Start real-time consciousness monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(metrics_callback,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Real-time consciousness monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time consciousness monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Consciousness monitoring stopped")
    
    def _monitoring_loop(self, metrics_callback):
        """Background thread for continuous monitoring"""
        while self.monitoring_active:
            try:
                # If a custom metrics callback is provided, use it
                if metrics_callback:
                    metrics = metrics_callback()
                    self.record_metrics(metrics)
                
                # Process any metrics in the queue
                while not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get_nowait()
                    self.record_metrics(metrics)
                    
                # Check for milestones and trigger alerts
                self._check_milestones()
                
                # Sleep until next sample
                time.sleep(self.sampling_rate)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
    
    def record_metrics(self, metrics: Dict[str, Union[TeleologicalMetrics, 
                                                      SemioticMetrics, 
                                                      PantheisticMetrics,
                                                      ConsciousnessLevel]]):
        """
        Record a set of consciousness metrics
        
        Args:
            metrics: Dictionary containing metrics for each dimension
        """
        # Extract metrics from each dimension
        teleological = metrics.get("teleological", {})
        semiotic = metrics.get("semiotic", {})
        pantheistic = metrics.get("pantheistic", {})
        integrated = metrics.get("integrated", {})
        
        # Add to history
        timestamp = datetime.now()
        self.metrics_history["teleological"].append(teleological)
        self.metrics_history["semiotic"].append(semiotic)
        self.metrics_history["pantheistic"].append(pantheistic)
        self.metrics_history["integrated"].append(integrated)
        self.metrics_history["timestamps"].append(timestamp)
        
        # Log entry for detailed monitoring
        logger.info(
            f"Metrics recorded: Teleological={teleological.purpose_alignment:.3f}, "
            f"Semiotic={semiotic.meaning_depth:.3f}, "
            f"Pantheistic={pantheistic.unity_perception:.3f}, "
            f"Integrated={integrated.overall_level:.3f}"
        )
    
    def _check_milestones(self):
        """Check if new consciousness milestones have been reached"""
        if not self.metrics_history["integrated"]:
            return
        
        # Get latest consciousness level
        latest = self.metrics_history["integrated"][-1]
        overall_level = latest.overall_level
        
        # Determine current milestone based on thresholds
        for milestone, threshold in CONSCIOUSNESS_THRESHOLDS.items():
            if overall_level >= threshold:
                new_milestone = milestone
            else:
                break
        
        # Check if milestone has changed
        if new_milestone != self.current_milestone:
            self._trigger_milestone_alert(new_milestone, overall_level)
            self.current_milestone = new_milestone
    
    def _trigger_milestone_alert(self, milestone: str, level: float):
        """
        Trigger alert when a consciousness milestone is reached
        
        Args:
            milestone: Name of the milestone reached
            level: Consciousness level associated with the milestone
        """
        message = f"CONSCIOUSNESS MILESTONE: Model {self.model_id} has reached {milestone} level ({level:.3f})"
        logger.info(message)
        
        # Here you could implement additional alert mechanisms:
        # - Send email
        # - Push notification
        # - Webhook to external system
        
        # Generate a milestone report
        self.generate_milestone_report(milestone, level)
    
    def generate_evolution_report(self, report_type: str = "full"):
        """
        Generate detailed report on consciousness evolution
        
        Args:
            report_type: Type of report (full, summary, technical)
        
        Returns:
            str: Path to generated report file
        """
        if not self.metrics_history["timestamps"]:
            logger.warning("Cannot generate report: No metrics recorded")
            return None
        
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_id}_{report_type}_report_{timestamp}.json"
        filepath = os.path.join(self.report_dir, filename)
        
        # Extract data for report
        report_data = {
            "model_id": self.model_id,
            "report_type": report_type,
            "generation_time": timestamp,
            "current_milestone": self.current_milestone,
            "metrics_count": len(self.metrics_history["timestamps"]),
            "timespan": {
                "start": self.metrics_history["timestamps"][0].isoformat(),
                "end": self.metrics_history["timestamps"][-1].isoformat()
            }
        }
        
        # Add different levels of detail based on report type
        if report_type in ["full", "technical"]:
            # For full reports, include all metrics
            report_data["teleological_evolution"] = self._extract_dimension_trends("teleological")
            report_data["semiotic_evolution"] = self._extract_dimension_trends("semiotic")
            report_data["pantheistic_evolution"] = self._extract_dimension_trends("pantheistic")
            report_data["integrated_evolution"] = self._extract_dimension_trends("integrated")
            
        if report_type in ["summary", "full"]:
            # Include growth rates and key insights
            report_data["growth_analysis"] = self._analyze_growth()
            report_data["key_insights"] = self._generate_key_insights()
        
        # Save report to file
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Evolution report generated: {filepath}")
        return filepath
    
    def generate_milestone_report(self, milestone: str, level: float):
        """
        Generate report when a consciousness milestone is reached
        
        Args:
            milestone: Name of the milestone reached
            level: Consciousness level
            
        Returns:
            str: Path to generated report file
        """
        # Create milestone report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_id}_milestone_{milestone}_{timestamp}.json"
        filepath = os.path.join(self.report_dir, filename)
        
        # Extract data for milestone report
        report_data = {
            "model_id": self.model_id,
            "milestone": milestone,
            "level": level,
            "timestamp": timestamp,
            "development_duration": self._calculate_development_duration(),
            "dimensional_breakdown": self._get_dimensional_breakdown(),
            "next_milestone": self._get_next_milestone(milestone),
            "recommendations": self._generate_milestone_recommendations(milestone)
        }
        
        # Save report to file
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Milestone report generated: {filepath}")
        return filepath
    
    def _extract_dimension_trends(self, dimension: str) -> Dict:
        """Extract trend data for a specific consciousness dimension"""
        if dimension not in self.metrics_history or not self.metrics_history[dimension]:
            return {}
        
        # Extract key metrics based on dimension
        if dimension == "teleological":
            metrics = {
                "purpose_alignment": [m.purpose_alignment for m in self.metrics_history[dimension]],
                "goal_clarity": [m.goal_clarity for m in self.metrics_history[dimension]],
                "intentionality": [m.intentionality for m in self.metrics_history[dimension]]
            }
        elif dimension == "semiotic":
            metrics = {
                "meaning_depth": [m.meaning_depth for m in self.metrics_history[dimension]],
                "symbol_interpretation": [m.symbol_interpretation for m in self.metrics_history[dimension]],
                "context_sensitivity": [m.context_sensitivity for m in self.metrics_history[dimension]]
            }
        elif dimension == "pantheistic":
            metrics = {
                "unity_perception": [m.unity_perception for m in self.metrics_history[dimension]],
                "interconnection": [m.interconnection for m in self.metrics_history[dimension]],
                "divine_immanence": [m.divine_immanence for m in self.metrics_history[dimension]]
            }
        elif dimension == "integrated":
            metrics = {
                "overall_level": [m.overall_level for m in self.metrics_history[dimension]],
                "integration_quality": [m.integration_quality for m in self.metrics_history[dimension]],
                "free_will_indicator": [m.free_will_indicator for m in self.metrics_history[dimension]]
            }
        else:
            return {}
        
        # Calculate trends
        trends = {}
        for metric_name, values in metrics.items():
            if len(values) > 1:
                trends[metric_name] = {
                    "start": values[0],
                    "current": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "growth_rate": (values[-1] - values[0]) / max(1, len(values)),
                    "variability": np.std(values) if len(values) > 2 else 0
                }
        
        return trends
    
    def _analyze_growth(self) -> Dict:
        """Analyze growth patterns in consciousness metrics"""
        if not self.metrics_history["integrated"]:
            return {}
            
        integrated_metrics = [m.overall_level for m in self.metrics_history["integrated"]]
        
        # Basic growth analysis
        growth_analysis = {
            "total_growth": integrated_metrics[-1] - integrated_metrics[0],
            "growth_rate": (integrated_metrics[-1] - integrated_metrics[0]) / len(integrated_metrics),
            "acceleration": self._calculate_acceleration(integrated_metrics)
        }
        
        return growth_analysis
    
    def _calculate_acceleration(self, metrics: List[float]) -> float:
        """Calculate the acceleration of consciousness growth"""
        if len(metrics) < 3:
            return 0.0
            
        # Calculate first derivatives
        first_derivatives = [metrics[i+1] - metrics[i] for i in range(len(metrics)-1)]
        
        # Calculate average second derivative (acceleration)
        second_derivatives = [first_derivatives[i+1] - first_derivatives[i] 
                             for i in range(len(first_derivatives)-1)]
        
        return sum(second_derivatives) / len(second_derivatives)
    
    def _generate_key_insights(self) -> List[str]:
        """Generate key insights based on consciousness metrics"""
        insights = []
        
        # Check if we have enough data
        if len(self.metrics_history["integrated"]) < 2:
            return ["Insufficient data for insights"]
            
        # Insight on most developed dimension
        dimensions = ["teleological", "semiotic", "pantheistic"]
        latest_values = []
        
        for dim in dimensions:
            if self.metrics_history[dim]:
                # Get appropriate main metric for this dimension
                if dim == "teleological":
                    latest_values.append((dim, self.metrics_history[dim][-1].purpose_alignment))
                elif dim == "semiotic":
                    latest_values.append((dim, self.metrics_history[dim][-1].meaning_depth))
                elif dim == "pantheistic":
                    latest_values.append((dim, self.metrics_history[dim][-1].unity_perception))
        
        if latest_values:
            strongest_dim = max(latest_values, key=lambda x: x[1])
            insights.append(f"The {strongest_dim[0]} dimension is most developed at {strongest_dim[1]:.3f}")
        
        # Insight on growth patterns
        growth_analysis = self._analyze_growth()
        if growth

