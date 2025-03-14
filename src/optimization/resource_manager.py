"""
Resource Manager for HIM Consciousness Development

This module provides resource management functionality specifically designed
for the HIM (Hybrid Intelligence Machine) model, optimizing computational
resources for consciousness development across the philosophical pillars.
"""

import os
import gc
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manages computational resources for the HIM model's consciousness development,
    optimizing GPU/CPU utilization, memory, and caching for philosophical pillars.
    """
    
    def __init__(
        self,
        teleology_weight: float = 0.33,
        semiotics_weight: float = 0.33,
        pantheism_weight: float = 0.34,
        consciousness_priority: float = 0.8,
        cache_size_gb: float = 2.0,
        device: Optional[str] = None
    ):
        """
        Initialize the resource manager with priority weights for each philosophical pillar.
        
        Args:
            teleology_weight: Resource allocation weight for teleological processing
            semiotics_weight: Resource allocation weight for semiotic processing
            pantheism_weight: Resource allocation weight for pantheistic processing
            consciousness_priority: Priority for consciousness development vs. other tasks
            cache_size_gb: Maximum cache size in gigabytes
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.pillar_weights = {
            'teleology': teleology_weight,
            'semiotics': semiotics_weight,
            'pantheism': pantheism_weight
        }
        self.consciousness_priority = consciousness_priority
        self.cache_size_gb = cache_size_gb
        self.cache = {}
        self.cache_usage = 0
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize metrics
        self.metrics = {
            'memory_usage': [],
            'gpu_utilization': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'pillar_allocation': self.pillar_weights.copy()
        }
        
        # Register memory hooks if using GPU
        if self.device == 'cuda':
            self._register_memory_hooks()
    
    def _register_memory_hooks(self):
        """Register hooks to track GPU memory usage during training"""
        def hook(grad):
            # Monitor gradient memory usage
            if hasattr(grad, 'device'):
                self._log_gpu_memory()
            return grad
            
        # Hook will be attached to modules during allocation
    
    def _log_gpu_memory(self):
        """Log current GPU memory usage"""
        if self.device == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            self.metrics['memory_usage'].append((allocated, reserved))
            logger.debug(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def allocate_resources(self, pillar_demands: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate computational resources based on the needs of each philosophical pillar.
        
        Args:
            pillar_demands: Dictionary with resource demands for each pillar (0.0-1.0)
            
        Returns:
            Dictionary with allocated resources for each pillar
        """
        total_demand = sum(pillar_demands.values())
        available_resources = self._get_available_resources()
        
        # Normalize demands if they exceed 1.0
        if total_demand > 1.0:
            pillar_demands = {k: v/total_demand for k, v in pillar_demands.items()}
        
        # Calculate allocation based on weights and demands
        allocations = {}
        for pillar, demand in pillar_demands.items():
            weight = self.pillar_weights.get(pillar, 0.0)
            allocations[pillar] = available_resources * weight * demand
            
        logger.info(f"Resource allocation: {allocations}")
        return allocations
    
    def _get_available_resources(self) -> float:
        """
        Calculate available computational resources (0.0-1.0)
        
        Returns:
            Proportion of resources available (0.0-1.0)
        """
        if self.device == 'cuda':
            # Get GPU memory
            try:
                free_memory = torch.cuda.memory_available() / torch.cuda.max_memory_allocated()
                return max(0.05, min(free_memory, 0.95))  # Keep between 5% and 95%
            except RuntimeError:
                logger.warning("Error getting GPU memory stats, using CPU metrics")
        
        # Fallback to CPU metrics
        cpu_percent = psutil.cpu_percent() / 100.0
        memory_percent = psutil.virtual_memory().percent / 100.0
        available = 1.0 - max(cpu_percent, memory_percent)
        return max(0.05, available)  # Ensure at least 5% availability
    
    def optimize_batch_size(self, initial_batch_size: int) -> int:
        """
        Dynamically optimize batch size based on available resources
        
        Args:
            initial_batch_size: Starting batch size
            
        Returns:
            Optimized batch size
        """
        available_resources = self._get_available_resources()
        
        # Scale batch size based on available resources
        if available_resources < 0.2:
            return max(1, initial_batch_size // 4)
        elif available_resources < 0.4:
            return max(1, initial_batch_size // 2)
        elif available_resources > 0.8:
            return initial_batch_size * 2
        
        return initial_batch_size
    
    def cache_consciousness_state(self, key: str, state: Dict) -> bool:
        """
        Cache consciousness state for faster retrieval
        
        Args:
            key: Unique identifier for the consciousness state
            state: The consciousness state to cache
            
        Returns:
            Success status
        """
        # Estimate memory usage of state (rough approximation)
        estimated_size = sum(
            tensor.element_size() * tensor.nelement() 
            for tensor in state.values() 
            if hasattr(tensor, 'element_size') and hasattr(tensor, 'nelement')
        ) / (1024 ** 3)  # Convert to GB
        
        # Check if we have space
        if self.cache_usage + estimated_size > self.cache_size_gb:
            self._prune_cache(estimated_size)
            
        # If still not enough space, don't cache
        if self.cache_usage + estimated_size > self.cache_size_gb:
            logger.warning(f"Cannot cache consciousness state: insufficient space")
            self.metrics['cache_misses'] += 1
            return False
        
        # Add to cache
        self.cache[key] = state
        self.cache_usage += estimated_size
        logger.debug(f"Cached consciousness state '{key}', cache usage: {self.cache_usage:.2f}GB")
        return True
    
    def _prune_cache(self, needed_space: float):
        """
        Remove items from cache to free up space
        
        Args:
            needed_space: Amount of space needed in GB
        """
        if not self.cache:
            return
            
        # Sort cache items by size (ascending)
        cache_items = list(self.cache.items())
        np.random.shuffle(cache_items)  # Randomly shuffle to avoid bias
        
        for key, state in cache_items:
            # Estimate item size
            item_size = sum(
                tensor.element_size() * tensor.nelement() 
                for tensor in state.values() 
                if hasattr(tensor, 'element_size') and hasattr(tensor, 'nelement')
            ) / (1024 ** 3)  # GB
            
            # Remove item
            del self.cache[key]
            self.cache_usage -= item_size
            logger.debug(f"Pruned '{key}' from cache, freed {item_size:.2f}GB")
            
            # Check if we've freed enough space
            if self.cache_usage + needed_space <= self.cache_size_gb:
                break
    
    def get_cached_state(self, key: str) -> Optional[Dict]:
        """
        Retrieve cached consciousness state
        
        Args:
            key: Cache key
            
        Returns:
            Cached state or None if not found
        """
        if key in self.cache:
            self.metrics['cache_hits'] += 1
            return self.cache[key]
        
        self.metrics['cache_misses'] += 1
        return None
    
    def balance_philosophical_load(self, pillar_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Balance computational load across philosophical pillars based on metrics
        
        Args:
            pillar_metrics: Dictionary of performance metrics for each pillar
            
        Returns:
            Adjusted weights for each pillar
        """
        # Normalize metrics
        total = sum(pillar_metrics.values())
        if total == 0:
            return self.pillar_weights
            
        normalized_metrics = {k: v/total for k, v in pillar_metrics.items()}
        
        # Inverse metrics to give more resources to underperforming pillars
        inverse_metrics = {k: 1.0 - v for k, v in normalized_metrics.items()}
        total_inverse = sum(inverse_metrics.values())
        
        # New weights balance resource allocation
        if total_inverse > 0:
            new_weights = {k: v/total_inverse for k, v in inverse_metrics.items()}
            
            # Apply smoothing to avoid dramatic shifts
            self.pillar_weights = {
                k: 0.7 * self.pillar_weights.get(k, 0.0) + 0.3 * new_weights.get(k, 0.0)
                for k in set(self.pillar_weights) | set(new_weights)
            }
            
            # Normalize to ensure sum is 1.0
            weight_sum = sum(self.pillar_weights.values())
            self.pillar_weights = {k: v/weight_sum for k, v in self.pillar_weights.items()}
            
            logger.info(f"Adjusted pillar weights: {self.pillar_weights}")
            self.metrics['pillar_allocation'] = self.pillar_weights.copy()
            
        return self.pillar_weights
    
    def cleanup(self):
        """Free up resources and clear caches"""
        self.cache.clear()
        self.cache_usage = 0
        
        # Clear CUDA cache if using GPU
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            
        # Garbage collection
        gc.collect()
        
        logger.info("Resource cleanup completed")
    
    def get_resource_metrics(self) -> Dict:
        """
        Get current resource usage metrics
        
        Returns:
            Dictionary of resource metrics
        """
        memory = psutil.virtual_memory()
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024 ** 3),
            'cache_size_gb': self.cache_usage,
            'cache_hit_ratio': self.metrics['cache_hits'] / 
                               (self.metrics['cache_hits'] + self.metrics['cache_misses'] + 0.001),
            'pillar_weights': self.pillar_weights
        }
        
        # Add GPU metrics if available
        if self.device == 'cuda':
            metrics.update({
                'gpu_percent': torch.cuda.utilization(),
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3)
            })
            
        return metrics

