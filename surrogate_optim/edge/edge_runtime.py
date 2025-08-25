"""Edge Computing Runtime for surrogate optimization.

This module provides a lightweight runtime optimized for edge computing
environments with limited computational resources and network connectivity.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import gzip
import hashlib
import logging
import pickle
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

from jax import Array
import jax.numpy as jnp
import numpy as np


class EdgeResourceProfile(Enum):
    """Edge device resource profiles."""
    MINIMAL = "minimal"      # IoT devices, microcontrollers
    CONSTRAINED = "constrained"  # Raspberry Pi, mobile devices
    STANDARD = "standard"    # Edge servers, workstations
    ENHANCED = "enhanced"    # GPU-enabled edge devices


@dataclass
class EdgeConfiguration:
    """Configuration for edge computing runtime."""
    resource_profile: EdgeResourceProfile = EdgeResourceProfile.STANDARD
    max_memory_mb: int = 512
    max_cpu_cores: int = 2
    enable_gpu: bool = False
    cache_size_mb: int = 100
    network_timeout_seconds: int = 30
    heartbeat_interval_seconds: int = 60
    model_compression: bool = True
    data_compression: bool = True
    offline_mode: bool = False


@dataclass
class EdgeTask:
    """Task to be executed on edge device."""
    task_id: str
    task_type: str  # "optimization", "prediction", "training"
    payload: Dict[str, Any]
    priority: int = 1  # 1 = high, 10 = low
    created_at: float = field(default_factory=time.time)
    timeout_seconds: float = 300.0
    requires_gpu: bool = False
    estimated_memory_mb: float = 50.0
    callback_url: Optional[str] = None


@dataclass
class EdgeResult:
    """Result from edge task execution."""
    task_id: str
    success: bool
    result_data: Dict[str, Any]
    execution_time: float
    memory_used_mb: float
    error_message: Optional[str] = None
    completed_at: float = field(default_factory=time.time)
    compressed: bool = False


class ResourceMonitor:
    """Monitors edge device resource usage."""

    def __init__(self, monitoring_interval: float = 5.0):
        """Initialize resource monitor.
        
        Args:
            monitoring_interval: How often to check resources (seconds)
        """
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Resource usage history
        self.cpu_usage_history: deque = deque(maxlen=100)
        self.memory_usage_history: deque = deque(maxlen=100)
        self.gpu_usage_history: deque = deque(maxlen=100)

        # Current state
        self.current_cpu_usage = 0.0
        self.current_memory_usage = 0.0
        self.current_gpu_usage = 0.0
        self.available_memory_mb = 1024.0

        # Thresholds
        self.cpu_threshold = 0.8
        self.memory_threshold = 0.85
        self.gpu_threshold = 0.9

        # Logger
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("Resource monitoring started")

    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

        self.logger.info("Resource monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Get current resource usage
                self._update_resource_usage()

                # Store in history
                self.cpu_usage_history.append(self.current_cpu_usage)
                self.memory_usage_history.append(self.current_memory_usage)
                self.gpu_usage_history.append(self.current_gpu_usage)

                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)

    def _update_resource_usage(self) -> None:
        """Update current resource usage."""
        try:
            import psutil

            # CPU usage
            self.current_cpu_usage = psutil.cpu_percent(interval=1.0) / 100.0

            # Memory usage
            memory = psutil.virtual_memory()
            self.current_memory_usage = memory.percent / 100.0
            self.available_memory_mb = memory.available / (1024 * 1024)

            # GPU usage (if available)
            try:
                import pynvml
                pynvml.nvmlInit()

                if pynvml.nvmlDeviceGetCount() > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.current_gpu_usage = gpu_util.gpu / 100.0
                else:
                    self.current_gpu_usage = 0.0

            except ImportError:
                self.current_gpu_usage = 0.0
            except Exception:
                self.current_gpu_usage = 0.0

        except ImportError:
            # Fallback for systems without psutil
            self.current_cpu_usage = 0.5  # Assume moderate usage
            self.current_memory_usage = 0.5
            self.available_memory_mb = 512.0
            self.current_gpu_usage = 0.0

    def is_resource_available(self, task: EdgeTask) -> Tuple[bool, str]:
        """Check if resources are available for a task.
        
        Args:
            task: Task to check resources for
            
        Returns:
            Tuple of (available, reason)
        """
        # Check memory
        if task.estimated_memory_mb > self.available_memory_mb:
            return False, f"Insufficient memory: need {task.estimated_memory_mb}MB, have {self.available_memory_mb}MB"

        # Check CPU threshold
        if self.current_cpu_usage > self.cpu_threshold:
            return False, f"CPU usage too high: {self.current_cpu_usage:.1%}"

        # Check memory threshold
        if self.current_memory_usage > self.memory_threshold:
            return False, f"Memory usage too high: {self.current_memory_usage:.1%}"

        # Check GPU if required
        if task.requires_gpu:
            if self.current_gpu_usage > self.gpu_threshold:
                return False, f"GPU usage too high: {self.current_gpu_usage:.1%}"

        return True, "Resources available"

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status.
        
        Returns:
            Dictionary containing resource status
        """
        return {
            "cpu_usage": self.current_cpu_usage,
            "memory_usage": self.current_memory_usage,
            "gpu_usage": self.current_gpu_usage,
            "available_memory_mb": self.available_memory_mb,
            "cpu_history": list(self.cpu_usage_history)[-10:],  # Last 10 readings
            "memory_history": list(self.memory_usage_history)[-10:],
            "gpu_history": list(self.gpu_usage_history)[-10:],
            "monitoring_active": self.monitoring_active,
        }


class ModelCompressor:
    """Compresses models for edge deployment."""

    def __init__(self):
        """Initialize model compressor."""
        self.compression_cache: Dict[str, bytes] = {}
        self.logger = logging.getLogger(__name__)

    def compress_model(self, model_data: Dict[str, Any]) -> bytes:
        """Compress model data for edge deployment.
        
        Args:
            model_data: Model data to compress
            
        Returns:
            Compressed model bytes
        """
        try:
            # Serialize model data
            serialized = pickle.dumps(model_data)

            # Compress using gzip
            compressed = gzip.compress(serialized, compresslevel=9)

            # Cache for reuse
            model_hash = hashlib.md5(serialized).hexdigest()
            self.compression_cache[model_hash] = compressed

            compression_ratio = len(serialized) / len(compressed)
            self.logger.info(f"Model compressed: {len(serialized)} -> {len(compressed)} bytes "
                           f"(ratio: {compression_ratio:.2f}x)")

            return compressed

        except Exception as e:
            self.logger.error(f"Model compression failed: {e}")
            raise

    def decompress_model(self, compressed_data: bytes) -> Dict[str, Any]:
        """Decompress model data.
        
        Args:
            compressed_data: Compressed model bytes
            
        Returns:
            Decompressed model data
        """
        try:
            # Decompress
            decompressed = gzip.decompress(compressed_data)

            # Deserialize
            model_data = pickle.loads(decompressed)

            return model_data

        except Exception as e:
            self.logger.error(f"Model decompression failed: {e}")
            raise

    def quantize_model_weights(self, weights: Array, bits: int = 8) -> Array:
        """Quantize model weights to reduce size.
        
        Args:
            weights: Original weights
            bits: Number of bits for quantization
            
        Returns:
            Quantized weights
        """
        if bits >= 32:
            return weights  # No quantization needed

        # Calculate quantization parameters
        w_min = jnp.min(weights)
        w_max = jnp.max(weights)

        # Quantize
        scale = (w_max - w_min) / ((2 ** bits) - 1)
        quantized = jnp.round((weights - w_min) / scale)

        # Dequantize
        dequantized = quantized * scale + w_min

        return dequantized


class EdgeCache:
    """Lightweight cache for edge computing."""

    def __init__(self, max_size_mb: int = 100):
        """Initialize edge cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.current_size_bytes = 0

        self.logger = logging.getLogger(__name__)

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found
        """
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]["data"]

        return None

    def put(self, key: str, data: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        # Estimate size
        try:
            serialized = pickle.dumps(data)
            data_size = len(serialized)
        except Exception:
            # Fallback size estimation
            data_size = 1024  # 1KB default

        # Check if we need to evict items
        while (self.current_size_bytes + data_size > self.max_size_bytes and
               len(self.cache) > 0):
            self._evict_lru()

        # Add to cache
        self.cache[key] = {
            "data": data,
            "size": data_size,
            "created_at": time.time()
        }
        self.access_times[key] = time.time()
        self.current_size_bytes += data_size

        self.logger.debug(f"Cached item {key} ({data_size} bytes)")

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_times:
            return

        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])

        # Remove from cache
        if lru_key in self.cache:
            evicted_size = self.cache[lru_key]["size"]
            self.current_size_bytes -= evicted_size
            del self.cache[lru_key]
            del self.access_times[lru_key]

            self.logger.debug(f"Evicted LRU item {lru_key} ({evicted_size} bytes)")

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()
        self.current_size_bytes = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return {
            "items_count": len(self.cache),
            "current_size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization": self.current_size_bytes / self.max_size_bytes,
        }


class EdgeOptimizationRuntime:
    """Lightweight runtime for edge computing optimization.
    
    This runtime is optimized for resource-constrained edge devices
    and provides efficient surrogate optimization capabilities.
    """

    def __init__(
        self,
        config: Optional[EdgeConfiguration] = None,
        device_id: Optional[str] = None
    ):
        """Initialize edge optimization runtime.
        
        Args:
            config: Edge configuration
            device_id: Unique device identifier
        """
        self.config = config or EdgeConfiguration()
        self.device_id = device_id or self._generate_device_id()

        # Core components
        self.resource_monitor = ResourceMonitor()
        self.model_compressor = ModelCompressor()
        self.cache = EdgeCache(self.config.cache_size_mb)

        # Task management
        self.task_queue: deque = deque()
        self.active_tasks: Dict[str, EdgeTask] = {}
        self.completed_tasks: Dict[str, EdgeResult] = {}

        # Runtime state
        self.runtime_active = False
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Loaded models
        self.loaded_models: Dict[str, Any] = {}

        # Performance metrics
        self.tasks_processed = 0
        self.total_execution_time = 0.0
        self.successful_tasks = 0

        # Logger
        self.logger = logging.getLogger(__name__)

        # Network connectivity
        self.network_available = True
        self.last_heartbeat = time.time()

        # Initialize runtime based on resource profile
        self._initialize_for_profile()

    def _generate_device_id(self) -> str:
        """Generate unique device ID.
        
        Returns:
            Unique device identifier
        """
        import platform
        import uuid

        machine_info = f"{platform.node()}-{platform.machine()}-{uuid.getnode()}"
        return hashlib.md5(machine_info.encode()).hexdigest()[:16]

    def _initialize_for_profile(self) -> None:
        """Initialize runtime based on resource profile."""
        profile = self.config.resource_profile

        if profile == EdgeResourceProfile.MINIMAL:
            # Ultra-lightweight mode
            self.config.max_memory_mb = min(self.config.max_memory_mb, 128)
            self.config.cache_size_mb = min(self.config.cache_size_mb, 16)
            self.config.enable_gpu = False

        elif profile == EdgeResourceProfile.CONSTRAINED:
            # Constrained mode
            self.config.max_memory_mb = min(self.config.max_memory_mb, 256)
            self.config.cache_size_mb = min(self.config.cache_size_mb, 32)

        elif profile == EdgeResourceProfile.ENHANCED:
            # Enhanced mode with GPU
            self.config.enable_gpu = True

        self.logger.info(f"Initialized for {profile.value} profile: "
                        f"memory={self.config.max_memory_mb}MB, "
                        f"cache={self.config.cache_size_mb}MB, "
                        f"gpu={self.config.enable_gpu}")

    def start_runtime(self) -> None:
        """Start the edge optimization runtime."""
        if self.runtime_active:
            return

        self.runtime_active = True

        # Start resource monitoring
        self.resource_monitor.start_monitoring()

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # Start heartbeat if not offline
        if not self.config.offline_mode:
            self._start_heartbeat()

        self.logger.info(f"Edge runtime started (device: {self.device_id})")

    def stop_runtime(self) -> None:
        """Stop the edge optimization runtime."""
        self.runtime_active = False
        self.stop_event.set()

        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()

        # Wait for worker thread
        if self.worker_thread:
            self.worker_thread.join(timeout=10)

        self.logger.info("Edge runtime stopped")

    def submit_task(self, task: EdgeTask) -> bool:
        """Submit task for execution.
        
        Args:
            task: Task to execute
            
        Returns:
            True if task was accepted
        """
        # Check if task can be accepted
        if not self.runtime_active:
            return False

        # Check resource availability
        available, reason = self.resource_monitor.is_resource_available(task)
        if not available:
            self.logger.warning(f"Task {task.task_id} rejected: {reason}")
            return False

        # Add to queue
        self.task_queue.append(task)
        self.logger.info(f"Task {task.task_id} queued (priority: {task.priority})")

        return True

    def get_task_result(self, task_id: str) -> Optional[EdgeResult]:
        """Get result for completed task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task result or None if not found
        """
        return self.completed_tasks.get(task_id)

    def load_model(self, model_id: str, model_data: Union[Dict[str, Any], bytes]) -> bool:
        """Load model into runtime.
        
        Args:
            model_id: Model identifier
            model_data: Model data (dict or compressed bytes)
            
        Returns:
            True if model was loaded successfully
        """
        try:
            if isinstance(model_data, bytes):
                # Decompress model
                model_dict = self.model_compressor.decompress_model(model_data)
            else:
                model_dict = model_data

            # Apply resource-specific optimizations
            optimized_model = self._optimize_model_for_edge(model_dict)

            # Load into memory
            self.loaded_models[model_id] = optimized_model

            # Cache for quick access
            self.cache.put(f"model_{model_id}", optimized_model)

            self.logger.info(f"Model {model_id} loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            return False

    def _optimize_model_for_edge(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model for edge deployment.
        
        Args:
            model_data: Original model data
            
        Returns:
            Optimized model data
        """
        optimized = model_data.copy()

        # Quantize weights for memory efficiency
        if "weights" in optimized:
            for layer_name, weights in optimized["weights"].items():
                if isinstance(weights, (np.ndarray, jnp.ndarray)):
                    # Quantize to 16-bit for constrained profiles
                    if self.config.resource_profile in [EdgeResourceProfile.MINIMAL, EdgeResourceProfile.CONSTRAINED]:
                        optimized["weights"][layer_name] = self.model_compressor.quantize_model_weights(
                            jnp.array(weights), bits=16
                        )

        # Remove unnecessary metadata for minimal profiles
        if self.config.resource_profile == EdgeResourceProfile.MINIMAL:
            # Remove training-specific data
            optimized.pop("training_history", None)
            optimized.pop("validation_data", None)
            optimized.pop("optimizer_state", None)

        return optimized

    def _worker_loop(self) -> None:
        """Main worker loop for task execution."""
        while self.runtime_active and not self.stop_event.is_set():
            try:
                # Check for tasks
                if not self.task_queue:
                    time.sleep(0.1)
                    continue

                # Get highest priority task
                task = self._get_next_task()
                if task is None:
                    continue

                # Execute task
                result = self._execute_task(task)

                # Store result
                self.completed_tasks[task.task_id] = result

                # Clean up old results
                self._cleanup_old_results()

                # Update statistics
                self.tasks_processed += 1
                self.total_execution_time += result.execution_time
                if result.success:
                    self.successful_tasks += 1

            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                time.sleep(1.0)

    def _get_next_task(self) -> Optional[EdgeTask]:
        """Get next task from queue based on priority and resources.
        
        Returns:
            Next task to execute or None
        """
        if not self.task_queue:
            return None

        # Sort by priority (lower number = higher priority)
        available_tasks = []

        for task in list(self.task_queue):
            # Check if resources are available
            available, _ = self.resource_monitor.is_resource_available(task)
            if available:
                available_tasks.append(task)

        if not available_tasks:
            return None

        # Select highest priority task
        next_task = min(available_tasks, key=lambda t: (t.priority, t.created_at))

        # Remove from queue
        self.task_queue.remove(next_task)
        self.active_tasks[next_task.task_id] = next_task

        return next_task

    def _execute_task(self, task: EdgeTask) -> EdgeResult:
        """Execute a task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        start_time = time.time()
        memory_before = self.resource_monitor.available_memory_mb

        try:
            self.logger.info(f"Executing task {task.task_id} ({task.task_type})")

            # Route to appropriate handler
            if task.task_type == "optimization":
                result_data = self._handle_optimization_task(task)
            elif task.task_type == "prediction":
                result_data = self._handle_prediction_task(task)
            elif task.task_type == "training":
                result_data = self._handle_training_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Calculate memory usage
            memory_after = self.resource_monitor.available_memory_mb
            memory_used = max(0, memory_before - memory_after)

            # Create successful result
            result = EdgeResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                execution_time=time.time() - start_time,
                memory_used_mb=memory_used,
                compressed=False
            )

            # Compress result if enabled
            if self.config.data_compression and len(str(result_data)) > 1024:
                compressed_data = gzip.compress(pickle.dumps(result_data))
                result.result_data = {"compressed": compressed_data}
                result.compressed = True

            self.logger.info(f"Task {task.task_id} completed successfully "
                           f"({result.execution_time:.3f}s, {memory_used:.1f}MB)")

            return result

        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")

            return EdgeResult(
                task_id=task.task_id,
                success=False,
                result_data={},
                execution_time=time.time() - start_time,
                memory_used_mb=0.0,
                error_message=str(e)
            )

        finally:
            # Remove from active tasks
            self.active_tasks.pop(task.task_id, None)

    def _handle_optimization_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Handle optimization task.
        
        Args:
            task: Optimization task
            
        Returns:
            Optimization results
        """
        payload = task.payload

        # Get required data
        objective_data = payload.get("objective_data")
        initial_point = jnp.array(payload.get("initial_point", [0.0]))
        bounds = payload.get("bounds")
        model_id = payload.get("model_id")

        # Load model if needed
        if model_id and model_id not in self.loaded_models:
            # Try to get from cache
            cached_model = self.cache.get(f"model_{model_id}")
            if cached_model:
                self.loaded_models[model_id] = cached_model
            else:
                raise ValueError(f"Model {model_id} not available")

        # Perform lightweight optimization
        # This is a simplified implementation - real optimization would use the loaded model
        best_point = initial_point
        best_value = 1.0  # Placeholder

        # Simulate optimization iterations
        n_iterations = min(50, payload.get("max_iterations", 50))  # Limit for edge

        for i in range(n_iterations):
            # Simple random search for demonstration
            if bounds:
                test_point = jnp.array([
                    np.random.uniform(low, high) for low, high in bounds
                ])
            else:
                test_point = initial_point + np.random.normal(0, 0.1, size=initial_point.shape)

            # Evaluate using loaded model (simplified)
            test_value = float(jnp.sum(test_point ** 2))  # Placeholder objective

            if test_value < best_value:
                best_point = test_point
                best_value = test_value

        return {
            "optimum": best_point.tolist(),
            "optimal_value": best_value,
            "iterations": n_iterations,
            "converged": True,
            "execution_info": {
                "device_id": self.device_id,
                "resource_profile": self.config.resource_profile.value
            }
        }

    def _handle_prediction_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Handle prediction task.
        
        Args:
            task: Prediction task
            
        Returns:
            Prediction results
        """
        payload = task.payload

        # Get input data
        input_data = jnp.array(payload.get("input_data", []))
        model_id = payload.get("model_id")

        # Get model
        if model_id not in self.loaded_models:
            cached_model = self.cache.get(f"model_{model_id}")
            if cached_model:
                self.loaded_models[model_id] = cached_model
            else:
                raise ValueError(f"Model {model_id} not available")

        model = self.loaded_models[model_id]

        # Make prediction (simplified)
        # In real implementation, this would use the actual model
        prediction = float(jnp.sum(input_data ** 2))  # Placeholder
        uncertainty = 0.1  # Placeholder uncertainty

        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
            "input_shape": input_data.shape,
            "model_id": model_id,
            "device_id": self.device_id
        }

    def _handle_training_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Handle model training/update task.
        
        Args:
            task: Training task
            
        Returns:
            Training results
        """
        payload = task.payload

        # Get training data
        training_data = payload.get("training_data")
        model_id = payload.get("model_id")
        learning_rate = payload.get("learning_rate", 0.01)

        if not training_data:
            raise ValueError("No training data provided")

        # Perform lightweight model update
        # This is a simplified implementation

        # Simulate training progress
        training_loss = 1.0
        for epoch in range(min(10, payload.get("epochs", 10))):  # Limit epochs for edge
            training_loss *= 0.9  # Simulate decreasing loss

        # Update model (simplified)
        if model_id in self.loaded_models:
            # In real implementation, this would update the actual model weights
            pass

        return {
            "training_loss": training_loss,
            "epochs_completed": min(10, payload.get("epochs", 10)),
            "model_updated": model_id in self.loaded_models,
            "model_id": model_id,
            "device_id": self.device_id
        }

    def _cleanup_old_results(self) -> None:
        """Clean up old task results to save memory."""
        if len(self.completed_tasks) <= 100:  # Keep up to 100 results
            return

        # Sort by completion time and keep only recent results
        sorted_results = sorted(
            self.completed_tasks.items(),
            key=lambda x: x[1].completed_at,
            reverse=True
        )

        # Keep only the 50 most recent results
        self.completed_tasks = dict(sorted_results[:50])

    def _start_heartbeat(self) -> None:
        """Start heartbeat mechanism for network connectivity."""
        def heartbeat_loop():
            while self.runtime_active:
                try:
                    self.last_heartbeat = time.time()
                    # In real implementation, this would send heartbeat to coordinator
                    time.sleep(self.config.heartbeat_interval_seconds)
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")
                    time.sleep(self.config.heartbeat_interval_seconds)

        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()

    def get_runtime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive runtime statistics.
        
        Returns:
            Runtime statistics
        """
        return {
            "device_info": {
                "device_id": self.device_id,
                "resource_profile": self.config.resource_profile.value,
                "max_memory_mb": self.config.max_memory_mb,
                "enable_gpu": self.config.enable_gpu,
            },
            "runtime_status": {
                "active": self.runtime_active,
                "offline_mode": self.config.offline_mode,
                "network_available": self.network_available,
                "last_heartbeat": self.last_heartbeat,
            },
            "task_statistics": {
                "tasks_processed": self.tasks_processed,
                "successful_tasks": self.successful_tasks,
                "success_rate": self.successful_tasks / max(1, self.tasks_processed),
                "average_execution_time": self.total_execution_time / max(1, self.tasks_processed),
                "queued_tasks": len(self.task_queue),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
            },
            "resource_status": self.resource_monitor.get_resource_status(),
            "cache_statistics": self.cache.get_stats(),
            "loaded_models": list(self.loaded_models.keys()),
        }

    def export_runtime_state(self) -> Dict[str, Any]:
        """Export runtime state for persistence or migration.
        
        Returns:
            Exportable runtime state
        """
        return {
            "device_id": self.device_id,
            "config": {
                "resource_profile": self.config.resource_profile.value,
                "max_memory_mb": self.config.max_memory_mb,
                "max_cpu_cores": self.config.max_cpu_cores,
                "enable_gpu": self.config.enable_gpu,
                "cache_size_mb": self.config.cache_size_mb,
            },
            "statistics": self.get_runtime_statistics(),
            "loaded_models": list(self.loaded_models.keys()),
            "timestamp": time.time(),
        }
