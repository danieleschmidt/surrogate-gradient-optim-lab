"""Distributed cluster management for large-scale surrogate optimization."""

import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from queue import Queue, Empty
import logging
import json
import socket
import hashlib

import jax
import jax.numpy as jnp
from jax import Array, pmap, device_put
import numpy as np

from ..observability.tracing import get_tracer
from ..monitoring.enhanced_logging import get_logger
from ..performance.gpu_acceleration import GPUManager, MultiGPUOptimizer


logger = get_logger()


class NodeStatus(Enum):
    """Status of a compute node."""
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


class TaskStatus(Enum):
    """Status of a distributed task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_AWARE = "resource_aware"
    ADAPTIVE = "adaptive"


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    hostname: str
    port: int
    status: NodeStatus
    capabilities: Dict[str, Any]
    current_load: float
    max_workers: int
    active_tasks: int
    total_completed: int
    total_failed: int
    last_heartbeat: datetime
    gpu_count: int
    memory_gb: float
    cpu_cores: int
    tags: List[str] = field(default_factory=list)


@dataclass
class Task:
    """Distributed task definition."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int
    timeout_seconds: int
    max_retries: int
    retry_count: int
    node_requirements: Dict[str, Any]
    status: TaskStatus
    assigned_node: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Any]
    error: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterMetrics:
    """Cluster-wide metrics."""
    total_nodes: int
    active_nodes: int
    failed_nodes: int
    total_tasks_pending: int
    total_tasks_running: int
    total_tasks_completed: int
    total_tasks_failed: int
    average_node_utilization: float
    cluster_throughput: float
    average_task_duration: float
    queue_wait_time: float
    timestamp: datetime


class ClusterManager:
    """Manages a distributed cluster for surrogate optimization workloads."""
    
    def __init__(
        self,
        cluster_name: str = "surrogate_cluster",
        load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE,
        heartbeat_interval: int = 30,
        task_timeout: int = 3600,
        max_retries: int = 3,
        enable_auto_scaling: bool = True,
    ):
        """Initialize cluster manager.
        
        Args:
            cluster_name: Name of the cluster
            load_balancing: Load balancing strategy
            heartbeat_interval: Heartbeat interval in seconds
            task_timeout: Default task timeout in seconds
            max_retries: Maximum task retries
            enable_auto_scaling: Enable automatic scaling
        """
        self.cluster_name = cluster_name
        self.load_balancing = load_balancing
        self.heartbeat_interval = heartbeat_interval
        self.task_timeout = task_timeout
        self.max_retries = max_retries
        self.enable_auto_scaling = enable_auto_scaling
        
        # Node management
        self.nodes: Dict[str, NodeInfo] = {}
        self.nodes_lock = threading.RLock()
        
        # Task management
        self.task_queue: Queue = Queue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.task_lock = threading.RLock()
        
        # Executor pools
        self.local_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cluster_mgr")
        self.compute_executor = ProcessPoolExecutor(max_workers=8)
        
        # Metrics and monitoring
        self.metrics_history: List[ClusterMetrics] = []
        self.metrics_lock = threading.Lock()
        
        # Control state
        self.is_running = False
        self.management_thread = None
        
        # Tracer
        self.tracer = get_tracer("cluster_manager")
        
        # Load balancing state
        self.round_robin_index = 0
        self.load_balancing_stats = {}
        
        logger.info(f"Cluster manager initialized: {cluster_name}")
    
    def start(self):
        """Start the cluster manager."""
        if self.is_running:
            logger.warning("Cluster manager already running")
            return
        
        self.is_running = True
        
        # Start management thread
        self.management_thread = threading.Thread(
            target=self._management_loop,
            name="cluster_management",
            daemon=True
        )
        self.management_thread.start()
        
        logger.info(f"Cluster manager started: {self.cluster_name}")
    
    def stop(self):
        """Stop the cluster manager."""
        if not self.is_running:
            return
        
        logger.info("Stopping cluster manager...")
        self.is_running = False
        
        # Wait for management thread
        if self.management_thread:
            self.management_thread.join(timeout=10)
        
        # Shutdown executors
        self.local_executor.shutdown(wait=True)
        self.compute_executor.shutdown(wait=True)
        
        logger.info("Cluster manager stopped")
    
    def register_node(
        self,
        hostname: str,
        port: int,
        capabilities: Dict[str, Any],
        max_workers: int = 4,
        tags: List[str] = None,
    ) -> str:
        """Register a new compute node.
        
        Args:
            hostname: Node hostname
            port: Node port
            capabilities: Node capabilities (CPU, GPU, memory, etc.)
            max_workers: Maximum concurrent workers
            tags: Optional node tags
            
        Returns:
            Node ID
        """
        node_id = hashlib.sha256(f"{hostname}:{port}".encode()).hexdigest()[:12]
        
        node_info = NodeInfo(
            node_id=node_id,
            hostname=hostname,
            port=port,
            status=NodeStatus.INITIALIZING,
            capabilities=capabilities,
            current_load=0.0,
            max_workers=max_workers,
            active_tasks=0,
            total_completed=0,
            total_failed=0,
            last_heartbeat=datetime.now(),
            gpu_count=capabilities.get("gpu_count", 0),
            memory_gb=capabilities.get("memory_gb", 8.0),
            cpu_cores=capabilities.get("cpu_cores", 4),
            tags=tags or [],
        )
        
        with self.nodes_lock:
            self.nodes[node_id] = node_info
        
        # Validate node connection
        self._validate_node_connection(node_id)
        
        logger.info(f"Registered node: {node_id} ({hostname}:{port})")
        return node_id
    
    def unregister_node(self, node_id: str):
        """Unregister a compute node."""
        with self.nodes_lock:
            if node_id in self.nodes:
                # Reassign active tasks
                self._reassign_node_tasks(node_id)
                del self.nodes[node_id]
                logger.info(f"Unregistered node: {node_id}")
    
    def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 1,
        timeout_seconds: Optional[int] = None,
        node_requirements: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a task to the cluster.
        
        Args:
            task_type: Type of task
            payload: Task payload data
            priority: Task priority (higher = more important)
            timeout_seconds: Task timeout
            node_requirements: Node requirements for task
            metadata: Optional task metadata
            
        Returns:
            Task ID
        """
        task_id = hashlib.sha256(
            f"{task_type}:{time.time()}:{id(payload)}".encode()
        ).hexdigest()[:16]
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout_seconds=timeout_seconds or self.task_timeout,
            max_retries=self.max_retries,
            retry_count=0,
            node_requirements=node_requirements or {},
            status=TaskStatus.PENDING,
            assigned_node=None,
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            result=None,
            error=None,
            metadata=metadata or {},
        )
        
        with self.task_lock:
            self.active_tasks[task_id] = task
        
        # Add to queue (priority queue would be better)
        self.task_queue.put((priority, task_id))
        
        logger.debug(f"Submitted task: {task_id} (type: {task_type}, priority: {priority})")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get task status."""
        with self.task_lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self.task_lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    self.completed_tasks[task_id] = task
                    del self.active_tasks[task_id]
                    logger.info(f"Cancelled task: {task_id}")
                    return True
                elif task.status == TaskStatus.RUNNING:
                    # Would need to send cancellation signal to node
                    task.status = TaskStatus.CANCELLED
                    logger.info(f"Marked task for cancellation: {task_id}")
                    return True
        return False
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        with self.nodes_lock:
            nodes_by_status = {}
            for status in NodeStatus:
                nodes_by_status[status.value] = [
                    node for node in self.nodes.values() if node.status == status
                ]
        
        with self.task_lock:
            tasks_by_status = {}
            for status in TaskStatus:
                tasks_by_status[status.value] = [
                    task for task in self.active_tasks.values() if task.status == status
                ]
            
            # Add completed tasks
            tasks_by_status[TaskStatus.COMPLETED.value].extend(self.completed_tasks.values())
        
        current_metrics = self._calculate_cluster_metrics()
        
        return {
            "cluster_name": self.cluster_name,
            "is_running": self.is_running,
            "total_nodes": len(self.nodes),
            "nodes_by_status": {k: len(v) for k, v in nodes_by_status.items()},
            "total_tasks_pending": len([t for t in self.active_tasks.values() 
                                      if t.status == TaskStatus.PENDING]),
            "total_tasks_running": len([t for t in self.active_tasks.values() 
                                      if t.status == TaskStatus.RUNNING]),
            "total_tasks_completed": len(self.completed_tasks),
            "queue_size": self.task_queue.qsize(),
            "load_balancing_strategy": self.load_balancing.value,
            "current_metrics": asdict(current_metrics) if current_metrics else None,
        }
    
    def _management_loop(self):
        """Main management loop."""
        logger.info("Starting cluster management loop")
        
        while self.is_running:
            try:
                # Process pending tasks
                self._process_pending_tasks()
                
                # Check node health
                self._check_node_health()
                
                # Handle task timeouts
                self._handle_task_timeouts()
                
                # Update metrics
                self._update_cluster_metrics()
                
                # Auto-scaling
                if self.enable_auto_scaling:
                    self._auto_scale_cluster()
                
                time.sleep(5)  # Management loop interval
                
            except Exception as e:
                logger.error(f"Error in management loop: {e}")
                time.sleep(10)  # Back off on error
        
        logger.info("Cluster management loop stopped")
    
    def _process_pending_tasks(self):
        """Process pending tasks and assign to nodes."""
        try:
            while not self.task_queue.empty():
                try:
                    priority, task_id = self.task_queue.get_nowait()
                except Empty:
                    break
                
                with self.task_lock:
                    if task_id not in self.active_tasks:
                        continue
                    
                    task = self.active_tasks[task_id]
                    if task.status != TaskStatus.PENDING:
                        continue
                
                # Find suitable node
                suitable_node = self._find_suitable_node(task)
                if suitable_node:
                    self._assign_task_to_node(task, suitable_node)
                else:
                    # Put back in queue if no suitable node
                    self.task_queue.put((priority, task_id))
                    break  # Avoid busy loop
        
        except Exception as e:
            logger.error(f"Error processing pending tasks: {e}")
    
    def _find_suitable_node(self, task: Task) -> Optional[str]:
        """Find a suitable node for the task."""
        with self.nodes_lock:
            available_nodes = [
                node for node in self.nodes.values()
                if node.status == NodeStatus.IDLE and 
                   node.active_tasks < node.max_workers and
                   self._node_meets_requirements(node, task.node_requirements)
            ]
        
        if not available_nodes:
            return None
        
        # Apply load balancing strategy
        if self.load_balancing == LoadBalancingStrategy.ROUND_ROBIN:
            self.round_robin_index = (self.round_robin_index + 1) % len(available_nodes)
            return available_nodes[self.round_robin_index].node_id
        
        elif self.load_balancing == LoadBalancingStrategy.LEAST_LOADED:
            least_loaded = min(available_nodes, key=lambda n: n.current_load)
            return least_loaded.node_id
        
        elif self.load_balancing == LoadBalancingStrategy.RESOURCE_AWARE:
            # Score nodes based on resource availability
            def resource_score(node):
                cpu_score = 1.0 - (node.active_tasks / node.max_workers)
                memory_score = 1.0 - node.current_load  # Simplified
                gpu_score = 1.0 if node.gpu_count > 0 else 0.5
                return cpu_score * 0.4 + memory_score * 0.4 + gpu_score * 0.2
            
            best_node = max(available_nodes, key=resource_score)
            return best_node.node_id
        
        else:
            # Default to first available
            return available_nodes[0].node_id
    
    def _node_meets_requirements(self, node: NodeInfo, requirements: Dict[str, Any]) -> bool:
        """Check if node meets task requirements."""
        if not requirements:
            return True
        
        # Check GPU requirements
        if requirements.get("requires_gpu", False) and node.gpu_count == 0:
            return False
        
        # Check memory requirements
        min_memory = requirements.get("min_memory_gb", 0)
        if node.memory_gb < min_memory:
            return False
        
        # Check CPU requirements
        min_cores = requirements.get("min_cpu_cores", 0)
        if node.cpu_cores < min_cores:
            return False
        
        # Check tags
        required_tags = requirements.get("required_tags", [])
        if required_tags and not set(required_tags).issubset(set(node.tags)):
            return False
        
        # Check capabilities
        required_capabilities = requirements.get("capabilities", {})
        for capability, required_value in required_capabilities.items():
            if capability not in node.capabilities:
                return False
            if node.capabilities[capability] < required_value:
                return False
        
        return True
    
    def _assign_task_to_node(self, task: Task, node_id: str):
        """Assign task to a specific node."""
        with self.nodes_lock:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            if node.active_tasks >= node.max_workers:
                return False
        
        # Update task status
        task.status = TaskStatus.ASSIGNED
        task.assigned_node = node_id
        task.started_at = datetime.now()
        
        # Update node
        with self.nodes_lock:
            node.active_tasks += 1
            node.current_load = min(1.0, node.active_tasks / node.max_workers)
            if node.status == NodeStatus.IDLE:
                node.status = NodeStatus.BUSY
        
        # Submit to execution (simplified - would use actual node communication)
        future = self.compute_executor.submit(self._execute_task, task)
        
        logger.debug(f"Assigned task {task.task_id} to node {node_id}")
        return True
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a task (simplified implementation)."""
        with self.tracer.trace("task_execution") as span:
            span.set_attribute("task.id", task.task_id)
            span.set_attribute("task.type", task.task_type)
            span.set_attribute("task.node", task.assigned_node)
            
            try:
                task.status = TaskStatus.RUNNING
                
                # Simulate task execution based on type
                start_time = time.time()
                
                if task.task_type == "surrogate_training":
                    result = self._execute_surrogate_training(task)
                elif task.task_type == "optimization":
                    result = self._execute_optimization(task)
                elif task.task_type == "benchmark_evaluation":
                    result = self._execute_benchmark_evaluation(task)
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")
                
                execution_time = time.time() - start_time
                
                # Update task
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                task.metadata["execution_time"] = execution_time
                
                # Update node
                if task.assigned_node:
                    self._update_node_after_task(task.assigned_node, success=True)
                
                # Move to completed tasks
                with self.task_lock:
                    if task.task_id in self.active_tasks:
                        self.completed_tasks[task.task_id] = task
                        del self.active_tasks[task.task_id]
                
                logger.debug(f"Task {task.task_id} completed successfully ({execution_time:.2f}s)")
                return result
                
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = str(e)
                
                # Update node
                if task.assigned_node:
                    self._update_node_after_task(task.assigned_node, success=False)
                
                # Handle retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.RETRY
                    task.assigned_node = None
                    self.task_queue.put((task.priority, task.task_id))
                    logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
                else:
                    # Move to completed tasks
                    with self.task_lock:
                        if task.task_id in self.active_tasks:
                            self.completed_tasks[task.task_id] = task
                            del self.active_tasks[task.task_id]
                
                raise
    
    def _execute_surrogate_training(self, task: Task) -> Dict[str, Any]:
        """Execute surrogate training task."""
        payload = task.payload
        
        # Simulate training
        time.sleep(payload.get("training_time", 2.0))  # Simulate training duration
        
        return {
            "model_id": f"model_{task.task_id}",
            "training_loss": np.random.uniform(0.01, 0.1),
            "training_time": payload.get("training_time", 2.0),
            "model_accuracy": np.random.uniform(0.8, 0.95),
        }
    
    def _execute_optimization(self, task: Task) -> Dict[str, Any]:
        """Execute optimization task."""
        payload = task.payload
        
        # Simulate optimization
        n_iterations = payload.get("n_iterations", 100)
        time.sleep(n_iterations * 0.01)  # Simulate optimization time
        
        # Generate fake convergence history
        initial_value = np.random.uniform(10, 100)
        convergence_history = []
        current_value = initial_value
        
        for i in range(n_iterations):
            current_value = current_value * (0.99 + np.random.uniform(-0.01, 0.01))
            convergence_history.append(float(current_value))
        
        return {
            "final_value": float(current_value),
            "initial_value": float(initial_value),
            "convergence_history": convergence_history,
            "n_iterations": n_iterations,
            "function_evaluations": n_iterations,
            "success": True,
        }
    
    def _execute_benchmark_evaluation(self, task: Task) -> Dict[str, Any]:
        """Execute benchmark evaluation task."""
        payload = task.payload
        
        # Simulate benchmark evaluation
        benchmark_name = payload.get("benchmark_name", "unknown")
        dimension = payload.get("dimension", 2)
        
        time.sleep(0.5)  # Simulate evaluation time
        
        return {
            "benchmark_name": benchmark_name,
            "dimension": dimension,
            "evaluation_result": np.random.uniform(-100, 0),
            "evaluation_time": 0.5,
        }
    
    def _update_node_after_task(self, node_id: str, success: bool):
        """Update node statistics after task completion."""
        with self.nodes_lock:
            if node_id not in self.nodes:
                return
            
            node = self.nodes[node_id]
            node.active_tasks = max(0, node.active_tasks - 1)
            node.current_load = node.active_tasks / node.max_workers if node.max_workers > 0 else 0.0
            
            if success:
                node.total_completed += 1
            else:
                node.total_failed += 1
            
            # Update status
            if node.active_tasks == 0:
                node.status = NodeStatus.IDLE
    
    def _check_node_health(self):
        """Check health of all registered nodes."""
        current_time = datetime.now()
        heartbeat_timeout = timedelta(seconds=self.heartbeat_interval * 3)
        
        with self.nodes_lock:
            for node_id, node in self.nodes.items():
                if current_time - node.last_heartbeat > heartbeat_timeout:
                    if node.status != NodeStatus.OFFLINE:
                        logger.warning(f"Node {node_id} appears offline (last heartbeat: {node.last_heartbeat})")
                        node.status = NodeStatus.OFFLINE
                        # Reassign tasks from offline node
                        self._reassign_node_tasks(node_id)
    
    def _reassign_node_tasks(self, node_id: str):
        """Reassign tasks from a failed or offline node."""
        tasks_to_reassign = []
        
        with self.task_lock:
            for task in self.active_tasks.values():
                if task.assigned_node == node_id and task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
                    tasks_to_reassign.append(task)
        
        for task in tasks_to_reassign:
            logger.info(f"Reassigning task {task.task_id} from failed node {node_id}")
            task.status = TaskStatus.PENDING
            task.assigned_node = None
            self.task_queue.put((task.priority, task.task_id))
    
    def _handle_task_timeouts(self):
        """Handle timed out tasks."""
        current_time = datetime.now()
        
        with self.task_lock:
            timed_out_tasks = []
            for task in self.active_tasks.values():
                if task.started_at and task.status == TaskStatus.RUNNING:
                    elapsed = (current_time - task.started_at).total_seconds()
                    if elapsed > task.timeout_seconds:
                        timed_out_tasks.append(task)
            
            for task in timed_out_tasks:
                logger.warning(f"Task {task.task_id} timed out after {task.timeout_seconds}s")
                task.status = TaskStatus.FAILED
                task.error = f"Task timed out after {task.timeout_seconds} seconds"
                task.completed_at = current_time
                
                # Update node
                if task.assigned_node:
                    self._update_node_after_task(task.assigned_node, success=False)
                
                # Move to completed tasks
                self.completed_tasks[task.task_id] = task
                del self.active_tasks[task.task_id]
    
    def _calculate_cluster_metrics(self) -> ClusterMetrics:
        """Calculate current cluster metrics."""
        current_time = datetime.now()
        
        with self.nodes_lock:
            total_nodes = len(self.nodes)
            active_nodes = len([n for n in self.nodes.values() if n.status in [NodeStatus.IDLE, NodeStatus.BUSY]])
            failed_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.FAILED])
            
            total_utilization = sum(node.current_load for node in self.nodes.values())
            avg_utilization = total_utilization / total_nodes if total_nodes > 0 else 0.0
        
        with self.task_lock:
            pending_tasks = len([t for t in self.active_tasks.values() if t.status == TaskStatus.PENDING])
            running_tasks = len([t for t in self.active_tasks.values() if t.status == TaskStatus.RUNNING])
            completed_tasks = len(self.completed_tasks)
            failed_tasks = len([t for t in self.completed_tasks.values() if t.status == TaskStatus.FAILED])
            
            # Calculate average task duration
            recent_completed = [
                t for t in self.completed_tasks.values()
                if t.completed_at and t.started_at and (current_time - t.completed_at).total_seconds() < 3600
            ]
            
            if recent_completed:
                avg_duration = sum(
                    (t.completed_at - t.started_at).total_seconds() for t in recent_completed
                ) / len(recent_completed)
            else:
                avg_duration = 0.0
        
        # Calculate throughput (tasks completed per hour)
        recent_hour_tasks = [
            t for t in self.completed_tasks.values()
            if t.completed_at and (current_time - t.completed_at).total_seconds() < 3600
        ]
        throughput = len(recent_hour_tasks)
        
        return ClusterMetrics(
            total_nodes=total_nodes,
            active_nodes=active_nodes,
            failed_nodes=failed_nodes,
            total_tasks_pending=pending_tasks,
            total_tasks_running=running_tasks,
            total_tasks_completed=completed_tasks,
            total_tasks_failed=failed_tasks,
            average_node_utilization=avg_utilization,
            cluster_throughput=throughput,
            average_task_duration=avg_duration,
            queue_wait_time=0.0,  # Would calculate from queue metrics
            timestamp=current_time,
        )
    
    def _update_cluster_metrics(self):
        """Update cluster metrics history."""
        try:
            current_metrics = self._calculate_cluster_metrics()
            
            with self.metrics_lock:
                self.metrics_history.append(current_metrics)
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
            
        except Exception as e:
            logger.error(f"Error updating cluster metrics: {e}")
    
    def _auto_scale_cluster(self):
        """Auto-scale cluster based on workload."""
        if not self.enable_auto_scaling:
            return
        
        try:
            # Simple auto-scaling logic
            queue_size = self.task_queue.qsize()
            active_nodes = len([n for n in self.nodes.values() 
                               if n.status in [NodeStatus.IDLE, NodeStatus.BUSY]])
            
            # Scale up if queue is large relative to capacity
            if queue_size > active_nodes * 2:
                logger.info("High queue size detected - consider scaling up")
                # Would trigger node provisioning in real implementation
            
            # Scale down if nodes are idle
            idle_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.IDLE]
            if len(idle_nodes) > active_nodes * 0.3 and queue_size == 0:
                logger.info("Many idle nodes detected - consider scaling down")
                # Would trigger node deprovisioning in real implementation
            
        except Exception as e:
            logger.error(f"Error in auto-scaling: {e}")
    
    def _validate_node_connection(self, node_id: str):
        """Validate connection to a node."""
        # Simplified validation - would implement actual node communication
        with self.nodes_lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = NodeStatus.IDLE
                self.nodes[node_id].last_heartbeat = datetime.now()
    
    def get_metrics_history(self, hours: int = 1) -> List[ClusterMetrics]:
        """Get metrics history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.metrics_lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def submit_batch_tasks(
        self,
        task_configs: List[Dict[str, Any]],
        batch_priority: int = 1,
    ) -> List[str]:
        """Submit a batch of tasks."""
        task_ids = []
        
        for config in task_configs:
            task_id = self.submit_task(
                task_type=config["task_type"],
                payload=config["payload"],
                priority=config.get("priority", batch_priority),
                timeout_seconds=config.get("timeout_seconds"),
                node_requirements=config.get("node_requirements"),
                metadata=config.get("metadata"),
            )
            task_ids.append(task_id)
        
        logger.info(f"Submitted batch of {len(task_ids)} tasks")
        return task_ids
    
    def wait_for_tasks(
        self,
        task_ids: List[str],
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Task]:
        """Wait for tasks to complete."""
        start_time = time.time()
        completed_tasks = {}
        
        while len(completed_tasks) < len(task_ids):
            for task_id in task_ids:
                if task_id in completed_tasks:
                    continue
                
                task = self.get_task_status(task_id)
                if task and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    completed_tasks[task_id] = task
            
            # Check timeout
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                logger.warning(f"Timeout waiting for tasks: {len(completed_tasks)}/{len(task_ids)} completed")
                break
            
            time.sleep(1)  # Polling interval
        
        return completed_tasks


# Convenience functions
def create_local_cluster(
    n_workers: int = 4,
    enable_gpu: bool = True,
) -> ClusterManager:
    """Create a local cluster for testing."""
    cluster = ClusterManager(
        cluster_name="local_test_cluster",
        load_balancing=LoadBalancingStrategy.LEAST_LOADED,
    )
    cluster.start()
    
    # Register local nodes
    for i in range(n_workers):
        gpu_count = 1 if enable_gpu and i == 0 else 0  # Only first worker has GPU
        capabilities = {
            "cpu_cores": 4,
            "memory_gb": 8.0,
            "gpu_count": gpu_count,
            "framework_support": ["jax", "numpy", "scipy"],
        }
        
        cluster.register_node(
            hostname="localhost",
            port=8000 + i,
            capabilities=capabilities,
            max_workers=2,
            tags=["local", "test"] + (["gpu"] if gpu_count > 0 else []),
        )
    
    return cluster