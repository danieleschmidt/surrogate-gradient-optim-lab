"""Federated Learning Framework for collaborative surrogate optimization.

This module enables multiple optimization nodes to collaboratively learn and
share knowledge while preserving privacy and data locality.
"""

import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import logging
import pickle
import threading
import time
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from jax import Array
import jax.numpy as jnp
import numpy as np


@dataclass
class FederatedNode:
    """Represents a node in the federated learning network."""
    node_id: str
    public_key: str
    last_seen: float
    trust_score: float = 1.0
    contribution_score: float = 0.0
    knowledge_quality: float = 0.0
    active: bool = True


@dataclass
class KnowledgePacket:
    """Package of knowledge to be shared between nodes."""
    source_node: str
    timestamp: float
    knowledge_type: str  # "model_weights", "hyperparameters", "patterns"
    encrypted_data: bytes
    metadata: Dict[str, Any]
    signature: str
    quality_score: float = 0.0


@dataclass
class FederationConfiguration:
    """Configuration for federated learning system."""
    max_nodes: int = 100
    min_trust_score: float = 0.3
    knowledge_retention_days: int = 30
    max_knowledge_packets: int = 1000
    aggregation_frequency_minutes: int = 60
    privacy_level: str = "high"  # "low", "medium", "high"
    differential_privacy_epsilon: float = 1.0


class SecureCommunication:
    """Handles secure communication between federated nodes."""

    def __init__(self, password: str):
        """Initialize secure communication with password.
        
        Args:
            password: Password for encryption key derivation
        """
        # Generate encryption key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"salt_",  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.cipher = Fernet(key)

    def encrypt_knowledge(self, data: Dict[str, Any]) -> bytes:
        """Encrypt knowledge data.
        
        Args:
            data: Knowledge data to encrypt
            
        Returns:
            Encrypted data bytes
        """
        serialized = pickle.dumps(data)
        return self.cipher.encrypt(serialized)

    def decrypt_knowledge(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt knowledge data.
        
        Args:
            encrypted_data: Encrypted data bytes
            
        Returns:
            Decrypted knowledge data
        """
        decrypted = self.cipher.decrypt(encrypted_data)
        return pickle.loads(decrypted)

    def sign_packet(self, packet_data: bytes) -> str:
        """Create signature for knowledge packet.
        
        Args:
            packet_data: Data to sign
            
        Returns:
            Signature string
        """
        return hashlib.sha256(packet_data).hexdigest()

    def verify_signature(self, packet_data: bytes, signature: str) -> bool:
        """Verify knowledge packet signature.
        
        Args:
            packet_data: Data to verify
            signature: Signature to verify against
            
        Returns:
            True if signature is valid
        """
        expected_signature = hashlib.sha256(packet_data).hexdigest()
        return expected_signature == signature


class DifferentialPrivacy:
    """Implements differential privacy for knowledge sharing."""

    def __init__(self, epsilon: float = 1.0):
        """Initialize differential privacy mechanism.
        
        Args:
            epsilon: Privacy budget parameter
        """
        self.epsilon = epsilon
        self.global_sensitivity = 1.0  # Depends on the specific application

    def add_laplace_noise(self, data: Array, sensitivity: Optional[float] = None) -> Array:
        """Add Laplace noise for differential privacy.
        
        Args:
            data: Original data
            sensitivity: Sensitivity of the query (uses global if None)
            
        Returns:
            Data with added noise
        """
        if sensitivity is None:
            sensitivity = self.global_sensitivity

        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, size=data.shape)

        return data + noise

    def add_gaussian_noise(self, data: Array, sensitivity: Optional[float] = None) -> Array:
        """Add Gaussian noise for differential privacy.
        
        Args:
            data: Original data
            sensitivity: Sensitivity of the query (uses global if None)
            
        Returns:
            Data with added noise
        """
        if sensitivity is None:
            sensitivity = self.global_sensitivity

        # For Gaussian mechanism, we need to compute sigma
        delta = 1e-5  # Small probability
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / self.epsilon

        noise = np.random.normal(0, sigma, size=data.shape)
        return data + noise

    def privatize_gradients(self, gradients: Array) -> Array:
        """Apply privacy to gradient information.
        
        Args:
            gradients: Original gradients
            
        Returns:
            Privatized gradients
        """
        # Clip gradients to bound sensitivity
        clipped_gradients = jnp.clip(gradients, -1.0, 1.0)

        # Add noise
        return self.add_laplace_noise(clipped_gradients, sensitivity=2.0)


class FederatedLearningFramework:
    """Framework for federated surrogate optimization learning.
    
    This framework enables multiple optimization nodes to collaboratively
    learn and improve their performance while maintaining data privacy.
    """

    def __init__(
        self,
        node_id: str,
        config: Optional[FederationConfiguration] = None,
        password: str = "default_federation_password"
    ):
        """Initialize federated learning framework.
        
        Args:
            node_id: Unique identifier for this node
            config: Federation configuration
            password: Password for secure communication
        """
        self.node_id = node_id
        self.config = config or FederationConfiguration()

        # Security and privacy
        self.secure_comm = SecureCommunication(password)
        self.privacy = DifferentialPrivacy(self.config.differential_privacy_epsilon)

        # Network state
        self.known_nodes: Dict[str, FederatedNode] = {}
        self.knowledge_store: Dict[str, KnowledgePacket] = {}
        self.local_knowledge: Dict[str, Any] = {}

        # Aggregation state
        self.aggregation_active = False
        self.aggregation_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Performance tracking
        self.collaboration_metrics: Dict[str, List[float]] = defaultdict(list)
        self.knowledge_quality_history: List[float] = []

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Logger
        self.logger = logging.getLogger(__name__)

        # Initialize local node
        self._initialize_local_node()

    def _initialize_local_node(self) -> None:
        """Initialize the local federated node."""
        self.local_node = FederatedNode(
            node_id=self.node_id,
            public_key="",  # Would be actual public key in production
            last_seen=time.time(),
            trust_score=1.0,
            contribution_score=0.0,
            knowledge_quality=0.0,
            active=True
        )

    def start_federation(self) -> None:
        """Start the federated learning system."""
        self.aggregation_active = True

        # Start background aggregation thread
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self.aggregation_thread.start()

        self.logger.info(f"Federated learning started for node {self.node_id}")

    def stop_federation(self) -> None:
        """Stop the federated learning system."""
        self.aggregation_active = False
        self.stop_event.set()

        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=10)

        self.executor.shutdown(wait=False)

        self.logger.info(f"Federated learning stopped for node {self.node_id}")

    def register_node(self, node: FederatedNode) -> bool:
        """Register a new node in the federation.
        
        Args:
            node: Node to register
            
        Returns:
            True if registration successful
        """
        if len(self.known_nodes) >= self.config.max_nodes:
            self.logger.warning(f"Cannot register node {node.node_id}: federation full")
            return False

        if node.trust_score < self.config.min_trust_score:
            self.logger.warning(f"Cannot register node {node.node_id}: insufficient trust score")
            return False

        self.known_nodes[node.node_id] = node
        self.logger.info(f"Registered node {node.node_id} in federation")

        return True

    def share_knowledge(
        self,
        knowledge_type: str,
        knowledge_data: Dict[str, Any],
        quality_score: Optional[float] = None
    ) -> bool:
        """Share knowledge with the federation.
        
        Args:
            knowledge_type: Type of knowledge being shared
            knowledge_data: Knowledge data to share
            quality_score: Optional quality score for the knowledge
            
        Returns:
            True if sharing successful
        """
        try:
            # Apply privacy protection if needed
            protected_data = self._apply_privacy_protection(knowledge_data, knowledge_type)

            # Encrypt the knowledge
            encrypted_data = self.secure_comm.encrypt_knowledge(protected_data)

            # Create knowledge packet
            packet = KnowledgePacket(
                source_node=self.node_id,
                timestamp=time.time(),
                knowledge_type=knowledge_type,
                encrypted_data=encrypted_data,
                metadata={
                    "data_size": len(str(knowledge_data)),
                    "privacy_protected": self.config.privacy_level != "low",
                    "node_trust": self.local_node.trust_score
                },
                signature=self.secure_comm.sign_packet(encrypted_data),
                quality_score=quality_score or self._estimate_knowledge_quality(knowledge_data)
            )

            # Store in local knowledge store
            packet_id = self._generate_packet_id(packet)
            self.knowledge_store[packet_id] = packet

            # Update local contribution score
            self.local_node.contribution_score += 1.0

            self.logger.info(f"Shared {knowledge_type} knowledge with quality score {packet.quality_score:.3f}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to share knowledge: {e}")
            return False

    def _apply_privacy_protection(
        self,
        data: Dict[str, Any],
        knowledge_type: str
    ) -> Dict[str, Any]:
        """Apply privacy protection to knowledge data.
        
        Args:
            data: Original knowledge data
            knowledge_type: Type of knowledge
            
        Returns:
            Privacy-protected knowledge data
        """
        if self.config.privacy_level == "low":
            return data

        protected_data = data.copy()

        # Apply differential privacy to numerical data
        for key, value in data.items():
            if isinstance(value, (np.ndarray, list)) and knowledge_type == "gradients":
                # Protect gradients with differential privacy
                value_array = jnp.array(value)
                protected_value = self.privacy.privatize_gradients(value_array)
                protected_data[key] = protected_value.tolist()

            elif isinstance(value, (np.ndarray, list)) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    # Add noise to numerical arrays
                    value_array = jnp.array(value)
                    protected_value = self.privacy.add_laplace_noise(value_array)
                    protected_data[key] = protected_value.tolist()

        return protected_data

    def _estimate_knowledge_quality(self, knowledge_data: Dict[str, Any]) -> float:
        """Estimate the quality of knowledge data.
        
        Args:
            knowledge_data: Knowledge data to evaluate
            
        Returns:
            Quality score between 0 and 1
        """
        # Simple heuristic for knowledge quality
        quality_factors = []

        # Data completeness
        if knowledge_data:
            quality_factors.append(0.5)

        # Data diversity (based on number of keys)
        num_keys = len(knowledge_data.keys())
        diversity_score = min(1.0, num_keys / 10.0)
        quality_factors.append(diversity_score * 0.3)

        # Data size (larger datasets generally more valuable)
        total_size = sum(len(str(v)) for v in knowledge_data.values())
        size_score = min(1.0, total_size / 10000.0)  # Normalize by 10k characters
        quality_factors.append(size_score * 0.2)

        return sum(quality_factors)

    def _generate_packet_id(self, packet: KnowledgePacket) -> str:
        """Generate unique ID for knowledge packet.
        
        Args:
            packet: Knowledge packet
            
        Returns:
            Unique packet ID
        """
        content = f"{packet.source_node}_{packet.timestamp}_{packet.knowledge_type}"
        return hashlib.md5(content.encode()).hexdigest()

    def aggregate_knowledge(self, knowledge_type: str) -> Optional[Dict[str, Any]]:
        """Aggregate knowledge from all federation nodes.
        
        Args:
            knowledge_type: Type of knowledge to aggregate
            
        Returns:
            Aggregated knowledge or None if not enough data
        """
        # Collect relevant knowledge packets
        relevant_packets = [
            packet for packet in self.knowledge_store.values()
            if (packet.knowledge_type == knowledge_type and
                self._is_packet_valid(packet) and
                self._is_node_trusted(packet.source_node))
        ]

        if len(relevant_packets) < 2:
            self.logger.debug(f"Insufficient packets for {knowledge_type} aggregation")
            return None

        # Decrypt and aggregate knowledge
        aggregated_data = self._perform_aggregation(relevant_packets)

        # Update collaboration metrics
        self.collaboration_metrics[knowledge_type].append(len(relevant_packets))

        # Estimate quality of aggregated knowledge
        aggregated_quality = np.mean([p.quality_score for p in relevant_packets])
        self.knowledge_quality_history.append(aggregated_quality)

        self.logger.info(f"Aggregated {knowledge_type} knowledge from {len(relevant_packets)} nodes")

        return aggregated_data

    def _is_packet_valid(self, packet: KnowledgePacket) -> bool:
        """Check if knowledge packet is valid.
        
        Args:
            packet: Packet to validate
            
        Returns:
            True if packet is valid
        """
        # Check timestamp (not too old)
        age_days = (time.time() - packet.timestamp) / (24 * 3600)
        if age_days > self.config.knowledge_retention_days:
            return False

        # Verify signature
        if not self.secure_comm.verify_signature(packet.encrypted_data, packet.signature):
            return False

        # Check quality threshold
        if packet.quality_score < 0.1:  # Minimum quality threshold
            return False

        return True

    def _is_node_trusted(self, node_id: str) -> bool:
        """Check if a node is trusted.
        
        Args:
            node_id: ID of node to check
            
        Returns:
            True if node is trusted
        """
        if node_id == self.node_id:
            return True

        node = self.known_nodes.get(node_id)
        if node is None:
            return False

        return node.trust_score >= self.config.min_trust_score and node.active

    def _perform_aggregation(self, packets: List[KnowledgePacket]) -> Dict[str, Any]:
        """Perform knowledge aggregation from multiple packets.
        
        Args:
            packets: List of knowledge packets to aggregate
            
        Returns:
            Aggregated knowledge data
        """
        if not packets:
            return {}

        # Decrypt all packets
        decrypted_data = []
        weights = []

        for packet in packets:
            try:
                data = self.secure_comm.decrypt_knowledge(packet.encrypted_data)
                decrypted_data.append(data)

                # Weight by quality and trust
                node_trust = self.known_nodes.get(packet.source_node, self.local_node).trust_score
                weight = packet.quality_score * node_trust
                weights.append(weight)

            except Exception as e:
                self.logger.warning(f"Failed to decrypt packet from {packet.source_node}: {e}")
                continue

        if not decrypted_data:
            return {}

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)

        # Aggregate data based on type
        return self._weighted_aggregation(decrypted_data, weights)

    def _weighted_aggregation(
        self,
        data_list: List[Dict[str, Any]],
        weights: List[float]
    ) -> Dict[str, Any]:
        """Perform weighted aggregation of knowledge data.
        
        Args:
            data_list: List of knowledge data dictionaries
            weights: Weights for each data item
            
        Returns:
            Aggregated data
        """
        if not data_list:
            return {}

        aggregated = {}

        # Find common keys across all data
        all_keys = set()
        for data in data_list:
            all_keys.update(data.keys())

        for key in all_keys:
            values = []
            valid_weights = []

            # Collect values for this key
            for i, data in enumerate(data_list):
                if key in data:
                    values.append(data[key])
                    valid_weights.append(weights[i])

            if not values:
                continue

            # Aggregate based on data type
            if isinstance(values[0], (int, float)):
                # Weighted average for numerical values
                aggregated[key] = sum(v * w for v, w in zip(values, valid_weights))

            elif isinstance(values[0], (list, np.ndarray)):
                # Weighted average for arrays
                arrays = [jnp.array(v) for v in values]
                if all(arr.shape == arrays[0].shape for arr in arrays):
                    weighted_sum = sum(arr * w for arr, w in zip(arrays, valid_weights))
                    aggregated[key] = weighted_sum.tolist()
                else:
                    # Use majority voting for differently shaped arrays
                    aggregated[key] = values[valid_weights.index(max(valid_weights))]

            elif isinstance(values[0], dict):
                # Recursively aggregate dictionaries
                nested_aggregated = self._weighted_aggregation(values, valid_weights)
                aggregated[key] = nested_aggregated

            else:
                # Use weighted voting for other types
                value_weights = defaultdict(float)
                for v, w in zip(values, valid_weights):
                    value_weights[str(v)] += w

                best_value = max(value_weights.items(), key=lambda x: x[1])
                aggregated[key] = eval(best_value[0]) if best_value[0].replace(".", "").replace("-", "").isdigit() else best_value[0]

        return aggregated

    def _aggregation_loop(self) -> None:
        """Main loop for periodic knowledge aggregation."""
        while self.aggregation_active and not self.stop_event.is_set():
            try:
                # Wait for aggregation frequency
                if self.stop_event.wait(self.config.aggregation_frequency_minutes * 60):
                    break

                # Perform aggregation for different knowledge types
                knowledge_types = set(packet.knowledge_type for packet in self.knowledge_store.values())

                for knowledge_type in knowledge_types:
                    aggregated = self.aggregate_knowledge(knowledge_type)
                    if aggregated:
                        # Store aggregated knowledge locally
                        self.local_knowledge[f"aggregated_{knowledge_type}"] = {
                            "data": aggregated,
                            "timestamp": time.time(),
                            "sources": len(self.knowledge_store)
                        }

                # Clean up old knowledge
                self._cleanup_old_knowledge()

                # Update node trust scores
                self._update_trust_scores()

            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {e}")

    def _cleanup_old_knowledge(self) -> None:
        """Clean up old knowledge packets."""
        current_time = time.time()
        retention_seconds = self.config.knowledge_retention_days * 24 * 3600

        old_packets = [
            packet_id for packet_id, packet in self.knowledge_store.items()
            if (current_time - packet.timestamp) > retention_seconds
        ]

        for packet_id in old_packets:
            del self.knowledge_store[packet_id]

        if old_packets:
            self.logger.debug(f"Cleaned up {len(old_packets)} old knowledge packets")

    def _update_trust_scores(self) -> None:
        """Update trust scores for federated nodes."""
        for node_id, node in self.known_nodes.items():
            # Simple trust update based on contribution and quality
            recent_contributions = sum(
                1 for packet in self.knowledge_store.values()
                if (packet.source_node == node_id and
                    (time.time() - packet.timestamp) < 7 * 24 * 3600)  # Last 7 days
            )

            recent_quality = np.mean([
                packet.quality_score for packet in self.knowledge_store.values()
                if (packet.source_node == node_id and
                    (time.time() - packet.timestamp) < 7 * 24 * 3600)
            ]) if recent_contributions > 0 else 0.5

            # Update trust score (simple exponential moving average)
            contribution_factor = min(1.0, recent_contributions / 10.0)  # Normalize by 10 contributions
            quality_factor = recent_quality

            new_trust = 0.8 * node.trust_score + 0.2 * (0.5 * contribution_factor + 0.5 * quality_factor)
            node.trust_score = max(0.1, min(1.0, new_trust))  # Bound between 0.1 and 1.0

    def get_local_knowledge(self, knowledge_type: Optional[str] = None) -> Dict[str, Any]:
        """Get locally aggregated knowledge.
        
        Args:
            knowledge_type: Optional filter for specific knowledge type
            
        Returns:
            Local knowledge data
        """
        if knowledge_type is None:
            return self.local_knowledge.copy()

        aggregated_key = f"aggregated_{knowledge_type}"
        return self.local_knowledge.get(aggregated_key, {})

    def get_federation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive federation statistics.
        
        Returns:
            Dictionary containing federation metrics
        """
        active_nodes = sum(1 for node in self.known_nodes.values() if node.active)
        trusted_nodes = sum(1 for node in self.known_nodes.values()
                          if node.trust_score >= self.config.min_trust_score)

        return {
            "node_id": self.node_id,
            "total_nodes": len(self.known_nodes),
            "active_nodes": active_nodes,
            "trusted_nodes": trusted_nodes,
            "knowledge_packets": len(self.knowledge_store),
            "local_knowledge_items": len(self.local_knowledge),
            "average_trust_score": np.mean([node.trust_score for node in self.known_nodes.values()])
                                  if self.known_nodes else 0.0,
            "local_contribution_score": self.local_node.contribution_score,
            "average_knowledge_quality": np.mean(self.knowledge_quality_history)
                                        if self.knowledge_quality_history else 0.0,
            "collaboration_activity": {
                k: len(v) for k, v in self.collaboration_metrics.items()
            },
            "federation_active": self.aggregation_active,
        }

    def export_federation_state(self) -> Dict[str, Any]:
        """Export federation state for persistence.
        
        Returns:
            Dictionary containing complete federation state
        """
        return {
            "node_id": self.node_id,
            "config": {
                "max_nodes": self.config.max_nodes,
                "min_trust_score": self.config.min_trust_score,
                "knowledge_retention_days": self.config.knowledge_retention_days,
                "privacy_level": self.config.privacy_level,
            },
            "known_nodes": {
                node_id: {
                    "node_id": node.node_id,
                    "trust_score": node.trust_score,
                    "contribution_score": node.contribution_score,
                    "knowledge_quality": node.knowledge_quality,
                    "active": node.active,
                    "last_seen": node.last_seen,
                }
                for node_id, node in self.known_nodes.items()
            },
            "local_knowledge_summary": {
                key: {"timestamp": data.get("timestamp"), "sources": data.get("sources")}
                for key, data in self.local_knowledge.items()
                if isinstance(data, dict)
            },
            "statistics": self.get_federation_statistics(),
        }
