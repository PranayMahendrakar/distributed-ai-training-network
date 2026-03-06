"""
Federated Client - Local node implementation for distributed AI training
Manages local training, gradient computation, and secure communication with server
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for a federated learning client node"""
    node_id: str
    server_address: str
    data_path: str
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    clip_norm: float = 1.0
    compression_ratio: float = 0.1
    use_differential_privacy: bool = True
    use_compression: bool = True
    max_rounds: int = 100
    timeout: int = 300


class FederatedClient:
    """
    Federated learning client that runs on each distributed node.
    
    Responsibilities:
    - Load and manage local (private) data
    - Train local model updates
    - Apply differential privacy to gradients
    - Compress and send updates to server
    - Receive and apply global model updates
    
    Privacy Guarantees:
    - Raw data never leaves the node
    - Gradients are clipped and noised before transmission
    - Privacy budget is tracked and enforced
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.node_id = config.node_id
        self.current_model: Optional[Dict[str, np.ndarray]] = None
        self.model_version: Optional[str] = None
        self.round_num = 0
        self.training_history: List[Dict] = []
        
        # Initialize privacy engine
        if config.use_differential_privacy:
            from privacy.differential_privacy import DifferentialPrivacyEngine
            self.dp_engine = DifferentialPrivacyEngine(
                epsilon=config.privacy_epsilon,
                delta=config.privacy_delta,
                mechanism="gaussian",
                clip_norm=config.clip_norm
            )
        else:
            self.dp_engine = None
        
        # Initialize gradient compressor
        if config.use_compression:
            from compression.gradient_compressor import GradientCompressor
            self.compressor = GradientCompressor(
                compression_ratio=config.compression_ratio
            )
        else:
            self.compressor = None
        
        logger.info(
            f"FederatedClient initialized | "
            f"Node: {self.node_id} | "
            f"Server: {config.server_address} | "
            f"DP: {config.use_differential_privacy}"
        )

    def load_local_data(self) -> Dict[str, Any]:
        """
        Load local private data for training.
        Data never leaves this node.
        """
        from client.data_loader import PrivacyDataLoader
        
        loader = PrivacyDataLoader(
            data_path=self.config.data_path,
            batch_size=self.config.batch_size
        )
        
        data = loader.load()
        logger.info(
            f"Node {self.node_id}: Loaded {data['num_samples']} local samples"
        )
        
        return data

    def train_local(
        self,
        model: Dict[str, np.ndarray],
        data: Dict[str, Any],
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the model on local data.
        
        Args:
            model: Global model weights to start from
            data: Local training data
            epochs: Number of local training epochs
            
        Returns:
            Dict with gradients, loss, and training metadata
        """
        from client.local_trainer import LocalTrainer
        
        epochs = epochs or self.config.local_epochs
        
        trainer = LocalTrainer(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size
        )
        
        start_time = time.time()
        
        result = trainer.train(
            model=model,
            data=data,
            epochs=epochs
        )
        
        duration = time.time() - start_time
        
        logger.info(
            f"Node {self.node_id}: Local training complete | "
            f"Loss: {result['final_loss']:.4f} | "
            f"Duration: {duration:.2f}s | "
            f"Steps: {result['total_steps']}"
        )
        
        result["duration"] = duration
        result["node_id"] = self.node_id
        
        return result

    def prepare_update(
        self,
        local_result: Dict[str, Any],
        num_samples: int
    ) -> Dict[str, Any]:
        """
        Prepare model update for transmission to server.
        
        Pipeline:
        1. Extract gradients from local training
        2. Apply differential privacy (clip + noise)
        3. Compress gradients
        4. Package for transmission
        
        Raw data is NEVER included in this update.
        """
        gradients = local_result["gradients"]
        
        # Step 1: Apply differential privacy
        if self.dp_engine is not None:
            gradients = self.dp_engine.privatize(
                gradients=gradients,
                num_samples=num_samples,
                total_samples=num_samples,
                clip=True
            )
            privacy_info = self.dp_engine.privacy_spent()
        else:
            privacy_info = {"epsilon_spent": 0.0}
        
        # Step 2: Compress gradients
        compressed_size = None
        if self.compressor is not None:
            compressed = self.compressor.compress(gradients)
            gradients = compressed["compressed_gradients"]
            compressed_size = compressed["compressed_size"]
        
        # Step 3: Package update
        update = {
            "node_id": self.node_id,
            "round_id": self.round_num,
            "gradients": gradients,
            "num_samples": num_samples,
            "metadata": {
                "loss": local_result.get("final_loss", float("inf")),
                "accuracy": local_result.get("accuracy", 0.0),
                "local_steps": local_result.get("total_steps", 0),
                "duration": local_result.get("duration", 0.0),
                "privacy_spent": privacy_info.get("epsilon_spent", 0.0),
                "compressed_size": compressed_size
            }
        }
        
        logger.info(
            f"Node {self.node_id}: Update prepared | "
            f"Privacy spent: {privacy_info.get('epsilon_spent', 0):.4f} | "
            f"Compression: {compressed_size}"
        )
        
        return update

    def apply_global_model(self, global_model: Dict[str, np.ndarray], version: str):
        """
        Apply the new global model received from the server.
        
        Args:
            global_model: Updated global model weights
            version: Model version identifier
        """
        self.current_model = global_model
        self.model_version = version
        
        logger.info(
            f"Node {self.node_id}: Applied global model | Version: {version}"
        )

    def run_training_round(self) -> Dict[str, Any]:
        """
        Execute a single federated learning round:
        1. Load local data
        2. Train locally
        3. Privatize gradients
        4. Prepare and return update
        """
        self.round_num += 1
        
        logger.info(f"Node {self.node_id}: Starting round {self.round_num}")
        
        # Load local private data
        data = self.load_local_data()
        num_samples = data["num_samples"]
        
        # Initialize model if needed
        if self.current_model is None:
            self.current_model = self._initialize_model()
        
        # Train locally
        local_result = self.train_local(
            model=self.current_model,
            data=data
        )
        
        # Prepare privatized update
        update = self.prepare_update(local_result, num_samples)
        
        # Record round history
        self.training_history.append({
            "round": self.round_num,
            "loss": local_result.get("final_loss", float("inf")),
            "accuracy": local_result.get("accuracy", 0.0),
            "num_samples": num_samples,
            "privacy_spent": update["metadata"].get("privacy_spent", 0.0)
        })
        
        return update

    def _initialize_model(self) -> Dict[str, np.ndarray]:
        """Initialize model with random weights (first round)"""
        # Simple initialization - in practice would use a specific architecture
        model = {
            "layer1.weight": np.random.randn(128, 784) * 0.01,
            "layer1.bias": np.zeros(128),
            "layer2.weight": np.random.randn(64, 128) * 0.01,
            "layer2.bias": np.zeros(64),
            "output.weight": np.random.randn(10, 64) * 0.01,
            "output.bias": np.zeros(10)
        }
        logger.info(f"Node {self.node_id}: Initialized model weights")
        return model

    def get_status(self) -> Dict:
        """Get current client status and training progress"""
        privacy_info = {}
        if self.dp_engine:
            privacy_info = self.dp_engine.privacy_spent()
        
        return {
            "node_id": self.node_id,
            "rounds_completed": self.round_num,
            "model_version": self.model_version,
            "privacy": privacy_info,
            "training_history": self.training_history[-5:],  # Last 5 rounds
            "data_path": self.config.data_path,
            "use_dp": self.config.use_differential_privacy
        }


def create_client_from_args(args) -> FederatedClient:
    """Create a FederatedClient from command-line arguments"""
    config = ClientConfig(
        node_id=args.node_id or str(uuid.uuid4())[:8],
        server_address=args.server,
        data_path=args.data_path,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        privacy_epsilon=args.epsilon,
        privacy_delta=args.delta,
        clip_norm=args.clip_norm,
        use_differential_privacy=not args.no_dp,
        use_compression=not args.no_compression,
        max_rounds=args.rounds
    )
    
    return FederatedClient(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Client Node")
    parser.add_argument("--server", default="localhost:8080", help="Server address")
    parser.add_argument("--node-id", default=None, help="Node identifier")
    parser.add_argument("--data-path", required=True, help="Path to local data")
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=1.0, help="DP epsilon")
    parser.add_argument("--delta", type=float, default=1e-5, help="DP delta")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--no-dp", action="store_true", help="Disable differential privacy")
    parser.add_argument("--no-compression", action="store_true", help="Disable compression")
    
    args = parser.parse_args()
    
    client = create_client_from_args(args)
    
    logger.info(f"Client node {client.node_id} ready | Server: {args.server}")
    status = client.get_status()
    print(f"Client Status: {status}")
