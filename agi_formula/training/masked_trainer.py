"""Masked trainer for self-supervised learning in AGI-Formula."""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import random
import time
from dataclasses import dataclass, field

from ..core.network import Network


@dataclass
class TrainingConfig:
    """Configuration for masked training."""
    # Masking strategy
    mask_probability: float = 0.15
    mask_random_neurons: bool = True
    mask_output_neurons: bool = False
    mask_input_neurons: bool = False
    
    # Training parameters
    learning_rate: float = 0.01
    batch_size: int = 1
    epochs: int = 100
    
    # Validation
    validation_split: float = 0.2
    patience: int = 10
    min_improvement: float = 1e-4
    
    # Logging
    log_frequency: int = 10
    save_checkpoints: bool = True
    
    # AGI-specific
    causal_weight: float = 0.3
    attention_feedback: bool = True
    enable_self_modification: bool = False


@dataclass
class TrainingMetrics:
    """Training metrics and statistics."""
    epoch: int = 0
    total_loss: float = 0.0
    prediction_accuracy: float = 0.0
    causal_consistency: float = 0.0
    attention_efficiency: float = 0.0
    
    # Detailed metrics
    masked_neurons_count: int = 0
    successful_predictions: int = 0
    total_predictions: int = 0
    
    # Performance
    forward_time: float = 0.0
    training_time: float = 0.0
    
    # AGI metrics
    concept_coherence: float = 0.0
    reasoning_depth: float = 0.0


class MaskedTrainer:
    """
    Trains AGI network using masked neuron prediction for self-supervised learning.
    
    This implements the core AGI training strategy:
    1. Mask random neurons in the network
    2. Let network predict masked neuron activations using causal reasoning
    3. Learn from prediction errors to improve causal understanding
    4. Adapt attention mechanism based on success/failure
    """
    
    def __init__(self, network: Network, config: TrainingConfig = None):
        """
        Initialize masked trainer.
        
        Args:
            network: AGI network to train
            config: Training configuration
        """
        self.network = network
        self.config = config or TrainingConfig()
        
        # Training state
        self.current_epoch = 0
        self.training_history: List[TrainingMetrics] = []
        self.best_metrics: Optional[TrainingMetrics] = None
        self.patience_counter = 0
        
        # Training data
        self.training_examples: List[np.ndarray] = []
        self.validation_examples: List[np.ndarray] = []
        
        # Performance tracking
        self.total_masked_predictions = 0
        self.successful_predictions = 0
        
    def prepare_training_data(self, data: List[np.ndarray]) -> None:
        """
        Prepare training and validation data.
        
        Args:
            data: List of input examples for training
        """
        # Shuffle data
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        # Split into training/validation
        split_idx = int(len(shuffled_data) * (1 - self.config.validation_split))
        self.training_examples = shuffled_data[:split_idx]
        self.validation_examples = shuffled_data[split_idx:]
        
        print(f"Training data prepared: {len(self.training_examples)} training, {len(self.validation_examples)} validation examples")
    
    def train(self, training_data: List[np.ndarray]) -> List[TrainingMetrics]:
        """
        Train the network using masked prediction.
        
        Args:
            training_data: List of input examples
            
        Returns:
            List of training metrics for each epoch
        """
        print(f"Starting AGI masked training for {self.config.epochs} epochs...")
        
        # Prepare data
        self.prepare_training_data(training_data)
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Clear only activation cache between epochs (keep causal entries)
            self.network.causal_cache.clear_activation_cache_only()
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Combine metrics
            epoch_metrics = self._combine_metrics(train_metrics, val_metrics, epoch)
            self.training_history.append(epoch_metrics)
            
            # Check for improvement
            if self._is_improvement(epoch_metrics):
                self.best_metrics = epoch_metrics
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Logging
            if epoch % self.config.log_frequency == 0:
                self._log_progress(epoch_metrics)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.config.patience} epochs)")
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Best metrics: Loss={self.best_metrics.total_loss:.4f}, Accuracy={self.best_metrics.prediction_accuracy:.4f}")
        
        return self.training_history
    
    def _train_epoch(self) -> TrainingMetrics:
        """Train for one epoch."""
        epoch_metrics = TrainingMetrics()
        total_loss = 0.0
        total_predictions = 0
        successful_predictions = 0
        
        # Shuffle training examples
        training_batch = self.training_examples.copy()
        random.shuffle(training_batch)
        
        for example in training_batch:
            # Select neurons to mask
            masked_neurons = self._select_neurons_to_mask()
            
            if not masked_neurons:
                continue
            
            for masked_neuron in masked_neurons:
                # Get ground truth
                ground_truth_result = self.network.forward(example)
                ground_truth_activation = self.network.neurons[masked_neuron].state.activation
                
                # Make prediction with masked neuron
                prediction, confidence = self.network.predict_masked(example, masked_neuron)
                
                # Calculate loss
                loss = self._calculate_loss(prediction, ground_truth_activation, confidence)
                total_loss += loss
                total_predictions += 1
                
                # Check if prediction is successful
                if abs(prediction - ground_truth_activation) < 0.1:  # Tolerance threshold
                    successful_predictions += 1
                
                # Update neuron weights based on prediction error
                error = ground_truth_activation - prediction
                if abs(error) > 0.01:  # Only update if there's significant error
                    self._update_neuron_weights(masked_neuron, error, example)
                
                # Update attention mechanism with feedback
                if self.config.attention_feedback:
                    success_score = 1.0 - min(1.0, abs(prediction - ground_truth_activation))
                    self._update_attention_feedback(masked_neuron, success_score)
                
                # Causal consistency check
                causal_consistency = self._check_causal_consistency(masked_neuron, prediction)
                epoch_metrics.causal_consistency += causal_consistency
        
        # Aggregate metrics
        if total_predictions > 0:
            epoch_metrics.total_loss = total_loss / total_predictions
            epoch_metrics.prediction_accuracy = successful_predictions / total_predictions
            epoch_metrics.masked_neurons_count = len(masked_neurons) if masked_neurons else 0
            epoch_metrics.successful_predictions = successful_predictions
            epoch_metrics.total_predictions = total_predictions
            epoch_metrics.causal_consistency /= total_predictions
        
        return epoch_metrics
    
    def _validate_epoch(self) -> TrainingMetrics:
        """Validate for one epoch."""
        if not self.validation_examples:
            return TrainingMetrics()
        
        val_metrics = TrainingMetrics()
        total_loss = 0.0
        total_predictions = 0
        successful_predictions = 0
        
        for example in self.validation_examples:
            masked_neurons = self._select_neurons_to_mask()
            
            if not masked_neurons:
                continue
            
            for masked_neuron in masked_neurons:
                # Ground truth
                ground_truth_result = self.network.forward(example)
                ground_truth_activation = self.network.neurons[masked_neuron].state.activation
                
                # Prediction
                prediction, confidence = self.network.predict_masked(example, masked_neuron)
                
                # Loss
                loss = self._calculate_loss(prediction, ground_truth_activation, confidence)
                total_loss += loss
                total_predictions += 1
                
                if abs(prediction - ground_truth_activation) < 0.1:
                    successful_predictions += 1
        
        # Aggregate
        if total_predictions > 0:
            val_metrics.total_loss = total_loss / total_predictions
            val_metrics.prediction_accuracy = successful_predictions / total_predictions
            val_metrics.total_predictions = total_predictions
            val_metrics.successful_predictions = successful_predictions
        
        return val_metrics
    
    def _select_neurons_to_mask(self) -> List[int]:
        """Select neurons to mask for training."""
        maskable_neurons = []
        
        # Get all neurons except input/output if configured
        for neuron_id, neuron in self.network.neurons.items():
            if not self.config.mask_input_neurons and neuron_id in self.network.input_neurons:
                continue
            if not self.config.mask_output_neurons and neuron_id in self.network.output_neurons:
                continue
            maskable_neurons.append(neuron_id)
        
        if not maskable_neurons:
            return []
        
        # Select neurons to mask
        if self.config.mask_random_neurons:
            num_to_mask = max(1, int(len(maskable_neurons) * self.config.mask_probability))
            return random.sample(maskable_neurons, min(num_to_mask, len(maskable_neurons)))
        else:
            # Strategic masking based on importance
            return self._select_strategic_neurons(maskable_neurons)
    
    def _select_strategic_neurons(self, candidates: List[int]) -> List[int]:
        """Select neurons strategically based on importance."""
        # Select random neurons for now (can be improved later)
        num_to_mask = max(1, int(len(candidates) * self.config.mask_probability))
        return random.sample(candidates, min(num_to_mask, len(candidates)))
    
    def _calculate_loss(self, prediction: float, ground_truth: float, confidence: float) -> float:
        """Calculate advanced training loss with multiple components."""
        # 1. Base prediction loss (MSE)
        mse_loss = (prediction - ground_truth) ** 2
        
        # 2. Huber loss for robustness to outliers
        delta = 0.1
        if abs(prediction - ground_truth) <= delta:
            huber_loss = 0.5 * mse_loss
        else:
            huber_loss = delta * (abs(prediction - ground_truth) - 0.5 * delta)
        
        # 3. Confidence-weighted component
        confidence_weight = max(0.1, confidence)
        confidence_loss = mse_loss / confidence_weight
        
        # 4. Entropy regularization to encourage exploration
        # Prevent predictions from being too certain
        pred_entropy = -prediction * np.log(max(prediction, 1e-8)) - (1-prediction) * np.log(max(1-prediction, 1e-8))
        entropy_bonus = 0.01 * pred_entropy  # Small bonus for entropy
        
        # Combine losses
        total_loss = (
            0.5 * huber_loss +     # Robust prediction loss
            0.3 * confidence_loss + # Confidence-aware loss
            0.2 * mse_loss -       # Standard MSE
            entropy_bonus          # Exploration bonus
        )
        
        return max(0.0, total_loss)
    
    def _update_neuron_weights(self, neuron_id: int, error: float, inputs: np.ndarray) -> None:
        """Update neuron weights based on prediction error."""
        if neuron_id not in self.network.neurons:
            return
        
        neuron = self.network.neurons[neuron_id]
        
        # Calculate weight updates using simple gradient descent
        # For masked prediction: minimize error between prediction and ground truth
        learning_rate = self.config.learning_rate
        
        # Get input features for this neuron (limited to weight size)
        input_features = inputs[:len(neuron.weights)] if len(neuron.weights) > 0 else inputs
        
        # Resize inputs to match weights if necessary
        if len(input_features) > len(neuron.weights) and len(neuron.weights) > 0:
            input_features = input_features[:len(neuron.weights)]
        elif len(input_features) < len(neuron.weights):
            # Pad with zeros
            padding = np.zeros(len(neuron.weights) - len(input_features))
            input_features = np.concatenate([input_features, padding])
        
        # Calculate weight deltas: delta_w = learning_rate * error * input
        if len(input_features) == len(neuron.weights) and len(neuron.weights) > 0:
            weight_deltas = learning_rate * error * input_features
            
            # Apply weight update with safety bounds
            neuron.update_weights(weight_deltas, learning_rate=1.0)  # Already scaled
        
        # Update bias
        bias_delta = learning_rate * error
        neuron.bias += bias_delta * 0.1  # Smaller learning rate for bias
        
        # Clip bias to reasonable range
        neuron.bias = np.clip(neuron.bias, -2.0, 2.0)
    
    def _update_attention_feedback(self, masked_neuron: int, success_score: float) -> None:
        """Update attention mechanism with feedback."""
        # Get causal influences for this neuron
        causal_entry = self.network.causal_cache.entries.get(masked_neuron)
        
        if causal_entry:
            # Update attention weights for neurons that caused this one
            for cause_id in causal_entry.caused_by_neurons:
                if hasattr(self.network.attention_module, 'update_attention_weights'):
                    self.network.attention_module.update_attention_weights(
                        masked_neuron, cause_id, success_score
                    )
    
    def _check_causal_consistency(self, neuron_id: int, prediction: float) -> float:
        """Check consistency of causal reasoning."""
        causal_entry = self.network.causal_cache.entries.get(neuron_id)
        
        if not causal_entry:
            return 0.5  # Neutral score
        
        # Check if prediction aligns with causal contributions
        expected_contribution = causal_entry.contribution
        prediction_strength = abs(prediction)
        
        # Consistency is high if prediction strength matches contribution
        consistency = 1.0 - abs(prediction_strength - abs(expected_contribution))
        return max(0.0, min(1.0, consistency))
    
    def _combine_metrics(self, train_metrics: TrainingMetrics, val_metrics: TrainingMetrics, epoch: int) -> TrainingMetrics:
        """Combine training and validation metrics."""
        combined = TrainingMetrics()
        combined.epoch = epoch
        
        # Use training metrics as base
        combined.total_loss = train_metrics.total_loss
        combined.prediction_accuracy = train_metrics.prediction_accuracy
        combined.causal_consistency = train_metrics.causal_consistency
        combined.masked_neurons_count = train_metrics.masked_neurons_count
        combined.successful_predictions = train_metrics.successful_predictions
        combined.total_predictions = train_metrics.total_predictions
        
        # Add validation loss if available
        if val_metrics.total_predictions > 0:
            combined.total_loss = (train_metrics.total_loss + val_metrics.total_loss) / 2
        
        # Calculate AGI-specific metrics
        combined.attention_efficiency = self._calculate_attention_efficiency()
        combined.concept_coherence = self._calculate_concept_coherence()
        combined.reasoning_depth = self._calculate_reasoning_depth()
        
        return combined
    
    def _calculate_attention_efficiency(self) -> float:
        """Calculate attention mechanism efficiency."""
        attention_stats = self.network.attention_module.get_attention_statistics()
        
        if not attention_stats or 'selection_stats' not in attention_stats:
            return 0.5
        
        selection_stats = attention_stats['selection_stats']
        
        # Efficiency is based on how focused the attention is
        avg_candidates = selection_stats.get('avg_candidates', 1)
        avg_selected = selection_stats.get('avg_k_used', 1)
        
        if avg_candidates == 0:
            return 0.5
        
        focus_ratio = avg_selected / avg_candidates
        return min(1.0, focus_ratio * 2)  # Scale to [0, 1]
    
    def _calculate_concept_coherence(self) -> float:
        """Calculate coherence of concept representations."""
        # Simple heuristic: neurons of same concept type should have similar activations
        concept_groups = {}
        
        for neuron_id, neuron in self.network.neurons.items():
            if neuron.concept_type:
                if neuron.concept_type not in concept_groups:
                    concept_groups[neuron.concept_type] = []
                concept_groups[neuron.concept_type].append(neuron.state.activation)
        
        total_coherence = 0.0
        group_count = 0
        
        for concept_type, activations in concept_groups.items():
            if len(activations) > 1:
                # Coherence is inverse of variance
                variance = np.var(activations)
                coherence = 1.0 / (1.0 + variance)
                total_coherence += coherence
                group_count += 1
        
        return total_coherence / group_count if group_count > 0 else 0.5
    
    def _calculate_reasoning_depth(self) -> float:
        """Calculate average reasoning depth in causal chains."""
        total_depth = 0.0
        chain_count = 0
        
        for output_id in self.network.output_neurons:
            chain = self.network.causal_cache.get_causal_chain(output_id, max_depth=10)
            if chain:
                max_depth = max(entry['depth'] for entry in chain)
                total_depth += max_depth
                chain_count += 1
        
        avg_depth = total_depth / chain_count if chain_count > 0 else 1.0
        return min(1.0, avg_depth / 5.0)  # Normalize to [0, 1], depth 5+ is excellent
    
    def _is_improvement(self, metrics: TrainingMetrics) -> bool:
        """Check if current metrics represent an improvement."""
        if self.best_metrics is None:
            return True
        
        # Improvement if loss decreased AND accuracy increased
        loss_improvement = (self.best_metrics.total_loss - metrics.total_loss) > self.config.min_improvement
        accuracy_improvement = (metrics.prediction_accuracy - self.best_metrics.prediction_accuracy) > self.config.min_improvement
        
        # At least one must improve significantly
        return loss_improvement or accuracy_improvement
    
    def _log_progress(self, metrics: TrainingMetrics) -> None:
        """Log training progress."""
        print(f"Epoch {metrics.epoch:3d}: "
              f"Loss={metrics.total_loss:.4f}, "
              f"Acc={metrics.prediction_accuracy:.3f}, "
              f"Causal={metrics.causal_consistency:.3f}, "
              f"Attention={metrics.attention_efficiency:.3f}, "
              f"Coherence={metrics.concept_coherence:.3f}, "
              f"Depth={metrics.reasoning_depth:.3f}")
    
    def evaluate_agi_capabilities(self, test_data: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate AGI capabilities on test data.
        
        Args:
            test_data: Test examples
            
        Returns:
            Dictionary of AGI capability scores
        """
        print("Evaluating AGI capabilities...")
        
        capabilities = {
            'prediction_accuracy': 0.0,
            'causal_reasoning': 0.0,
            'attention_quality': 0.0,
            'concept_coherence': 0.0,
            'reasoning_depth': 0.0,
            'adaptation_ability': 0.0,
            'overall_agi_score': 0.0
        }
        
        total_examples = len(test_data)
        if total_examples == 0:
            return capabilities
        
        total_accuracy = 0.0
        total_causal = 0.0
        
        for i, example in enumerate(test_data):
            # Test prediction accuracy
            masked_neurons = self._select_neurons_to_mask()
            if masked_neurons:
                for masked_neuron in masked_neurons:
                    # Ground truth
                    ground_truth_result = self.network.forward(example)
                    ground_truth = self.network.neurons[masked_neuron].state.activation
                    
                    # Prediction
                    prediction, confidence = self.network.predict_masked(example, masked_neuron)
                    
                    # Accuracy
                    accuracy = 1.0 - min(1.0, abs(prediction - ground_truth))
                    total_accuracy += accuracy
                    
                    # Causal reasoning quality
                    causal_quality = self._check_causal_consistency(masked_neuron, prediction)
                    total_causal += causal_quality
        
        # Calculate averages
        num_predictions = len(test_data) * max(1, len(self._select_neurons_to_mask()))
        capabilities['prediction_accuracy'] = total_accuracy / num_predictions
        capabilities['causal_reasoning'] = total_causal / num_predictions
        
        # Other capabilities
        capabilities['attention_quality'] = self._calculate_attention_efficiency()
        capabilities['concept_coherence'] = self._calculate_concept_coherence()
        capabilities['reasoning_depth'] = self._calculate_reasoning_depth()
        
        # Adaptation ability (how much the network improved during training)
        if len(self.training_history) > 1:
            initial_acc = self.training_history[0].prediction_accuracy
            final_acc = self.training_history[-1].prediction_accuracy
            capabilities['adaptation_ability'] = max(0.0, final_acc - initial_acc)
        
        # Overall AGI score (weighted average)
        weights = {
            'prediction_accuracy': 0.25,
            'causal_reasoning': 0.25,
            'attention_quality': 0.15,
            'concept_coherence': 0.15,
            'reasoning_depth': 0.15,
            'adaptation_ability': 0.05
        }
        
        overall_score = sum(capabilities[key] * weight for key, weight in weights.items())
        capabilities['overall_agi_score'] = overall_score
        
        print("AGI Capabilities Assessment:")
        for capability, score in capabilities.items():
            print(f"   {capability}: {score:.3f}")
        
        return capabilities
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_history:
            return {}
        
        final_metrics = self.training_history[-1]
        best_metrics = self.best_metrics or final_metrics
        
        return {
            'training_completed': True,
            'total_epochs': len(self.training_history),
            'best_epoch': best_metrics.epoch,
            'final_metrics': {
                'loss': final_metrics.total_loss,
                'accuracy': final_metrics.prediction_accuracy,
                'causal_consistency': final_metrics.causal_consistency,
                'attention_efficiency': final_metrics.attention_efficiency,
                'concept_coherence': final_metrics.concept_coherence,
                'reasoning_depth': final_metrics.reasoning_depth
            },
            'best_metrics': {
                'loss': best_metrics.total_loss,
                'accuracy': best_metrics.prediction_accuracy,
                'causal_consistency': best_metrics.causal_consistency
            },
            'training_data_size': len(self.training_examples),
            'validation_data_size': len(self.validation_examples),
            'total_predictions': self.total_masked_predictions,
            'network_info': self.network.get_network_info()
        }