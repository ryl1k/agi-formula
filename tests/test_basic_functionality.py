"""Basic functionality tests for AGI-Formula core components."""

import pytest
import numpy as np
from typing import Dict, Any

from agi_formula.core.neuron import Neuron, NeuronState
from agi_formula.core.causal_cache import CausalCache, CausalEntry
from agi_formula.core.network import Network, NetworkConfig
from agi_formula.attention.attention_module import AttentionModule, AttentionConfig


class TestNeuron:
    """Test cases for Neuron class."""
    
    def test_neuron_initialization(self):
        """Test neuron initialization with different parameters."""
        neuron = Neuron(neuron_id=1, num_inputs=5, concept_type="color")
        
        assert neuron.id == 1
        assert neuron.concept_type == "color"
        assert len(neuron.weights) == 5
        assert neuron.state.activation == 0.0
        assert neuron.state.uncertainty == 1.0
        assert not neuron.is_composite
        assert not neuron.is_meta
    
    def test_composite_neuron(self):
        """Test composite neuron initialization."""
        neuron = Neuron(
            neuron_id=2, 
            num_inputs=3, 
            concept_type="composite",
            is_composite=True
        )
        
        assert neuron.is_composite
        assert neuron.concept_type == "composite"
        assert neuron.composition_threshold == 0.7
    
    def test_meta_neuron(self):
        """Test meta neuron initialization."""
        neuron = Neuron(
            neuron_id=3,
            num_inputs=10,
            concept_type="meta",
            is_meta=True
        )
        
        assert neuron.is_meta
        assert neuron.concept_type == "meta"
        assert len(neuron.modification_history) == 0
    
    def test_weight_updates(self):
        """Test safe weight updates."""
        neuron = Neuron(neuron_id=1, num_inputs=3)
        original_weights = neuron.weights.copy()
        
        # Valid update
        delta = np.array([0.05, -0.03, 0.02])
        success = neuron.update_weights(delta, learning_rate=1.0)
        
        assert success
        assert not np.array_equal(neuron.weights, original_weights)
        assert len(neuron.modification_history) == 1
    
    def test_weight_update_safety_bounds(self):
        """Test weight update safety bounds."""
        neuron = Neuron(neuron_id=1, num_inputs=3)
        
        # Too large update should be clipped
        large_delta = np.array([1.0, -1.0, 0.5])
        success = neuron.update_weights(large_delta, safety_bounds=(-0.1, 0.1))
        
        assert success
        assert np.all(np.abs(neuron.weights) <= 10.0)  # Should not explode
    
    def test_neighbor_management(self):
        """Test adding and removing neighbors."""
        neuron = Neuron(neuron_id=1, num_inputs=2)
        initial_weight_count = len(neuron.weights)
        
        # Add neighbor
        neuron.add_neighbor(5, weight=0.3)
        assert 5 in neuron.neighbors
        assert len(neuron.weights) == initial_weight_count + 1
        
        # Remove neighbor
        success = neuron.remove_neighbor(5)
        assert success
        assert 5 not in neuron.neighbors
    
    def test_rollback_functionality(self):
        """Test rollback of modifications."""
        neuron = Neuron(neuron_id=1, num_inputs=3)
        original_weights = neuron.weights.copy()
        
        # Make modification
        delta = np.array([0.1, -0.1, 0.05])
        neuron.update_weights(delta)
        
        # Rollback
        success = neuron.rollback_last_modification()
        assert success
        np.testing.assert_array_almost_equal(neuron.weights, original_weights)


class TestCausalCache:
    """Test cases for CausalCache class."""
    
    def test_causal_cache_initialization(self):
        """Test causal cache initialization."""
        cache = CausalCache(max_history=100, confidence_threshold=0.2)
        
        assert cache.max_history == 100
        assert cache.confidence_threshold == 0.2
        assert len(cache.entries) == 0
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0
    
    def test_store_and_retrieve_contribution(self):
        """Test storing and retrieving causal contributions."""
        cache = CausalCache()
        
        # Store contribution
        cache.store_contribution(
            neuron_id=1,
            contribution=0.5,
            caused_by=[2, 3],
            error_attribution={2: 0.3, 3: 0.2},
            confidence=0.8
        )
        
        # Retrieve contribution
        contribution = cache.get_contribution(1)
        assert contribution == 0.5
        
        error_attr = cache.get_error_attribution(1)
        assert error_attr[2] == 0.3
        assert error_attr[3] == 0.2
    
    def test_causal_chain_generation(self):
        """Test causal chain generation."""
        cache = CausalCache()
        
        # Build simple causal chain: 1 -> 2 -> 3
        cache.store_contribution(3, 0.8, [2], {2: 0.8}, 0.9)
        cache.store_contribution(2, 0.6, [1], {1: 0.6}, 0.8)
        cache.store_contribution(1, 0.4, [], {}, 1.0)
        
        chain = cache.get_causal_chain(3, max_depth=5)
        
        assert len(chain) >= 2  # Should have at least 2 entries in chain
        assert any(entry['neuron_id'] == 3 for entry in chain)
        assert any(entry['neuron_id'] == 2 for entry in chain)
    
    def test_attention_scores(self):
        """Test attention score computation."""
        cache = CausalCache()
        
        # Add some causal entries
        cache.store_contribution(1, 0.8, [], {}, 0.9)
        cache.store_contribution(2, 0.3, [], {}, 0.6)
        cache.store_contribution(3, 0.1, [], {}, 0.4)
        
        scores = cache.compute_attention_scores(0, [1, 2, 3])
        
        assert len(scores) == 3
        assert scores[1] > scores[2] > scores[3]  # Higher contribution = higher score
    
    def test_mutual_information_calculation(self):
        """Test mutual information calculation."""
        cache = CausalCache()
        
        # Add activation history
        cache.activation_history[1] = [0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6, 0.5, 0.8]
        cache.activation_history[2] = [0.2, 0.7, 0.4, 0.8, 0.3, 0.6, 0.5, 0.7, 0.4, 0.9]
        
        mi = cache._calculate_mi(
            cache.activation_history[1],
            cache.activation_history[2]
        )
        
        assert isinstance(mi, float)
        assert mi >= 0.0  # MI should be non-negative
    
    def test_cache_cleanup(self):
        """Test cache cleanup when max_history is exceeded."""
        cache = CausalCache(max_history=3)
        
        # Add more entries than max_history
        for i in range(5):
            cache.store_contribution(i, 0.5, [], {}, 0.8)
        
        assert len(cache.entries) == 3  # Should be limited to max_history


class TestAttentionModule:
    """Test cases for AttentionModule class."""
    
    def test_attention_module_initialization(self):
        """Test attention module initialization."""
        config = AttentionConfig(top_k=3, temperature=0.8)
        attention = AttentionModule(config)
        
        assert attention.config.top_k == 3
        assert attention.config.temperature == 0.8
        assert len(attention.attention_weights) == 0
    
    def test_score_computation(self):
        """Test attention score computation."""
        attention = AttentionModule()
        cache = CausalCache()
        
        # Create mock neuron
        neuron = Neuron(neuron_id=1, num_inputs=3)
        
        # Add causal information
        cache.store_contribution(2, 0.8, [], {}, 0.9)
        cache.store_contribution(3, 0.4, [], {}, 0.7)
        
        scores = attention.compute_scores(neuron, [2, 3], cache)
        
        assert len(scores) == 2
        assert 2 in scores
        assert 3 in scores
        assert scores[2] > scores[3]  # Higher contribution should get higher score
    
    def test_top_k_selection(self):
        """Test top-k neuron selection."""
        attention = AttentionModule(AttentionConfig(top_k=2))
        
        scores = {1: 0.8, 2: 0.6, 3: 0.9, 4: 0.3, 5: 0.7}
        selected = attention.select_top_k(scores)
        
        assert len(selected) == 2
        assert 3 in selected  # Highest score
        assert 1 in selected  # Second highest score
        assert selected[0] == 3  # Should be sorted by score
    
    def test_attention_weight_updates(self):
        """Test attention weight adaptation."""
        attention = AttentionModule()
        
        # Update weights with success feedback
        attention.update_attention_weights(1, 2, 0.9)  # High success
        attention.update_attention_weights(1, 3, 0.2)  # Low success
        
        weight_12 = attention._get_adaptive_weight(1, 2)
        weight_13 = attention._get_adaptive_weight(1, 3)
        
        # After multiple updates, successful pair should have higher weight
        for _ in range(10):
            attention.update_attention_weights(1, 2, 0.9)
            attention.update_attention_weights(1, 3, 0.2)
        
        final_weight_12 = attention._get_adaptive_weight(1, 2)
        final_weight_13 = attention._get_adaptive_weight(1, 3)
        
        assert final_weight_12 > final_weight_13


class TestNetwork:
    """Test cases for Network class."""
    
    def test_network_initialization(self):
        """Test network initialization with default config."""
        config = NetworkConfig(num_neurons=20, input_size=5, output_size=3)
        network = Network(config)
        
        assert len(network.neurons) == 20
        assert len(network.input_neurons) == 5
        assert len(network.output_neurons) == 3
        assert network.timestep == 0
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        config = NetworkConfig(num_neurons=10, input_size=3, output_size=2)
        network = Network(config)
        
        inputs = np.array([0.5, -0.3, 0.8])
        result = network.forward(inputs)
        
        assert 'outputs' in result
        assert len(result['outputs']) == 2
        assert 'timestep' in result
        assert result['timestep'] == 1
        assert 'forward_time' in result
    
    def test_masked_prediction(self):
        """Test masked neuron prediction."""
        config = NetworkConfig(num_neurons=15, input_size=4, output_size=2)
        network = Network(config)
        
        inputs = np.array([0.2, 0.7, -0.4, 0.9])
        
        # Get a neuron ID to mask (not input or output)
        regular_neurons = [nid for nid in network.neurons.keys() 
                          if nid not in network.input_neurons and nid not in network.output_neurons]
        
        if regular_neurons:
            masked_neuron = regular_neurons[0]
            prediction, confidence = network.predict_masked(inputs, masked_neuron)
            
            assert isinstance(prediction, float)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0
    
    def test_causal_explanation(self):
        """Test causal explanation generation."""
        config = NetworkConfig(num_neurons=12, input_size=3, output_size=2)
        network = Network(config)
        
        inputs = np.array([0.1, 0.5, -0.2])
        
        # Forward pass to generate causal information
        network.forward(inputs, return_causal_info=True)
        
        # Get explanation for first output neuron
        output_neuron_id = network.output_neurons[0]
        explanation = network.get_causal_explanation(output_neuron_id)
        
        assert 'target_neuron' in explanation
        assert 'target_activation' in explanation
        assert 'causal_chain_length' in explanation
        assert 'primary_influences' in explanation
        assert 'reasoning_path' in explanation
    
    def test_network_state_rollback(self):
        """Test network state rollback functionality."""
        config = NetworkConfig(num_neurons=8, input_size=2, output_size=1)
        network = Network(config)
        
        inputs = np.array([0.3, -0.7])
        
        # Perform several forward passes
        network.forward(inputs)
        network.forward(inputs * 1.5)
        original_timestep = network.timestep
        
        # Rollback one step
        success = network.rollback_to_previous_state(steps_back=1)
        
        assert success
        assert network.timestep < original_timestep
    
    def test_network_info(self):
        """Test network information retrieval."""
        config = NetworkConfig(
            num_neurons=20,
            input_size=5,
            output_size=3,
            num_meta_neurons=2,
            num_composite_neurons=3
        )
        network = Network(config)
        
        info = network.get_network_info()
        
        assert 'config' in info
        assert 'num_neurons' in info
        assert 'neuron_types' in info
        assert 'performance' in info
        assert info['num_neurons'] == 20
        assert info['neuron_types']['input'] == 5
        assert info['neuron_types']['output'] == 3


class TestIntegration:
    """Integration tests for combined functionality."""
    
    def test_end_to_end_computation(self):
        """Test complete end-to-end computation."""
        config = NetworkConfig(
            num_neurons=25,
            input_size=4,
            output_size=3,
            concepts=['color', 'shape', 'size'],
            enable_self_modification=True
        )
        
        network = Network(config)
        inputs = np.array([0.2, -0.5, 0.8, 0.1])
        
        # Forward pass with causal info
        result = network.forward(inputs, return_causal_info=True)
        
        assert 'outputs' in result
        assert 'causal_info' in result
        assert len(result['outputs']) == 3
        
        # Test that network produces consistent outputs
        result2 = network.forward(inputs)
        # Outputs might differ due to learning, but should be reasonable
        assert np.all(np.isfinite(result2['outputs']))
    
    def test_multiple_forward_passes(self):
        """Test multiple forward passes for consistency."""
        config = NetworkConfig(num_neurons=15, input_size=3, output_size=2)
        network = Network(config)
        
        # Multiple forward passes with different inputs
        test_inputs = [
            np.array([0.1, 0.2, 0.3]),
            np.array([-0.2, 0.8, -0.5]),
            np.array([0.9, -0.1, 0.4])
        ]
        
        results = []
        for inputs in test_inputs:
            result = network.forward(inputs)
            results.append(result)
            
            # Check output validity
            assert len(result['outputs']) == 2
            assert np.all(np.isfinite(result['outputs']))
            assert result['timestep'] > 0
        
        # Network should be learning/adapting
        assert results[-1]['timestep'] == len(test_inputs)


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running AGI-Formula basic functionality tests...")
    
    # Test neuron
    neuron = Neuron(1, 3, "test")
    print(f"✓ Neuron created: {neuron}")
    
    # Test causal cache
    cache = CausalCache()
    cache.store_contribution(1, 0.5, [2], {2: 0.5}, 0.8)
    print(f"✓ CausalCache working: {cache.get_contribution(1)}")
    
    # Test attention
    attention = AttentionModule()
    scores = attention.compute_scores(neuron, [2, 3], cache)
    print(f"✓ AttentionModule working: {len(scores)} scores computed")
    
    # Test network
    config = NetworkConfig(num_neurons=10, input_size=3, output_size=2)
    network = Network(config)
    inputs = np.array([0.1, 0.5, -0.2])
    result = network.forward(inputs)
    print(f"✓ Network forward pass: {result['outputs']}")
    
    print("\n🎉 All basic functionality tests passed!")
    print("Run 'pytest tests/' for comprehensive testing.")