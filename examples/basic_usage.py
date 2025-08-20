"""
Basic usage example for AGI-Formula library.

This example demonstrates:
1. Creating an AGI network
2. Performing forward passes
3. Masked neuron prediction
4. Causal explanations
5. Basic network analysis
"""

import numpy as np
import agi_formula as agi


def main():
    """Run basic AGI-Formula demonstration."""
    
    print("🧠 AGI-Formula Basic Usage Example")
    print("=" * 50)
    
    # 1. Create network configuration
    print("\n1. Creating Network Configuration...")
    config = agi.NetworkConfig(
        num_neurons=30,
        input_size=5,
        output_size=3,
        concepts=['color', 'shape', 'size', 'action'],
        enable_self_modification=True,
        memory_depth=100
    )
    print(f"   Configuration: {config.num_neurons} neurons, {config.input_size} inputs, {config.output_size} outputs")
    
    # 2. Initialize the network
    print("\n2. Initializing AGI Network...")
    network = agi.Network(config)
    print(f"   Network created with {len(network.neurons)} neurons")
    print(f"   Input neurons: {len(network.input_neurons)}")
    print(f"   Output neurons: {len(network.output_neurons)}")
    print(f"   Meta neurons: {len(network.meta_neurons)}")
    print(f"   Composite neurons: {len(network.composite_neurons)}")
    
    # 3. Prepare test data
    print("\n3. Preparing Test Data...")
    test_inputs = [
        np.array([0.2, -0.5, 0.8, 0.1, -0.3]),  # Test case 1
        np.array([0.9, 0.3, -0.7, 0.6, 0.4]),   # Test case 2
        np.array([-0.1, 0.7, 0.2, -0.8, 0.5])   # Test case 3
    ]
    print(f"   Prepared {len(test_inputs)} test cases")
    
    # 4. Perform forward passes
    print("\n4. Performing Forward Passes...")
    results = []
    
    for i, inputs in enumerate(test_inputs):
        print(f"   Test case {i+1}: {inputs}")
        
        # Forward pass with causal information
        result = network.forward(inputs, return_causal_info=True)
        results.append(result)
        
        print(f"   → Outputs: {result['outputs']}")
        print(f"   → Forward time: {result['forward_time']:.4f}s")
        print(f"   → Neurons activated: {result['neurons_activated']}")
        print(f"   → Memory cache size: {result['memory_size']}")
        
        # Show most influential neurons
        if 'causal_info' in result:
            influential = result['causal_info']['most_influential'][:3]
            print(f"   → Most influential neurons: {[(nid, f'{contrib:.3f}') for nid, contrib in influential]}")
        
        print()
    
    # 5. Demonstrate masked prediction
    print("5. Demonstrating Masked Prediction...")
    
    # Select a regular neuron to mask (not input/output)
    regular_neurons = [
        nid for nid in network.neurons.keys() 
        if nid not in network.input_neurons 
        and nid not in network.output_neurons
        and nid not in network.meta_neurons
    ]
    
    if regular_neurons:
        masked_neuron = regular_neurons[0]
        test_input = test_inputs[0]
        
        print(f"   Masking neuron {masked_neuron} (type: {network.neurons[masked_neuron].concept_type})")
        
        # Get original activation
        original_result = network.forward(test_input)
        original_activation = network.neurons[masked_neuron].state.activation
        print(f"   Original activation: {original_activation:.4f}")
        
        # Predict masked neuron
        prediction, confidence = network.predict_masked(test_input, masked_neuron)
        print(f"   Predicted activation: {prediction:.4f}")
        print(f"   Prediction confidence: {confidence:.4f}")
        print(f"   Prediction error: {abs(original_activation - prediction):.4f}")
    else:
        print("   No suitable neurons available for masking")
    
    # 6. Generate causal explanations
    print("\n6. Generating Causal Explanations...")
    
    # Get explanation for first output neuron
    output_neuron_id = network.output_neurons[0]
    explanation = network.get_causal_explanation(output_neuron_id)
    
    print(f"   Explanation for output neuron {output_neuron_id}:")
    print(f"   → Final activation: {explanation['target_activation']:.4f}")
    print(f"   → Causal chain length: {explanation['causal_chain_length']}")
    
    if explanation['primary_influences']:
        print("   → Primary influences:")
        for influence in explanation['primary_influences'][:3]:
            print(f"     - Neuron {influence['neuron_id']} ({influence['concept_type']}): "
                  f"activation={influence['activation']:.3f}, "
                  f"contribution={influence['contribution']:.3f}, "
                  f"confidence={influence['confidence']:.3f}")
    
    if explanation['reasoning_path']:
        print("   → Reasoning path:")
        for step in explanation['reasoning_path'][:3]:
            print(f"     - Depth {step['depth']}: Neuron {step['neuron_id']} ({step['concept_type']}) "
                  f"contributed {step['contribution']:.3f} with confidence {step['confidence']:.3f}")
    
    # 7. Network analysis
    print("\n7. Network Analysis...")
    
    network_info = network.get_network_info()
    
    print(f"   Network state after {network_info['timestep']} timesteps:")
    print(f"   → Average forward pass time: {network_info['performance']['avg_forward_time']:.4f}s")
    print(f"   → Memory usage: {network_info['performance']['memory_usage']} states stored")
    print(f"   → Causal cache size: {network_info['performance']['causal_cache_size']} entries")
    
    # Causal cache statistics
    causal_stats = network_info['causal_stats']
    print(f"   → Cache hit rate: {causal_stats['hit_rate']:.2%}")
    print(f"   → Neurons with history: {causal_stats['neurons_with_history']}")
    
    # Attention statistics
    attention_stats = network_info['attention_stats']
    if attention_stats:
        selection_stats = attention_stats.get('selection_stats', {})
        print(f"   → Average attention candidates: {selection_stats.get('avg_candidates', 0):.1f}")
        print(f"   → Average k selected: {selection_stats.get('avg_k_used', 0):.1f}")
        print(f"   → Total attention selections: {selection_stats.get('total_selections', 0)}")
    
    # 8. Demonstrate network adaptation
    print("\n8. Demonstrating Network Adaptation...")
    
    print("   Training network with repeated patterns...")
    
    # Create training pattern
    pattern_a = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    pattern_b = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    
    # Train for several iterations
    initial_output_a = network.forward(pattern_a)['outputs']
    initial_output_b = network.forward(pattern_b)['outputs']
    
    for epoch in range(10):
        network.forward(pattern_a)
        network.forward(pattern_b)
    
    final_output_a = network.forward(pattern_a)['outputs']
    final_output_b = network.forward(pattern_b)['outputs']
    
    print(f"   Pattern A outputs - Initial: {initial_output_a}")
    print(f"   Pattern A outputs - Final:   {final_output_a}")
    print(f"   Pattern B outputs - Initial: {initial_output_b}")
    print(f"   Pattern B outputs - Final:   {final_output_b}")
    
    # Calculate adaptation (outputs should become more distinctive)
    initial_difference = np.mean(np.abs(initial_output_a - initial_output_b))
    final_difference = np.mean(np.abs(final_output_a - final_output_b))
    
    print(f"   Output difference - Initial: {initial_difference:.4f}")
    print(f"   Output difference - Final:   {final_difference:.4f}")
    
    if final_difference > initial_difference:
        print("   ✓ Network adapted: outputs became more distinctive!")
    else:
        print("   → Network behavior: outputs remained similar")
    
    # 9. Summary
    print("\n9. Summary")
    print("=" * 50)
    print("✓ Successfully created AGI network")
    print("✓ Performed recursive forward passes with attention")
    print("✓ Demonstrated masked neuron prediction")
    print("✓ Generated causal explanations")
    print("✓ Analyzed network performance and statistics")
    print("✓ Showed network adaptation capabilities")
    
    final_info = network.get_network_info()
    print(f"\nFinal network state:")
    print(f"• Total forward passes: {final_info['timestep']}")
    print(f"• Total neurons: {final_info['num_neurons']}")
    print(f"• Causal relationships tracked: {final_info['causal_stats']['total_entries']}")
    print(f"• Attention pairs learned: {final_info['attention_stats'].get('success_feedback_pairs', 0)}")
    
    print("\n🎉 AGI-Formula basic usage demonstration complete!")
    print("\nNext steps:")
    print("• Explore concept composition features")
    print("• Try self-modification capabilities")
    print("• Experiment with different network configurations")
    print("• Implement custom training loops")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("❌ Import error - please install AGI-Formula first:")
        print("   pip install -e .")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()