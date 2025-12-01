"""
Test script to verify that attention control parameters actually impact the results.
We'll test with extreme parameters to see visible differences.
"""

import os
import sys

# Run the same image with different parameters
test_configs = [
    {
        "name": "extreme_low",
        "cross": 0.1,  # Very low - allow lots of changes
        "self": 0.2,
        "output_suffix": "_extreme_low"
    },
    {
        "name": "extreme_high",
        "cross": 0.9,  # Very high - preserve structure heavily
        "self": 0.95,
        "output_suffix": "_extreme_high"
    },
    {
        "name": "baseline_fixed",
        "cross": 0.4,  # Original P2P default
        "self": 0.6,
        "output_suffix": "_baseline"
    },
    {
        "name": "adaptive",
        "cross": None,  # Will use adaptive control
        "self": None,
        "output_suffix": "_adaptive"
    }
]

print("=" * 60)
print("Parameter Impact Test")
print("=" * 60)
print("\nThis test will run the same image with 4 different parameter settings:")
print("1. Extreme Low (cross=0.1, self=0.2) - Maximum editing freedom")
print("2. Extreme High (cross=0.9, self=0.95) - Maximum preservation")
print("3. Baseline (cross=0.4, self=0.6) - Original P2P defaults")
print("4. Adaptive - Your semantic-aware control")
print("\nIf parameters work correctly, you should see DIFFERENT results!")
print("=" * 60)

print("\nTo run this test, you need to:")
print("1. Modify models/p2p_editor.py to temporarily disable adaptive control")
print("2. Or add a command-line flag to control adaptive vs fixed parameters")
print("\nSuggested modification in p2p_editor.py line 178-189:")
print("""
# Add an environment variable check
import os
USE_ADAPTIVE = os.environ.get('USE_ADAPTIVE_CONTROL', 'true').lower() == 'true'

if USE_ADAPTIVE:
    # Use adaptive control
    adaptive_cross, adaptive_self = get_adaptive_parameters(...)
    final_cross = adaptive_cross
    final_self = adaptive_self
else:
    # Use fixed parameters from arguments
    final_cross = cross_replace_steps
    final_self = self_replace_steps
    print(f"[Fixed Parameters]")
    print(f"  Cross-replace steps: {final_cross:.2f}")
    print(f"  Self-replace steps:  {final_self:.2f}")
""")

print("\nThen run:")
for config in test_configs:
    if config["cross"] is not None:
        print(f"\n# Test {config['name']}:")
        print(f"USE_ADAPTIVE_CONTROL=false python run_editing_p2p_metrics.py \\")
        print(f"  --data_path data \\")
        print(f"  --output_path ./output{config['output_suffix']} \\")
        print(f"  --edit_category_list \"6\" \\")
        print(f"  --edit_method_list ddim+p2p \\")
        print(f"  --metrics_csv metrics{config['output_suffix']}.csv")
        print(f"  # (manually set cross={config['cross']}, self={config['self']} in code)")
    else:
        print(f"\n# Test {config['name']}:")
        print(f"USE_ADAPTIVE_CONTROL=true python run_editing_p2p_metrics.py \\")
        print(f"  --data_path data \\")
        print(f"  --output_path ./output{config['output_suffix']} \\")
        print(f"  --edit_category_list \"6\" \\")
        print(f"  --edit_method_list ddim+p2p \\")
        print(f"  --metrics_csv metrics{config['output_suffix']}.csv")
