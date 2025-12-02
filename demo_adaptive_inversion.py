"""
Demo: Adaptive Inversion Strategy

Demonstrates how num_inner_steps is intelligently adjusted based on
the semantic distance between source and target prompts.
"""

from models.adaptive_inversion import AdaptiveInversionController


def main():
    print("\n" + "🎯"*40)
    print("ADAPTIVE INVERSION DEMO")
    print("Intelligently adjusting num_inner_steps based on semantic analysis")
    print("🎯"*40 + "\n")

    # Initialize controller
    controller = AdaptiveInversionController(device='cpu')  # Use 'cuda' if available

    # Test cases with different levels of semantic distance
    test_cases = [
        # (source_prompt, target_prompt, expected_difficulty)
        ("a photo of a cat", "a photo of a dog", "HARD - Object Replacement"),
        ("a smiling person", "an angry person", "MEDIUM - Attribute Change"),
        ("a photo of a house", "a painting of a house", "MEDIUM - Style Transfer"),
        ("a red car", "a blue car", "EASY - Color Change"),
        ("a small dog", "a big dog", "EASY - Size Change"),
        ("a beautiful sunset", "a beautiful sunrise", "MEDIUM - Concept Change"),
    ]

    print("\n" + "="*80)
    print("COMPARISON OF DIFFERENT STRATEGIES")
    print("="*80 + "\n")

    for src, tar, difficulty in test_cases:
        print(f"{'='*80}")
        print(f"Difficulty: {difficulty}")
        print(f"Source: {src}")
        print(f"Target: {tar}")
        print(f"{'-'*80}")

        # Get recommendations for all strategies
        recommendations = controller.get_recommendations(src, tar)

        print(f"Recommended num_inner_steps:")
        print(f"  Conservative: {recommendations['conservative']:2d} (safer, slower)")
        print(f"  Moderate:     {recommendations['moderate']:2d} (balanced) ⭐")
        print(f"  Aggressive:   {recommendations['aggressive']:2d} (faster, may lose quality)")
        print(f"{'='*80}\n")

    # Detailed analysis for one example
    print("\n" + "🔍"*40)
    print("DETAILED ANALYSIS EXAMPLE")
    print("🔍"*40 + "\n")

    example_src = "a photo of a cat sitting on a chair"
    example_tar = "a photo of a dog standing on a table"

    num_steps = controller.calculate_num_inner_steps(
        example_src,
        example_tar,
        strategy='moderate',
        verbose=True
    )

    print("\n" + "💡"*40)
    print("INTERPRETATION")
    print("💡"*40)
    print(f"""
This analysis shows:

1. CLIP Similarity: Measures semantic closeness
   - High (>0.85): Very similar prompts → Few steps needed
   - Low (<0.7): Very different prompts → More steps needed

2. Token Change Ratio: Tracks word-level changes
   - Low: Only a few words changed
   - High: Many words changed

3. Edit Type Detection:
   - Object Replacement: Hardest (needs most steps)
   - Attribute Change: Medium difficulty
   - Style Transfer: Medium difficulty
   - Minor Edit: Easiest (needs fewest steps)

4. Final Decision:
   - Combines all factors intelligently
   - Provides optimal num_inner_steps: {num_steps}

This ensures better inversion quality without wasting computation!
    """)

    print("\n" + "✅"*40)
    print("Key Benefits:")
    print("✅"*40)
    print("""
✓ Automatic: No manual tuning required
✓ Intelligent: Analyzes semantic distance and edit type
✓ Efficient: Uses minimal steps for easy edits, more for hard ones
✓ Quality: Better inversion → Better editing results
✓ Flexible: Three strategies (conservative/moderate/aggressive)
    """)

    print("\n" + "🚀"*40)
    print("USAGE IN YOUR CODE")
    print("🚀"*40)
    print("""
# Original (fixed num_inner_steps):
editor = P2PEditor(...)
result = editor(
    edit_method="ddim+p2p",
    prompt_src="a cat",
    prompt_tar="a dog",
    # num_inner_steps was always 0
)

# Now (adaptive num_inner_steps):
editor = P2PEditor(...)
result = editor(
    edit_method="ddim+p2p",
    prompt_src="a cat",
    prompt_tar="a dog",
    # num_inner_steps automatically adjusted!
    # Will show analysis in output
)

The adaptive system is already integrated into p2p_editor.py!
    """)


if __name__ == "__main__":
    main()
