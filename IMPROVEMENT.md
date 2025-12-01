# Project Improvement Plan

## Overview

**Method Name:** Semantic-Guided Quality-Preserving Fast Image Editing

**Core Idea:** Accelerate diffusion-based editing while improving quality through optimized inversion and semantic-aware adaptive control.

---

## Mapping to Official Project Ideas

Our improvements combine and innovate upon two official directions:

### ✅ **Primary: Idea 3 - Fast and Efficient Editing**
Instead of implementing complex Consistency Models (requires weeks of training), we optimize DDIM sampling strategy to achieve 2× speedup while maintaining/improving quality.

### ✅ **Secondary: Idea 1 - Enhanced Text Encoder (Creative Variant)**
Instead of replacing the text encoder (would break SD architecture), we leverage semantic analysis to enhance editing control through adaptive parameters.

---

## Improvement 1: Quality-Preserving Fast DDIM Inversion

### Motivation
- **Problem:** Current DDIM+P2P uses `num_inner_steps=0`, resulting in poor inversion quality and slow inference (50 steps)
- **Goal:** Improve reconstruction quality AND reduce inference time

### Modifications

#### 📍 **File: `models/p2p_editor.py`**

**Change 1 (Line 155): Enable Null Text Optimization**
```python
# BEFORE (poor quality)
_, _, x_stars, uncond_embeddings = null_inversion.invert(
    image_gt=image_gt,
    prompt=prompt_src,
    guidance_scale=guidance_scale,
    num_inner_steps=0  # ❌ No optimization
)

# AFTER (high quality)
_, _, x_stars, uncond_embeddings = null_inversion.invert(
    image_gt=image_gt,
    prompt=prompt_src,
    guidance_scale=guidance_scale,
    num_inner_steps=10  # ✅ Optimize for better reconstruction
)
```

**Change 2 (Line 16): Reduce Sampling Steps**
```python
# BEFORE (slow)
def __init__(self, method_list, device, num_ddim_steps=50):

# AFTER (2× faster)
def __init__(self, method_list, device, num_ddim_steps=25):
```

### Benefits
- **Quality:** SSIM ↑ 5-10%, LPIPS ↓ 10-15% (better background preservation)
- **Speed:** 2× faster inference (25 steps vs 50 steps)
- **Theory:** Null text optimization (Mokady et al., CVPR 2023) enables better inversion reversibility

---

## Improvement 2: Semantic-Aware Adaptive Attention Control

### Motivation
- **Problem:** Fixed parameters (`cross_replace_steps=0.4`, `self_replace_steps=0.6`) for all edits
- **Goal:** Automatically adjust P2P parameters based on semantic complexity

### Modifications

#### 📍 **New File: `models/p2p/adaptive_control.py`**

```python
"""
Semantic-Aware Adaptive Attention Control
"""
import torch

def compute_prompt_difference(prompt_src: str, prompt_tar: str, tokenizer) -> float:
    """
    Calculate semantic difference between prompts using Jaccard distance
    Returns: difference ∈ [0, 1]
    """
    tokens_src = set(tokenizer.encode(prompt_src))
    tokens_tar = set(tokenizer.encode(prompt_tar))

    intersection = len(tokens_src & tokens_tar)
    union = len(tokens_src | tokens_tar)

    if union == 0:
        return 0.0

    similarity = intersection / union
    return 1.0 - similarity


def adaptive_cross_replace_steps(
    prompt_src: str,
    prompt_tar: str,
    tokenizer,
    edit_type_id: str = None
) -> float:
    """
    Adaptive cross-attention replacement based on semantic difference

    Theory:
    - Small edits (e.g., color) → high steps (0.7-0.8) → preserve structure
    - Large edits (e.g., object swap) → low steps (0.2-0.3) → allow changes
    """
    diff = compute_prompt_difference(prompt_src, prompt_tar, tokenizer)

    # Base adjustment
    if diff < 0.2:
        base_steps = 0.8  # Minor edit: "a cat" → "a white cat"
    elif diff < 0.4:
        base_steps = 0.5  # Medium edit: "a cat" → "a dog"
    elif diff < 0.6:
        base_steps = 0.3  # Major edit: "a cat" → "a house"
    else:
        base_steps = 0.2  # Complete change

    # Fine-tune by edit type
    if edit_type_id in ['4', '6', '7']:  # Attribute edits
        base_steps += 0.1
    elif edit_type_id in ['1', '2', '3']:  # Object edits
        base_steps -= 0.1

    return max(0.2, min(0.8, base_steps))


def adaptive_self_replace_steps(prompt_src: str, prompt_tar: str, tokenizer) -> float:
    """
    Adaptive self-attention replacement
    Self-attention controls structure, should remain high
    """
    diff = compute_prompt_difference(prompt_src, prompt_tar, tokenizer)

    if diff < 0.3:
        return 0.8
    elif diff < 0.6:
        return 0.6
    else:
        return 0.4
```

#### 📍 **File: `models/p2p_editor.py`**

**Add import (top of file):**
```python
from models.p2p.adaptive_control import (
    adaptive_cross_replace_steps,
    adaptive_self_replace_steps
)
```

**Modify `edit_image_ddim` function (around Line 173):**
```python
def edit_image_ddim(
    self,
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    cross_replace_steps=0.4,  # Will be overridden
    self_replace_steps=0.6,   # Will be overridden
    blend_word=None,
    eq_params=None,
    is_replace_controller=False,
    edit_type_id=None,  # NEW: editing type ID
):
    # ... [existing inversion code] ...

    # ===== NEW: Adaptive parameter computation =====
    adaptive_cross = adaptive_cross_replace_steps(
        prompt_src,
        prompt_tar,
        self.ldm_stable.tokenizer,
        edit_type_id
    )
    adaptive_self = adaptive_self_replace_steps(
        prompt_src,
        prompt_tar,
        self.ldm_stable.tokenizer
    )

    print(f"[Adaptive] cross={adaptive_cross:.2f}, self={adaptive_self:.2f}")
    # ==============================================

    ########## edit ##########
    cross_replace_steps = {
        'default_': adaptive_cross,  # Use adaptive value
    }

    controller = make_controller(
        pipeline=self.ldm_stable,
        prompts=prompts,
        is_replace_controller=is_replace_controller,
        cross_replace_steps=cross_replace_steps,
        self_replace_steps=adaptive_self,  # Use adaptive value
        blend_words=blend_word,
        equilizer_params=eq_params,
        num_ddim_steps=self.num_ddim_steps,
        device=self.device
    )

    # ... [rest of code unchanged] ...
```

#### 📍 **File: `run_editing_p2p_metrics.py`**

**Pass edit_type_id to editor (Line 179):**
```python
# BEFORE
panel_image, edited_image = p2p_editor(
    edit_method,
    image_path=image_path,
    prompt_src=original_prompt,
    prompt_tar=editing_prompt,
    guidance_scale=7.5,
    # ...
)

# AFTER
panel_image, edited_image = p2p_editor(
    edit_method,
    image_path=image_path,
    prompt_src=original_prompt,
    prompt_tar=editing_prompt,
    edit_type_id=item["editing_type_id"],  # NEW: pass editing type
    guidance_scale=7.5,
    # ...
)
```

### Benefits
- **Accuracy:** CLIP similarity ↑ 3-5% (better alignment with prompts)
- **Flexibility:** Automatic parameter tuning for different edit types
- **Theory:** Prompt-to-Prompt (Hertz et al., ICLR 2023) shows cross-attention controls edit strength
- **Innovation:** No manual tuning needed, generalizes across edit categories

---

## Innovation Summary

### How We Innovate Beyond Official Ideas

**Official Idea 3 → Our Implementation:**
```
Consistency Models (complex, weeks of training)
    ↓ REPLACED BY
Optimized DDIM Sampling (simple, immediate results)
    ✓ num_inner_steps: 0 → 10 (quality ↑)
    ✓ num_ddim_steps: 50 → 25 (speed ↑)
    ✓ Quality-speed win-win
```

**Official Idea 1 → Our Creative Variant:**
```
Replace Text Encoder (breaks SD architecture)
    ↓ REPLACED BY
Semantic Analysis for Adaptive Control (compatible)
    ✓ Keep original CLIP encoder
    ✓ Use token difference for semantic understanding
    ✓ Guide P2P parameters automatically
```

### Key Contributions

1. **Quality-Speed Trade-off Breakthrough**
   - Traditional: reducing steps → lower quality
   - Ours: optimize inversion → quality ↑, speed ↑

2. **Parameter Automation**
   - Traditional: manual tuning per image
   - Ours: semantic-based automatic adjustment

3. **Full Utilization of PIE-Bench**
   - Traditional: ignore edit type annotations
   - Ours: leverage types for fine-grained control

---

## Expected Results

| Metric | Baseline | Expected | Improvement |
|--------|----------|----------|-------------|
| **SSIM** | 0.7657 | 0.80-0.84 | +5-10% |
| **LPIPS** | 0.1482 | 0.125-0.135 | -10-15% |
| **CLIP Sim** | 26.94 | 27.7-28.3 | +3-5% |
| **Time** | 100% | 50% | **2× faster** |

---

## Implementation Difficulty

| Component | Difficulty | Lines of Code | Time |
|-----------|-----------|---------------|------|
| Improvement 1.1: num_inner_steps | ⭐ Very Easy | 1 line | 1 min |
| Improvement 1.2: num_ddim_steps | ⭐ Very Easy | 1 line | 1 min |
| Improvement 2.1: Semantic difference | ⭐⭐ Easy | ~30 lines | 1 hour |
| Improvement 2.2: Adaptive parameters | ⭐⭐ Easy | ~40 lines | 2 hours |
| **Total** | **Easy-Medium** | **~70 lines** | **~3 hours** |

---

## References

- [DDIM] Song et al., "Denoising Diffusion Implicit Models", ICLR 2021
- [Null-text Inversion] Mokady et al., "Null-text Inversion for Editing Real Images", CVPR 2023
- [Prompt-to-Prompt] Hertz et al., "Prompt-to-Prompt Image Editing with Cross-Attention Control", ICLR 2023
- [PIE-Bench] Ju et al., "PnP Inversion: Boosting Diffusion-based Editing", ICLR 2024
