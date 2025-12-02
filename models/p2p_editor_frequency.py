"""
P2P Editor with Frequency-Preserving Inversion

This is a specialized version of P2PEditor that uses frequency-preserving inversion
instead of standard null-text inversion. It focuses on demonstrating the effect of
frequency domain regularization.

Key difference from standard P2PEditor:
- Uses FrequencyPreservingInversion instead of NullInversion
- Does NOT use adaptive num_inner_steps (to isolate the frequency effect)
- Simplified to focus on the frequency preservation innovation
"""

from models.p2p.scheduler_dev import DDIMSchedulerDev
from models.frequency_preserving_inversion import FrequencyPreservingInversion
from models.p2p.attention_control import EmptyControl, AttentionStore, make_controller
from models.p2p.p2p_guidance_forward import p2p_guidance_forward
from diffusers import StableDiffusionPipeline
from utils.utils import load_512, latent2image, txt_draw
from PIL import Image
import numpy as np


class P2PEditorFrequency:
    """
    P2P Editor with Frequency-Preserving Inversion

    Uses frequency domain regularization to better preserve image structure
    during the inversion process.
    """

    def __init__(self, device, num_ddim_steps=50, freq_weight=0.3):
        """
        Args:
            device: Device to run on ('cuda' or 'cpu')
            num_ddim_steps: Number of DDIM steps
            freq_weight: Weight for frequency preservation (0-1)
                        Higher = more structure preservation
                        Lower = more flexibility for editing
        """
        self.device = device
        self.num_ddim_steps = num_ddim_steps
        self.freq_weight = freq_weight

        print(f"\n{'='*80}")
        print("Initializing P2P Editor with Frequency-Preserving Inversion")
        print(f"{'='*80}")
        print(f"Device: {device}")
        print(f"DDIM steps: {num_ddim_steps}")
        print(f"Frequency preservation weight: {freq_weight}")
        print(f"{'='*80}\n")

        # Initialize model
        self.scheduler = DDIMSchedulerDev(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )

        self.ldm_stable = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            scheduler=self.scheduler
        ).to(device)

        self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)

        print("✓ Model loaded successfully\n")

    def edit_image(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        #我这里改成0，只比较frequency这个模块的差异
        num_inner_steps=0,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        """
        Edit an image using frequency-preserving P2P

        Args:
            image_path: Path to input image
            prompt_src: Source prompt
            prompt_tar: Target prompt
            num_inner_steps: Number of optimization steps (fixed, not adaptive)
            guidance_scale: CFG guidance scale
            cross_replace_steps: Cross-attention replacement ratio
            self_replace_steps: Self-attention replacement ratio
            blend_word: Optional blending configuration
            eq_params: Optional equalizer parameters
            is_replace_controller: Whether to use replace controller

        Returns:
            panel_image: Visualization panel (prompt, original, reconstruction, edited)
            edited_image: Edited image only
        """
        # Load image
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        print(f"\n{'='*80}")
        print("EDITING CONFIGURATION")
        print(f"{'='*80}")
        print(f"Source prompt: {prompt_src}")
        print(f"Target prompt: {prompt_tar}")
        print(f"Num inner steps: {num_inner_steps} (FIXED - not adaptive)")
        print(f"Guidance scale: {guidance_scale}")
        print(f"Cross replace steps: {cross_replace_steps}")
        print(f"Self replace steps: {self_replace_steps}")
        print(f"{'='*80}\n")

        # ⭐ Frequency-preserving inversion
        freq_inversion = FrequencyPreservingInversion(
            model=self.ldm_stable,
            num_ddim_steps=self.num_ddim_steps,
            freq_weight=self.freq_weight
        )

        _, _, x_stars, uncond_embeddings = freq_inversion.invert(
            image_gt=image_gt,
            prompt=prompt_src,
            guidance_scale=guidance_scale,
            num_inner_steps=num_inner_steps
        )

        x_t = x_stars[-1]

        # Reconstruction (to verify inversion quality)
        print("Reconstructing image to verify inversion quality...")
        controller = AttentionStore()
        reconstruct_latent, x_t = p2p_guidance_forward(
            model=self.ldm_stable,
            prompt=[prompt_src],
            controller=controller,
            latent=x_t,
            num_inference_steps=self.num_ddim_steps,
            guidance_scale=guidance_scale,
            generator=None,
            uncond_embeddings=uncond_embeddings
        )

        reconstruct_image = latent2image(
            model=self.ldm_stable.vae,
            latents=reconstruct_latent
        )[0]

        # Create instruction text
        image_instruct = txt_draw(
            f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}"
        )

        # ⭐ P2P Editing
        print(f"\n{'='*80}")
        print("Performing P2P editing...")
        print(f"{'='*80}\n")

        controller = make_controller(
            pipeline=self.ldm_stable,
            prompts=prompts,
            is_replace_controller=is_replace_controller,
            cross_replace_steps={'default_': cross_replace_steps},
            self_replace_steps=self_replace_steps,
            blend_words=blend_word,
            equilizer_params=eq_params,
            num_ddim_steps=self.num_ddim_steps,
            device=self.device
        )

        latents, _ = p2p_guidance_forward(
            model=self.ldm_stable,
            prompt=prompts,
            controller=controller,
            latent=x_t,
            num_inference_steps=self.num_ddim_steps,
            guidance_scale=guidance_scale,
            generator=None,
            uncond_embeddings=uncond_embeddings
        )

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        # Extract edited image (last one)
        edited_np = images[-1]
        edited_image = Image.fromarray(edited_np)

        # Create visualization panel
        panel_np = np.concatenate(
            (image_instruct, image_gt, reconstruct_image, edited_np),
            axis=1
        )
        panel_image = Image.fromarray(panel_np)

        print(f"\n{'='*80}")
        print("✓ Editing completed successfully")
        print(f"{'='*80}\n")

        return panel_image, edited_image


    def __call__(self, *args, **kwargs):
        """Allow the editor to be called directly"""
        return self.edit_image(*args, **kwargs)


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("P2P Editor with Frequency-Preserving Inversion - Demo")
    print("=" * 80)

    # Initialize editor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    editor = P2PEditorFrequency(
        device=device,
        num_ddim_steps=50,
        freq_weight=0.3  # Adjust this to control structure preservation
    )

    # Example usage
    panel, edited = editor.edit_image(
        image_path="path/to/your/image.jpg",
        prompt_src="a photo of a cat",
        prompt_tar="a photo of a dog",
        num_inner_steps=10  # Fixed value - not adaptive
    )

    # Save results
    panel.save("frequency_result_panel.png")
    edited.save("frequency_result_edited.png")

    print("\n✓ Demo completed!")
