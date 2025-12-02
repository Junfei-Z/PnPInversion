"""
Frequency-Preserving Null-Text Inversion

This module extends null-text inversion by adding frequency domain regularization
to better preserve image structure during the inversion process.

Key innovation:
- Extracts high-frequency components from latent space (corresponds to edges/structure)
- Adds regularization during null-text optimization to preserve these frequencies
- Results in better structure preservation without additional decode operations
"""

import torch
import torch.nn.functional as nnf
from torch.optim.adam import Adam
from models.p2p.attention_control import register_attention_control
from utils.utils import image2latent, latent2image
import numpy as np


class FrequencyPreservingInversion:
    """
    Null-text inversion with frequency domain regularization

    Enhances standard null-text inversion by preserving high-frequency components
    in latent space, which correspond to image structure and edges.
    """

    def __init__(self, model, num_ddim_steps, freq_weight=0.3):
        """
        Args:
            model: StableDiffusionPipeline
            num_ddim_steps: Number of DDIM steps
            freq_weight: Weight for frequency preservation loss (0-1)
        """
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps = num_ddim_steps
        self.freq_weight = freq_weight

    @property
    def scheduler(self):
        return self.model.scheduler

    def prev_step(self, model_output, timestep: int, sample):
        """DDIM backward step (denoising)"""
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output, timestep: int, sample):
        """DDIM forward step (add noise)"""
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        """Predict noise using UNet"""
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, guidance_scale, is_forward=True, context=None):
        """Predict noise with classifier-free guidance"""
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        """Initialize text embeddings"""
        uncond_input = self.model.tokenizer(
            [""],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        """DDIM inversion loop"""
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def ddim_inversion(self, image):
        """Perform DDIM inversion"""
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def extract_high_frequency(self, latent):
        """
        Extract high-frequency components from latent using FFT

        High frequencies correspond to edges and fine details.
        We use a high-pass filter in frequency domain.
        """
        # Apply FFT to each channel
        fft_latent = torch.fft.fft2(latent)

        # Create high-pass filter mask
        b, c, h, w = latent.shape
        mask = torch.zeros((h, w), device=latent.device)

        center_h, center_w = h // 2, w // 2
        radius = int(min(h, w) * 0.3)  # Keep outer 70% frequencies (high freq)

        # Create circular high-pass mask
        y, x = torch.meshgrid(
            torch.arange(h, device=latent.device),
            torch.arange(w, device=latent.device),
            indexing='ij'
        )

        distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
        mask = (distance > radius).float()

        # Apply mask to FFT
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
        high_freq_fft = fft_latent * mask

        return high_freq_fft

    def compute_frequency_loss(self, latent_current, latent_target):
        """
        Compute loss between high-frequency components

        This encourages preserving image structure during optimization.
        """
        # Extract high-frequency components
        high_freq_current = self.extract_high_frequency(latent_current)
        high_freq_target = self.extract_high_frequency(latent_target)

        # Compute L2 loss in frequency domain
        freq_loss = nnf.mse_loss(
            torch.abs(high_freq_current),
            torch.abs(high_freq_target)
        )

        return freq_loss

    def null_optimization_with_frequency(self, latents, num_inner_steps, epsilon, guidance_scale):
        """
        Null-text optimization with frequency preservation

        This is the core innovation: we optimize uncond_embeddings while
        preserving high-frequency structure in latent space.
        """
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]

        print(f"\n{'='*80}")
        print("FREQUENCY-PRESERVING INVERSION")
        print(f"{'='*80}")
        print(f"Frequency preservation weight: {self.freq_weight}")
        print(f"Number of DDIM steps: {self.num_ddim_steps}")
        print(f"Inner optimization steps: {num_inner_steps}")
        print(f"{'='*80}\n")

        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            t = self.model.scheduler.timesteps[i]

            if num_inner_steps != 0:
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                latent_prev = latents[len(latents) - i - 2]

                # Get conditional noise prediction (fixed)
                with torch.no_grad():
                    noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)

                # Inner optimization loop
                for j in range(num_inner_steps):
                    # Predict noise with current uncond_embeddings
                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # Reconstruct previous latent
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)

                    # Standard reconstruction loss
                    recon_loss = nnf.mse_loss(latents_prev_rec, latent_prev)

                    # ⭐ Frequency preservation loss (KEY INNOVATION)
                    freq_loss = self.compute_frequency_loss(latents_prev_rec, latent_prev)

                    # Combined loss
                    total_loss = recon_loss + self.freq_weight * freq_loss

                    # Optimize
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # Early stopping
                    recon_loss_item = recon_loss.item()
                    if recon_loss_item < epsilon + i * 2e-5:
                        break

                # Print progress every 10 steps
                if i % 10 == 0:
                    print(f"Step {i}/{self.num_ddim_steps}: "
                          f"Recon Loss = {recon_loss_item:.6f}, "
                          f"Freq Loss = {freq_loss.item():.6f}, "
                          f"Total = {total_loss.item():.6f}")

            # Store optimized uncond_embeddings
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())

            # Move to next timestep
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, guidance_scale, False, context)

        print(f"\n{'='*80}")
        print("✓ Frequency-preserving inversion completed")
        print(f"{'='*80}\n")

        return uncond_embeddings_list

    def invert(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        """
        Perform frequency-preserving inversion

        Args:
            image_gt: Input image [512, 512, 3]
            prompt: Source prompt
            guidance_scale: Guidance scale for CFG
            num_inner_steps: Number of optimization steps per timestep
            early_stop_epsilon: Early stopping threshold

        Returns:
            image_gt: Original image
            image_rec: VAE reconstruction
            ddim_latents: Inverted latent trajectory
            uncond_embeddings: Optimized unconditional embeddings
        """
        # Initialize prompt embeddings
        self.init_prompt(prompt)
        register_attention_control(self.model, None)

        # DDIM inversion
        image_rec, ddim_latents = self.ddim_inversion(image_gt)

        # Null-text optimization with frequency preservation
        uncond_embeddings = self.null_optimization_with_frequency(
            ddim_latents,
            num_inner_steps,
            early_stop_epsilon,
            guidance_scale
        )

        return image_gt, image_rec, ddim_latents, uncond_embeddings
