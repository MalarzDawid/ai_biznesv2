from __future__ import annotations

import gc
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

# Cache dictionary to hold the loaded model pipeline in memory
_pipeline_cache = {}

def clear_gpu_cache() -> None:
    """Clears the pipeline cache and empties PyTorch's CUDA cache to free up VRAM."""
    global _pipeline_cache
    _pipeline_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_cached_pipeline(model_id: str) -> Flux2KleinPipeline:
    """Retrieves or loads the FLUX.2-klein-4B pipeline in FP16."""
    global _pipeline_cache
    
    # We only use black-forest-labs/FLUX.2-klein-4B as requested
    target_model_id = "black-forest-labs/FLUX.2-klein-4B"
    
    if target_model_id in _pipeline_cache:
        return _pipeline_cache[target_model_id]
        
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. A GPU is required to run this model.")

    # Select parameters for optimization (using fp16)
    kwargs = {
        "torch_dtype": torch.float16,
        "use_safetensors": True
    }
    
    pipe = Flux2KleinPipeline.from_pretrained(
        target_model_id,
        **kwargs
    )
        
    pipe = pipe.to("cuda")
    
    _pipeline_cache[target_model_id] = pipe
    return pipe

def generate_img2img(
    model_id: str,
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    strength: float = 0.5,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 30,
    seed: int = -1,
    callback: callable = None
) -> Image.Image:
    """
    Takes an initial image, resizes it appropriately, and runs FLUX.2-klein-4B.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Image generation requires a local GPU.")
        
    # Resize image to fit model constraints (FLUX prefers 1024x1024)
    max_dim = 1024
    
    init_image = init_image.convert("RGB")
    init_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    
    # Width and height must be divisible by 16 for FLUX
    w, h = init_image.size
    w = (w // 16) * 16
    h = (h // 16) * 16
    if (w, h) != init_image.size:
        init_image = init_image.resize((w, h), Image.Resampling.LANCZOS)
        
    pipe = get_cached_pipeline(model_id)
    
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        generator = torch.Generator("cuda").manual_seed(torch.randint(0, 1000000, (1,)).item())
        
    with torch.inference_mode():
        # Flux2KleinPipeline does not support negative_prompt or strength parameters
        output = pipe(
            prompt=prompt,
            image=init_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            callback_on_step_end=callback
        )
        
    generated_image = output.images[0]
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return generated_image
