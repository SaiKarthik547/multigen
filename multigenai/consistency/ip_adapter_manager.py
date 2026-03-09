from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

class IPAdapterManager:
    """
    Manages IP-Adapter loading and application for Character Identity consistency.
    """
    def __init__(self, device: str):
        self.device = device
        self.adapter_loaded = False

    def load(self, pipe, model_type: str = "sdxl") -> None:
        """
        Attaches IP-Adapter weights to the provided Diffusers pipeline.
        Supports both SDXL and SD1.5 (used by AnimateDiff).
        """
        if self.adapter_loaded and getattr(self, "_current_model_type", None) == model_type:
            return

        if model_type == "sdxl":
            LOG.info("Loading SDXL IP-Adapter (sdxl_models/ip-adapter_sdxl.bin)...")
            pipe.load_ip_adapter(
                "h94/IP-Adapter", 
                subfolder="sdxl_models", 
                weight_name="ip-adapter_sdxl.bin"
            )
        elif model_type == "sd15":
            LOG.info("Loading SD1.5 IP-Adapter (models/ip-adapter_sd15.bin)...")
            pipe.load_ip_adapter(
                "h94/IP-Adapter", 
                subfolder="models", 
                weight_name="ip-adapter_sd15.bin"
            )
        else:
            raise ValueError(f"IPAdapterManager: Unsupported model_type '{model_type}'")

        pipe.set_ip_adapter_scale(0.6)
        self.adapter_loaded = True
        self._current_model_type = model_type

    def apply(self, pipe, reference_image) -> dict:
        """
        Returns the pipeline kwargs required to use the IP-Adapter.
        Does NOT manually extract embeddings (Diffusers handles this internally).

        Returns:
            dict: Kwargs to splat into the pipeline `__call__`
        """
        from PIL import Image as PILImage
        
        if not self.adapter_loaded or reference_image is None:
            return {}

        LOG.debug("Applying IP-Adapter conditioning image.")
        
        # --- Normalize IP-Adapter image input ---
        if isinstance(reference_image, (list, tuple)):
            reference_image = reference_image[0]
            
        if not isinstance(reference_image, PILImage.Image):
             import numpy as np
             import torch
             
             if isinstance(reference_image, torch.Tensor):
                 reference_image = reference_image.detach().cpu().numpy()
                 
             if isinstance(reference_image, np.ndarray):
                 if reference_image.dtype == np.float32 or reference_image.dtype == np.float16:
                     reference_image = (reference_image * 255.0).clip(0, 255)
                 reference_image = PILImage.fromarray(np.uint8(reference_image))
             else:
                 # Fallback for unexpected types
                 LOG.warning(f"IPAdapterManager: reference_image is {type(reference_image)}, attempting cast.")
                 reference_image = PILImage.fromarray(np.uint8(reference_image))
             
        return {"ip_adapter_image": reference_image}
