from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

class IPAdapterManager:
    """
    Manages IP-Adapter loading and application for Character Identity consistency.
    """
    def __init__(self, device: str):
        self.device = device
        self.adapter_loaded = False

    def load(self, pipe) -> None:
        """
        Attaches IP-Adapter weights to the provided Diffusers pipeline.
        Only loads once to prevent redundant disk/VRAM churn.
        """
        if self.adapter_loaded:
            return

        LOG.info("Loading IP-Adapter (h94/IP-Adapter sdxl_models)...")
        pipe.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder="sdxl_models", 
            weight_name="ip-adapter_sdxl.bin"
        )
        pipe.set_ip_adapter_scale(0.6)
        
        # Further optimize VRAM by enabling CPU offload for the entire pipeline
        # (including the newly added IP-Adapter layers)
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()

        self.adapter_loaded = True

    def apply(self, pipe, reference_image) -> dict:
        """
        Returns the pipeline kwargs required to use the IP-Adapter.
        Does NOT manually extract embeddings (Diffusers handles this internally).

        Returns:
            dict: Kwargs to splat into the pipeline `__call__`
        """
        if not self.adapter_loaded or reference_image is None:
            return {}

        LOG.debug("Applying IP-Adapter conditioning image.")
        return {"ip_adapter_image": reference_image}
