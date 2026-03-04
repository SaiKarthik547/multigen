import torch
from multigenai.models.rife.IFNet_2R import IFNet
from pathlib import Path

class RIFEModel:
    def __init__(self, device="cuda"):
        self.device = "cpu" if device.lower() == "directml" else (device if torch.cuda.is_available() else "cpu")
        self.model = IFNet().to(self.device)

        weight_path = Path(__file__).parent / "flownet.pkl"
        checkpoint = torch.load(weight_path, map_location=self.device)

        state = checkpoint.get("model", checkpoint)
        
        # Strip DataParallel "module." prefix if present
        state = {k.replace("module.", ""): v for k, v in state.items()}

        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    @torch.no_grad()
    def interpolate(self, frame1, frame2):
        inp = torch.cat([frame1, frame2], 1)
        flow_list, mask, merged = self.model(inp)
        return merged[2]
