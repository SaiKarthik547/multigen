import pytest
import torch
from unittest.mock import MagicMock
from PIL import Image
from multigenai.temporal.trajectory_encoder import TrajectoryEncoder

class TestTrajectoryEncoder:
    def test_encode_success(self):
        pipe = MagicMock()
        pipe.device = torch.device('cpu')
        pipe._execution_device = torch.device('cpu')
        pipe.dtype = torch.float32
        pipe.vae.config.scaling_factor = 0.5
        
        processor_mock = MagicMock()
        pipe.image_processor = processor_mock
        img_tensor = torch.zeros(1, 3, 512, 512)
        processor_mock.preprocess.return_value = img_tensor
        
        dist_mock = MagicMock()
        dist_mock.latent_dist.sample.return_value = torch.ones(1, 4, 64, 64)
        pipe.vae.encode.return_value = dist_mock
        
        encoder = TrajectoryEncoder()
        dummy_img = Image.new("RGB", (512, 512), "black")
        
        result = encoder.encode(pipe, dummy_img)
        assert result is not None
        assert result.shape == (1, 4, 64, 64)
        assert torch.allclose(result, torch.ones(1, 4, 64, 64) * 0.5)
