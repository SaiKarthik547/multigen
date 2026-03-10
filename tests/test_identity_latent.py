import pytest
import torch
from unittest.mock import MagicMock
from PIL import Image
from multigenai.identity.identity_latent_encoder import IdentityLatentEncoder

class TestIdentityLatentEncoder:
    def test_encoder_returns_correct_shape(self):
        # Arrange
        mock_pipe = MagicMock()
        mock_pipe.device = torch.device("cpu")
        mock_pipe.dtype = torch.float32
        mock_pipe.vae.config.scaling_factor = 0.18215
        
        # Mock the image processor output
        mock_tensor = torch.randn(1, 3, 512, 512)
        mock_pipe.image_processor.preprocess.return_value = mock_tensor
        
        # Mock VAE encoding response (Distribution output with a sample method)
        mock_dist = MagicMock()
        # SD1.5/SDXL VAE reduces spatial dims by factor of 8, and channels out is usually 4
        expected_shape = (1, 4, 64, 64)
        mock_latent = torch.randn(*expected_shape)
        mock_dist.latent_dist.sample.return_value = mock_latent
        mock_pipe.vae.encode.return_value = mock_dist
        
        encoder = IdentityLatentEncoder()
        dummy_img = Image.new("RGB", (512, 512), color="black")
        
        # Act
        identity_latent = encoder.encode(mock_pipe, dummy_img)
        
        # Assert (13. Verification Plan Review check)
        assert identity_latent is not None
        assert identity_latent.ndim == 4, f"Expected 4 dimensions [1, C, H, W], got {identity_latent.ndim}"
        assert identity_latent.shape == expected_shape
        
        # Verify scaling factor applied (mock latent * 0.18215)
        # Using allclose to ignore minor floating point drift
        assert torch.allclose(identity_latent, mock_latent * 0.18215)
