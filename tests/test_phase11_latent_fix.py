import pytest
import torch
from unittest.mock import MagicMock, patch
from multigenai.engines.video_engine.engine import VideoEngine
from multigenai.llm.schema_validator import VideoGenerationRequest
from PIL import Image

@pytest.fixture
def engine(tmp_path):
    from multigenai.core.execution_context import ExecutionContext
    ctx = MagicMock(spec=ExecutionContext)
    ctx.device = "cpu"
    ctx.settings = MagicMock()
    ctx.settings.output_dir = str(tmp_path)
    ctx.behaviour = MagicMock()
    ctx.behaviour.auto_unload_after_gen = False
    return VideoEngine(ctx)

def test_latent_propagation_shape_guard(engine):
    # Mock pipe to avoid loading the model
    engine.pipe = MagicMock()
    engine.pipe.unet = MagicMock()
    engine.pipe.unet.dtype = torch.float32
    
    request = VideoGenerationRequest(
        prompt="test prompt",
        width=512,
        height=512,
        num_frames=14
    )
    
    # Previous latent with WRONG frame count (e.g. 25)
    wrong_latent = torch.randn(1, 25, 4, 64, 64)
    
    with patch.object(engine, 'pipe') as mock_pipe:
        mock_pipe.unet = MagicMock()
        mock_pipe.unet.dtype = torch.float32
        # Mocking the single-pass behavior:
        # returns (output, latents)
        mock_output = MagicMock()
        mock_output.frames = [[Image.new("RGB", (512, 512))]]
        mock_latents = torch.randn(1, 14, 4, 64, 64)
        
        mock_pipe.return_value = (mock_output, mock_latents)
        
        # This will call the pipe once in _generate_video
        engine._generate_video(request, Image.new("RGB", (512, 512)), seed=42, num_frames=14, previous_latent=wrong_latent)
        
        # Inspect call's kwargs
        call_kwargs = mock_pipe.call_args.kwargs
        # Shape guard should have caught the mismatch and passed None to pipe
        assert call_kwargs['latents'] is None
        assert call_kwargs['num_frames'] == 14
        assert call_kwargs['return_latents'] is True

def test_latent_propagation_real_injection(engine):
    engine.pipe = MagicMock()
    engine.pipe.unet = MagicMock()
    engine.pipe.unet.dtype = torch.float32
    
    request = VideoGenerationRequest(
        prompt="test prompt",
        width=512,
        height=512,
        num_frames=14
    )
    
    # Correct latent shape
    correct_latent = torch.randn(1, 14, 4, 64, 64)
    
    with patch.object(engine, 'pipe') as mock_pipe:
        mock_pipe.unet = MagicMock()
        mock_pipe.unet.dtype = torch.float32
        mock_output = MagicMock()
        mock_output.frames = [[Image.new("RGB", (512, 512))]]
        mock_latents = torch.randn(1, 14, 4, 64, 64)
        
        mock_pipe.return_value = (mock_output, mock_latents)
        
        engine._generate_video(request, Image.new("RGB", (512, 512)), seed=42, num_frames=14, previous_latent=correct_latent)
        
        # Inspect call's kwargs
        call_kwargs = mock_pipe.call_args.kwargs
        # Should have passed the renoised latent
        assert call_kwargs['latents'] is not None
        assert call_kwargs['latents'].shape == correct_latent.shape
        # Note: torch.equal will fail here because renoising adds random noise
        assert call_kwargs['return_latents'] is True
