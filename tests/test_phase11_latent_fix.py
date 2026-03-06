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
    
    request = VideoGenerationRequest(
        prompt="test prompt",
        width=512,
        height=512,
        num_frames=14
    )
    
    # Previous latent with WRONG frame count (e.g. 25)
    wrong_latent = torch.randn(1, 25, 4, 64, 64)
    
    with patch.object(engine, 'pipe') as mock_pipe:
        # Mocking the two-pass behavior:
        # 1st pass (return_latents=False) -> returns output
        # 2nd pass (return_latents=True) -> returns (output, latents)
        mock_output = MagicMock()
        mock_output.frames = [[Image.new("RGB", (512, 512))]]
        mock_latents = torch.randn(1, 14, 4, 64, 64)
        
        def pipe_side_effect(*args, **kwargs):
            if kwargs.get('return_latents', False):
                return mock_output, mock_latents
            return mock_output
            
        mock_pipe.side_effect = pipe_side_effect
        
        # This will call the pipe twice in _generate_video
        engine._generate_video(request, Image.new("RGB", (512, 512)), seed=42, num_frames=14, previous_latent=wrong_latent)
        
        # Inspect first call's kwargs
        first_call_kwargs = mock_pipe.call_args_list[0].kwargs
        assert first_call_kwargs['latents'] is None
        assert first_call_kwargs['num_frames'] == 14

def test_latent_propagation_real_injection(engine):
    engine.pipe = MagicMock()
    
    request = VideoGenerationRequest(
        prompt="test prompt",
        width=512,
        height=512,
        num_frames=14
    )
    
    # Correct latent shape
    correct_latent = torch.randn(1, 14, 4, 64, 64)
    
    with patch.object(engine, 'pipe') as mock_pipe:
        mock_output = MagicMock()
        mock_output.frames = [[Image.new("RGB", (512, 512))]]
        mock_latents = torch.randn(1, 14, 4, 64, 64)
        
        def pipe_side_effect(*args, **kwargs):
            if kwargs.get('return_latents', False):
                return mock_output, mock_latents
            return mock_output
            
        mock_pipe.side_effect = pipe_side_effect
        
        engine._generate_video(request, Image.new("RGB", (512, 512)), seed=42, num_frames=14, previous_latent=correct_latent)
        
        # Inspect first call's kwargs
        first_call_kwargs = mock_pipe.call_args_list[0].kwargs
        assert first_call_kwargs['latents'] is not None
        assert torch.equal(first_call_kwargs['latents'], correct_latent)
