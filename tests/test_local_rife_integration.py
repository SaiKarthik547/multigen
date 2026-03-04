import pytest
from PIL import Image
from multigenai.engines.interpolation_engine.engine import InterpolationEngine

@pytest.fixture
def mock_frames():
    # Create two small 128x128 images for testing to avoid OOM
    img1 = Image.linear_gradient("L").convert("RGB").resize((128, 128))
    img2 = Image.linear_gradient("L").transpose(Image.FLIP_TOP_BOTTOM).convert("RGB").resize((128, 128))
    return [img1, img2]

def test_local_rife_interpolation_factor_2(mock_frames):
    """Test that the local RIFE model properly extends 2 frames to 3 frames."""
    torch = pytest.importorskip("torch")
    
    class DummyCtx:
        device = "cpu"
        
    engine = InterpolationEngine(DummyCtx())
    
    # Run interpolation
    interpolated = engine.interpolate(mock_frames, factor=2)
    
    assert len(interpolated) == 3, f"Expected 3 frames for factor 2 on 2 input frames, got {len(interpolated)}"
    assert interpolated[0] is mock_frames[0]
    assert interpolated[2] is mock_frames[1]
    
    # Check that the intermediate frame has correct dimensions
    assert interpolated[1].size == (128, 128)

def test_local_rife_interpolation_factor_4(mock_frames):
    """Test recursive interpolation for factor 4 -> 5 frames out of 2 inputs."""
    torch = pytest.importorskip("torch")
    
    class DummyCtx:
        device = "cpu"
        
    engine = InterpolationEngine(DummyCtx())
    
    interpolated = engine.interpolate(mock_frames, factor=4)
    # n + (n-1)*(factor-1) -> 2 + 1*3 = 5 frames
    assert len(interpolated) == 5, f"Expected 5 frames for factor 4, got {len(interpolated)}"
    assert interpolated[0] is mock_frames[0]
    assert interpolated[4] is mock_frames[1]
