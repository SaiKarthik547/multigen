import pytest
from PIL import Image
from multigenai.consistency.scene_memory import SceneMemory, SceneState

def test_scene_memory_initialization():
    mem = SceneMemory()
    state = mem.get()
    
    # Assert fresh initialization
    assert isinstance(state, SceneState)
    assert state.character_reference_path is None
    assert state.reference_frame_path is None
    assert state.environment_prompt is None
    assert state.lighting_prompt is None
    assert state.style_prompt is None

def test_scene_memory_update_only_valid_fields():
    mem = SceneMemory()
    
    # Asset path
    path = "/fake/img.png"
    
    # Assert successful field updates
    mem.update(
        character_reference_path=path,
        environment_prompt="A dark spooky forest",
        lighting_prompt="cinematic dramatic lighting"
    )
    
    state = mem.get()
    assert state.character_reference_path == path
    assert state.reference_frame_path is None
    assert state.environment_prompt == "A dark spooky forest"
    assert state.lighting_prompt == "cinematic dramatic lighting"

    # Assert ignoring invalid fields safely
    mem.update(fake_field_name="should ignore this")
    assert not hasattr(state, 'fake_field_name')

def test_scene_memory_reset():
    mem = SceneMemory()
    path = "/fake/img.png"
    mem.update(reference_frame_path=path, style_prompt="cyberpunk")
    
    assert mem.get().style_prompt == "cyberpunk"
    
    mem.reset()
    state = mem.get()
    assert state.style_prompt is None
    assert state.reference_frame_path is None
