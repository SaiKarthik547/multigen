"""
MultiGenAI OS — Streamlit UI

Thin UI layer only. Zero generation logic here.
All generation goes through ExecutionContext + engines.

Architecture rule: this file must NEVER import diffusers, torch,
transformers, or anything heavy. Only mgos public API.

Run:
    py -m streamlit run apps/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MultiGenAI OS",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Context — built once per browser session (ModelRegistry persists in session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initialising MGOS…")
def _get_context():
    """Build and cache ExecutionContext for the full session."""
    from multigenai.core.config.settings import get_settings
    from multigenai.core.execution_context import ExecutionContext
    settings = get_settings()
    return ExecutionContext.build(settings)


@st.cache_resource(show_spinner="Booting Orchestrator…")
def _get_manager(_ctx):
    """Lazy-boot the GenerationManager."""
    from multigenai.core.generation_manager import GenerationManager
    return GenerationManager(_ctx)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(ctx):
    with st.sidebar:
        st.title("🎬 MultiGenAI OS")
        st.caption(f"v0.1.0 · mode: `{ctx.settings.mode}`")

        st.divider()

        # LLM status indicator
        if ctx.llm is not None:
            st.success(f"🧠 LLM: {ctx.settings.llm.provider} / {ctx.settings.llm.model}")
        else:
            st.info("🔧 LLM: rule-based (offline mode)")

        st.divider()

        # Modality selector
        modality = st.selectbox(
            "Modality",
            ["Image", "Video", "Audio", "Document", "Code"],
            key="modality",
        )

        st.divider()
        st.markdown("**🌍 Environment**")
        env = ctx.environment
        if env is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Platform", env.platform.capitalize())
                st.metric("Device", env.device_type.upper())
            with col_b:
                st.metric("VRAM", f"{env.vram_mb} MB" if env.vram_mb else "—")
                st.metric("RAM", f"{env.ram_mb} MB" if env.ram_mb else "—")
            st.caption(f"Mode: **{env.mode}** · max res: {env.behaviour.max_image_resolution}px")
        else:
            st.caption("Environment: undetected")

        st.divider()

        # Capability expander
        with st.expander("⚙️ System Capabilities", expanded=False):
            cap = ctx.capability
            st.write(f"**Device:** `{ctx.device}`")
            st.write(f"**OS:** {cap.get('os', '?')}")
            st.write(f"**Python:** {cap.get('python', '?')}")
            st.write(f"**GPU:** {'✅ ' + cap.get('gpu_name', '') if cap.get('has_gpu') else '❌ CPU only'}")
            libs = cap.get("libraries", {})
            if libs:
                st.write("**Libraries:**")
                for lib, info in libs.items():
                    status = "✅" if info.get("available") else "❌"
                    version = info.get("version", "")
                    st.write(f"  {status} {lib} {version}")

    return modality


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

def _render_main(ctx, modality: str):
    st.header(f"Generate · {modality}")

    prompt = st.text_area(
        "Prompt",
        placeholder=f"Describe the {modality.lower()} you want to generate…",
        height=120,
        key="prompt",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        generate = st.button("✨ Generate", type="primary", use_container_width=True)
    with col2:
        st.caption(
            "LLM enhancement is " +
            ("**active**" if ctx.llm else "**disabled** (rule-based fallback)")
        )

    # Modality-specific options
    with st.expander("Additional Options", expanded=True):
        cols = st.columns(3)
        options = {}
        if modality == "Image":
            with cols[0]: options["use_refiner"] = st.checkbox("Use SDXL Refiner", value=True)
            with cols[1]: options["style"] = st.selectbox("Style", ["cinematic", "photorealistic", "anime", "watercolor", "sci-fi"])
            with cols[2]: options["seed"] = st.number_input("Seed", value=42)
        elif modality == "Video":
            with cols[0]: options["num_frames"] = st.slider("Frames", 8, 48, 16)
            with cols[1]: options["fps"] = st.slider("FPS", 4, 30, 8)
            with cols[2]: options["interpolate"] = st.checkbox("RIFE Interpolation", value=True)
        elif modality == "Audio":
            with cols[0]: options["audio_type"] = st.selectbox("Type", ["voice", "music", "ambient"])
            with cols[1]: options["duration_seconds"] = st.slider("Duration (s)", 1, 30, 5)
        elif modality == "Document":
            with cols[0]: options["output_format"] = st.selectbox("Format", ["docx", "pdf", "pptx"])
            with cols[1]: options["target_pages"] = st.number_input("Pages", value=5)

    if not generate:
        return

    if not prompt.strip():
        st.warning("Please enter a prompt first.")
        return

    _run_generation(ctx, modality, prompt, options)


def _run_generation(ctx, modality: str, prompt: str, options: dict):
    """Route to GenerationManager and display results."""
    import traceback
    manager = _get_manager(ctx)

    with st.spinner(f"Orchestrating {modality.lower()} generation…"):
        try:
            from multigenai.llm.schema_validator import (
                ImageGenerationRequest, VideoGenerationRequest, 
                AudioGenerationRequest, DocumentGenerationRequest
            )

            if modality == "Image":
                req = ImageGenerationRequest(prompt=prompt, **options)
                result = manager.generate_image(req)
            elif modality == "Video":
                req = VideoGenerationRequest(prompt=prompt, **options)
                result = manager.generate_video(req)
            elif modality == "Audio":
                req = AudioGenerationRequest(prompt=prompt, **options)
                result = manager.generate_audio(req)
            elif modality == "Document":
                req = DocumentGenerationRequest(prompt=prompt, **options)
                if options.get("output_format") == "pptx":
                    result = manager.generate_presentation(req)
                else:
                    result = manager.generate_document(req)
            elif modality == "Code":
                req = CodeGenerationRequest(prompt=prompt, **options)
                result = manager.generate_code(req)
            else:
                st.error(f"Modality '{modality}' not implemented in orchestrator.")
                return

            _display_result(modality, result)

        except Exception as exc:
            st.error(f"Orchestration failed: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


def _display_result(modality: str, result):
    """Display result based on modality and Result class."""
    if not result:
        st.warning("Engine returned no result.")
        return

    if hasattr(result, "success") and not result.success:
        st.error(f"Generation failed: {result.error}")
        return

    path = getattr(result, "path", None)
    if not path:
        st.warning("No output path found in result.")
        st.json(vars(result))
        return

    st.success(f"✅ {modality} ready!")
    
    if modality == "Image":
        st.image(path, caption="Base Generation Result", use_column_width=True)
    elif modality == "Video":
        st.video(path)
    elif modality == "Audio":
        st.audio(path)
    
    st.write(f"📁 Path: `{path}`")
    with open(path, "rb") as f:
        st.download_button(
            "⬇️ Download Output",
            data=f.read(),
            file_name=path.split("/")[-1].split("\\")[-1]
        )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main():
    ctx = _get_context()
    modality = _render_sidebar(ctx)
    _render_main(ctx, modality)


if __name__ == "__main__":
    main()

# Streamlit auto-calls the module; calling main() here drives the layout
main()
