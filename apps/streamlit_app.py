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


def _get_engine(ctx, modality: str):
    """Lazy-import and return the appropriate engine instance."""
    if modality == "Image":
        from multigenai.engines.image_engine import ImageEngine
        return ImageEngine(ctx)
    elif modality == "Video":
        from multigenai.engines.video_engine import VideoEngine
        return VideoEngine(ctx)
    elif modality == "Audio":
        from multigenai.engines.audio_engine import AudioEngine
        return AudioEngine(ctx)
    elif modality == "Document":
        from multigenai.engines.document_engine import DocumentEngine
        return DocumentEngine(ctx)
    elif modality == "Code":
        from multigenai.engines.code_engine import CodeEngine
        return CodeEngine(ctx)
    else:
        return None


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

        # Style selector (loaded from StyleRegistry)
        style_names = ["None"] + list(ctx.style_registry.list_styles())
        style = st.selectbox("Style preset", style_names, key="style")

        # Identity selector (loaded from IdentityStore)
        identity_ids = ["None"] + ctx.identity_store.list_all()
        identity = st.selectbox("Identity", identity_ids, key="identity")

        st.divider()

        # Environment badge — always visible, shows key runtime metrics
        st.divider()
        env = ctx.environment
        if env is not None:
            st.markdown("**🌍 Environment**")
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

    return modality, style, identity


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

def _render_main(ctx, modality: str, style: str, identity: str):
    st.header(f"Generate · {modality}")

    prompt = st.text_area(
        "Prompt",
        placeholder=f"Describe the {modality.lower()} you want to generate…",
        height=120,
        key="prompt",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        generate = st.button("✨ Generate", type="primary", use_container_width=True)
    with col2:
        st.caption(
            "LLM enhancement is " +
            ("**active**" if ctx.llm else "**disabled** (rule-based fallback)")
        )

    if not generate:
        return

    if not prompt.strip():
        st.warning("Please enter a prompt first.")
        return

    _run_generation(ctx, modality, prompt, style, identity)


def _run_generation(ctx, modality: str, prompt: str, style: str, identity: str):
    """Route to the correct engine and display results."""
    import traceback

    with st.spinner(f"Generating {modality.lower()}…"):
        try:
            engine = _get_engine(ctx, modality)
            if engine is None:
                st.error(f"Engine for '{modality}' is not available.")
                return

            # Build a minimal request dict — engines accept **kwargs or typed requests
            request_kwargs: dict = {"prompt": prompt}
            if style and style != "None":
                request_kwargs["style_preset"] = style
            if identity and identity != "None":
                request_kwargs["identity_id"] = identity

            # Each engine exposes a typed request dataclass
            result = _call_engine(engine, modality, request_kwargs)
            _render_result(modality, result)

        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


def _call_engine(engine, modality: str, kwargs: dict):
    """Construct the typed request and call engine.run()."""
    if modality == "Image":
        from multigenai.engines.image_engine.engine import ImageRequest
        req = ImageRequest(prompt=kwargs["prompt"])
        return engine.run(req)
    elif modality == "Video":
        from multigenai.engines.video_engine.engine import VideoRequest
        req = VideoRequest(prompt=kwargs["prompt"])
        return engine.run(req)
    elif modality == "Audio":
        from multigenai.engines.audio_engine.engine import AudioRequest
        req = AudioRequest(prompt=kwargs["prompt"])
        return engine.run(req)
    elif modality == "Document":
        from multigenai.engines.document_engine.engine import DocumentRequest
        req = DocumentRequest(topic=kwargs["prompt"])
        return engine.run(req)
    elif modality == "Code":
        from multigenai.engines.code_engine.engine import CodeRequest
        req = CodeRequest(description=kwargs["prompt"])
        return engine.run(req)
    return None


def _render_result(modality: str, result):
    """Display the generation result appropriately for the modality."""
    if result is None:
        st.warning("No result returned.")
        return

    output_path = getattr(result, "output_path", None)
    st.success("✅ Generation complete!")

    if modality == "Image" and output_path:
        try:
            st.image(output_path, caption="Generated Image", use_column_width=True)
        except Exception:
            st.write(f"Output saved: `{output_path}`")

    elif modality == "Audio" and output_path:
        try:
            st.audio(output_path)
        except Exception:
            st.write(f"Output saved: `{output_path}`")

    elif output_path:
        st.write(f"📁 Output saved: `{output_path}`")
        if str(output_path).endswith((".docx", ".pptx", ".py", ".js", ".ts")):
            st.download_button(
                "⬇️ Download",
                data=open(output_path, "rb").read(),
                file_name=str(output_path).split("/")[-1].split("\\")[-1],
            )
    else:
        st.json(vars(result) if hasattr(result, "__dict__") else str(result))


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main():
    ctx = _get_context()
    modality, style, identity = _render_sidebar(ctx)
    _render_main(ctx, modality, style, identity)


if __name__ == "__main__":
    main()

# Streamlit auto-calls the module; calling main() here drives the layout
main()
