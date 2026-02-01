"""
Streamlit camera snapshot demo:
- Capture a snapshot from the webcam
- Send it to an OpenAI-compatible VLM (e.g. moondream via Ollama)
- Show responses on the right as a dialogue

pip install streamlit openai pillow
"""

import base64
import io
import time
from dataclasses import dataclass
from typing import Tuple

from openai import OpenAI
from PIL import Image
import streamlit as st


# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------
@dataclass
class VLMConfig:
    api_base: str
    api_key: str
    model: str
    prompt: str
    max_tokens: int


# ---------------------------------------------------------------------
# VLM helper
# ---------------------------------------------------------------------
def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_vlm(cfg: VLMConfig, image: Image.Image) -> Tuple[str, float]:
    """
    Call an OpenAI-compatible vision endpoint and return (text, latency_ms).
    """
    client = OpenAI(base_url=cfg.api_base, api_key=cfg.api_key or "EMPTY")

    img_b64 = encode_image(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": cfg.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }
    ]

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        max_tokens=cfg.max_tokens,
        temperature=0.7,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    text = response.choices[0].message.content.strip()

    return text, latency_ms


# ---------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Streamlit VLM Snapshot Demo", layout="wide")
    st.title("Streamlit Camera + OpenAI-compatible VLM")
    st.caption("Experimental demo mirroring Live VLM WebUI (camera snapshot → VLM → dialogue).")

    # 初始化 session_state
    if "vlm_config_snapshot" not in st.session_state:
        st.session_state.vlm_config_snapshot = VLMConfig(
            api_base="http://192.168.11.20:40512/v1",
            api_key="",
            model="qwen3-vl:latest",
            prompt=(
            "請以 HP 伺服器現場工程師（Field Engineer）的專業角度，"
            "詳細描述影像中的伺服器硬體狀態。"
            "請逐一說明所有可見的 LED 指示燈，包括其顏色、閃爍狀態（恆亮、閃爍、熄滅）、"
            "所在位置（如電源供應器、硬碟、系統狀態燈、網路介面卡等），"
            "並解釋每一種指示燈狀態在 HP 伺服器中的實際意義。"
            "若有任何可能代表硬體故障、降級運作、預警或嚴重錯誤的 LED 指示，"
            "請特別標示並明確指出可能的故障類型（例如：硬碟故障、電源異常、風扇故障、"
            "系統板錯誤或溫度異常），並說明建議的初步檢查或維修行動。"
            "請避免模糊描述，並使用精確、工程師等級的技術用語。"
            ),
            max_tokens=300,
        )

    # 最新結果 (右側顯示用)
    if "vlm_latest" not in st.session_state:
        st.session_state.vlm_latest = None  # dict: {text, latency, ts}

    # ---- Sidebar: config UI ----
    with st.sidebar:
        st.subheader("VLM Settings")

        cfg0: VLMConfig = st.session_state.vlm_config_snapshot

        api_base = st.text_input("API Base", cfg0.api_base)
        api_key = st.text_input("API Key (if needed)", cfg0.api_key, type="password")
        model = st.text_input("Model", cfg0.model)
        prompt = st.text_area("Prompt", cfg0.prompt, height=80)
        max_tokens = st.slider("Max tokens", 32, 1024, cfg0.max_tokens, step=32)

        # 更新 snapshot
        st.session_state.vlm_config_snapshot = VLMConfig(
            api_base=api_base,
            api_key=api_key,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )

    cfg_snapshot: VLMConfig = st.session_state.vlm_config_snapshot

    # 兩欄：左邊影像，右邊對話
    col1, col2 = st.columns([2, 1])

    # ---- 左側：Camera & snapshot ----
    with col1:
        st.subheader("Webcam")
        uploaded_image = st.file_uploader(
            "Upload an image for VLM analysis",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            help="Select an image file, then click the button below to run VLM.",
            key="uploaded_image",
        )
        # Show the uploaded image
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        snapshot_btn = st.button(
            "Run VLM",
            type="primary",
            use_container_width=True,
            disabled=uploaded_image is None,
        )

    latest_text = "Waiting for snapshot..."
    latest_latency = 0.0

    # ---- Run inference when button pressed ----
    if snapshot_btn and uploaded_image is not None:
        with st.spinner("Running VLM..."):
            try:
                pil_image = Image.open(uploaded_image).convert("RGB")
                resized = pil_image.resize((640, 360))
                latest_text, latest_latency = call_vlm(cfg_snapshot, resized)
            except Exception as exc:
                latest_text = f"Error: {exc}"
                latest_latency = 0.0

        st.session_state.vlm_latest = {
            "text": latest_text,
            "latency": latest_latency,
            "ts": time.time(),
        }
    elif st.session_state.vlm_latest:
        latest_text = st.session_state.vlm_latest["text"]
        latest_latency = st.session_state.vlm_latest["latency"]

    # ---- 右側：對話視窗 ----
    with col2:
        st.subheader("VLM Inference")

        if st.session_state.vlm_latest:
            msg = st.session_state.vlm_latest
            with st.chat_message("assistant"):
                st.markdown(msg["text"])
                st.caption(f"Latency: {msg['latency']:.0f} ms")
        else:
            st.info("Waiting for a snapshot and first response from the VLM...")

    st.markdown(
        "Tip: run a local Ollama/vLLM/NIM endpoint with a vision model "
        "(e.g., `moondream` on Ollama) and set API Base to its `/v1` URL. "
        "Accept webcam permissions, take a snapshot, then click **Take snapshot & run VLM**."
    )


if __name__ == "__main__":
    main()
