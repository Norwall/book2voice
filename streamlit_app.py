from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from audiobook_tts.generator import ProgressEvent, generate_audiobook, make_default_output_dir
from audiobook_tts.settings import SAMPLE_RATES, SILERO_VOICES, GenerationSettings
from audiobook_tts.text_utils import safe_name


st.set_page_config(page_title="Озвучка книг", layout="wide")

st.title("Озвучка книг")

uploaded = st.file_uploader("Книга", type=("epub", "txt"))

with st.sidebar:
    st.header("Параметры")
    voice = st.selectbox("Голос", SILERO_VOICES, index=SILERO_VOICES.index("baya"))
    sample_rate = st.selectbox("Частота", SAMPLE_RATES, index=SAMPLE_RATES.index(24000))
    speed = st.slider("Скорость", 0.7, 1.4, 1.0, 0.05)
    pause_ms = st.slider("Пауза между фрагментами, мс", 0, 1500, 350, 50)
    chunk_chars = st.slider("Размер фрагмента TTS, символы", 400, 1500, 850, 50)
    txt_chapter_chars = st.number_input(
        "Размер главы TXT без заголовков",
        min_value=3000,
        max_value=50000,
        value=12000,
        step=1000,
    )
    threads = st.number_input("Потоки CPU", min_value=1, max_value=16, value=4, step=1)
    merge = st.checkbox("Склеить в одну книгу", value=False)

default_project = safe_name(Path(uploaded.name).stem) if uploaded else ""
project_name = st.text_input("Имя папки результата", value=default_project)
output_root = st.text_input("Корневая папка", value=str(Path.cwd() / "outputs"))
add_timestamp = st.checkbox("Добавить дату и время к папке", value=True)

left, right = st.columns([1, 2])
with left:
    start = st.button("Сгенерировать", disabled=uploaded is None, type="primary")
with right:
    st.write("")

if start and uploaded is not None:
    settings = GenerationSettings(
        voice=voice,
        sample_rate=int(sample_rate),
        max_chunk_chars=int(chunk_chars),
        pause_ms=int(pause_ms),
        speech_speed=float(speed),
        torch_threads=int(threads),
    )

    progress_bar = st.progress(0)
    status = st.empty()
    log = st.empty()
    messages: list[str] = []

    def on_progress(event: ProgressEvent) -> None:
        if event.total_chapters and event.chapter_index:
            progress_bar.progress(event.chapter_index / event.total_chapters)
        elif event.stage == "done":
            progress_bar.progress(1.0)
        status.write(event.message)
        messages.append(event.message)
        log.text("\n".join(messages[-8:]))

    with tempfile.TemporaryDirectory(prefix="audiobook_upload_") as temp_dir:
        input_path = Path(temp_dir) / uploaded.name
        input_path.write_bytes(uploaded.getbuffer())

        if project_name.strip():
            folder_name = safe_name(project_name)
            if add_timestamp:
                folder_name = f"{folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir = Path(output_root).expanduser() / folder_name
        else:
            output_dir = make_default_output_dir(input_path)

        try:
            result = generate_audiobook(
                input_path,
                output_dir=output_dir,
                merge=merge,
                settings=settings,
                txt_chapter_chars=int(txt_chapter_chars),
                progress=on_progress,
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"Готово: {result.output_dir}")
            st.write(f"Глав: {result.chapters_count}")
            if result.merged_file:
                st.write(f"Общая книга: {result.merged_file}")
            st.dataframe(
                {"Файл": [str(path) for path in result.chapter_files]},
                use_container_width=True,
                hide_index=True,
            )
