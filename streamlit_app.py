from __future__ import annotations

import logging
import queue
import shutil
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st

from audiobook_tts.generator import (
    GenerationCancelled,
    GenerationResult,
    ProgressEvent,
    generate_audiobook,
    make_default_output_dir,
)
from audiobook_tts.settings import SAMPLE_RATES, SILERO_VOICES, GenerationSettings
from audiobook_tts.text_utils import safe_name


@dataclass
class GenerationJob:
    thread: threading.Thread | None
    cancel_event: threading.Event
    events: queue.Queue[tuple[str, object]]
    input_temp_dir: Path
    output_dir: Path
    delete_output_on_cancel: bool
    messages: list[str] = field(default_factory=list)
    progress: float = 0.0
    status: str = ""
    result: GenerationResult | None = None
    error: str | None = None
    error_details: str | None = None
    cleanup_error: str | None = None
    cancelled: bool = False
    stop_requested: bool = False
    done: bool = False


def _run_generation_job(
    job: GenerationJob,
    input_path: Path,
    output_dir: Path,
    *,
    merge: bool,
    settings: GenerationSettings,
) -> None:
    def on_progress(event: ProgressEvent) -> None:
        job.events.put(("progress", event))

    try:
        result = generate_audiobook(
            input_path,
            output_dir=output_dir,
            merge=merge,
            settings=settings,
            progress=on_progress,
            cancel_requested=job.cancel_event.is_set,
        )
    except GenerationCancelled:
        job.events.put(("cancelled", None))
        if job.delete_output_on_cancel:
            try:
                if output_dir.exists():
                    shutil.rmtree(output_dir)
            except Exception as exc:
                job.events.put(("cleanup_error", str(exc)))
            else:
                job.events.put(("cleanup_done", output_dir))
    except Exception as exc:
        logging.exception("Audiobook generation failed")
        message = str(exc).strip() or type(exc).__name__
        job.events.put(("error", {"message": message, "details": traceback.format_exc()}))
    else:
        job.events.put(("done", result))
    finally:
        shutil.rmtree(job.input_temp_dir, ignore_errors=True)
        job.events.put(("finished", None))


def _drain_job_events(job: GenerationJob) -> None:
    while True:
        try:
            event_type, payload = job.events.get_nowait()
        except queue.Empty:
            break

        if event_type == "progress":
            event = payload
            if not isinstance(event, ProgressEvent):
                continue
            if event.total_chapters and event.chapter_index:
                job.progress = min(event.chapter_index / event.total_chapters, 1.0)
            elif event.stage == "done":
                job.progress = 1.0
            job.status = event.message
            job.messages.append(event.message)
        elif event_type == "done":
            if isinstance(payload, GenerationResult):
                job.result = payload
            job.progress = 1.0
            job.done = True
        elif event_type == "cancelled":
            job.cancelled = True
            job.done = True
            job.status = "Генерация остановлена"
            job.messages.append(job.status)
        elif event_type == "cleanup_done":
            job.messages.append(f"Папка результата удалена: {payload}")
        elif event_type == "cleanup_error":
            job.cleanup_error = str(payload)
            job.messages.append("Не удалось удалить папку результата")
        elif event_type == "error":
            if isinstance(payload, dict):
                job.error = str(payload.get("message") or "Неизвестная ошибка")
                job.error_details = str(payload.get("details") or "")
            else:
                job.error = str(payload) or "Неизвестная ошибка"
            job.done = True
            job.status = "Ошибка генерации"
            job.messages.append(job.status)
        elif event_type == "finished":
            job.done = True


def _build_output_dir(
    *,
    input_path: Path,
    project_name: str,
    output_root: str,
    add_timestamp: bool,
) -> Path:
    if project_name.strip():
        folder_name = safe_name(project_name)
        if add_timestamp:
            folder_name = f"{folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return Path(output_root).expanduser() / folder_name
    return make_default_output_dir(input_path)


def _start_generation_job(
    *,
    uploaded_name: str,
    uploaded_bytes: bytes,
    project_name: str,
    output_root: str,
    add_timestamp: bool,
    merge: bool,
    settings: GenerationSettings,
    delete_output_on_cancel: bool,
) -> GenerationJob:
    input_temp_dir = Path(tempfile.mkdtemp(prefix="audiobook_upload_"))
    input_path = input_temp_dir / Path(uploaded_name).name
    input_path.write_bytes(uploaded_bytes)
    output_dir = _build_output_dir(
        input_path=input_path,
        project_name=project_name,
        output_root=output_root,
        add_timestamp=add_timestamp,
    )

    job = GenerationJob(
        thread=None,
        cancel_event=threading.Event(),
        events=queue.Queue(),
        input_temp_dir=input_temp_dir,
        output_dir=output_dir,
        delete_output_on_cancel=delete_output_on_cancel,
        status="Генерация запускается",
        messages=["Генерация запускается"],
    )
    thread = threading.Thread(
        target=_run_generation_job,
        args=(job, input_path, output_dir),
        kwargs={
            "merge": merge,
            "settings": settings,
        },
        daemon=True,
    )
    job.thread = thread
    thread.start()
    return job


st.set_page_config(page_title="Озвучка книг", layout="wide")

if "generation_job" not in st.session_state:
    st.session_state.generation_job = None

job: GenerationJob | None = st.session_state.generation_job
if job is not None:
    _drain_job_events(job)

is_running = job is not None and job.thread is not None and job.thread.is_alive()

st.title("Озвучка книг")

uploaded = st.file_uploader("Книга", type=("epub", "txt", "fb2"), disabled=is_running)

with st.sidebar:
    st.header("Параметры")
    voice = st.selectbox("Голос", SILERO_VOICES, index=SILERO_VOICES.index("baya"), disabled=is_running)
    sample_rate = st.selectbox(
        "Частота",
        SAMPLE_RATES,
        index=SAMPLE_RATES.index(24000),
        disabled=is_running,
    )
    speed = st.slider("Скорость", 0.7, 2.0, 1.0, 0.05, disabled=is_running)
    pause_ms = st.slider("Пауза между фрагментами, мс", 0, 1500, 150, 50, disabled=is_running)
    chunk_chars = st.slider("Размер фрагмента TTS, символы", 400, 1500, 850, 50, disabled=is_running)
    chapter_minutes = st.number_input(
        "Длительность главы, минут",
        min_value=1,
        max_value=240,
        value=20,
        step=1,
        disabled=is_running,
    )
    threads = st.number_input("Потоки CPU", min_value=1, max_value=16, value=4, step=1, disabled=is_running)
    merge = st.checkbox("Склеить в одну книгу", value=False, disabled=is_running)
    stop_action = st.radio(
        "После нажатия «Стоп»",
        ("Оставить готовые главы", "Удалить папку результата"),
        index=0,
        disabled=is_running,
    )

default_project = safe_name(Path(uploaded.name).stem) if uploaded else ""
project_name = st.text_input("Имя папки результата", value=default_project, disabled=is_running)
output_root = st.text_input("Корневая папка", value=str(Path.cwd() / "outputs"), disabled=is_running)
add_timestamp = st.checkbox("Добавить дату и время к папке", value=True, disabled=is_running)

left, right = st.columns([1, 2])
with left:
    start = st.button("Сгенерировать", disabled=uploaded is None or is_running, type="primary")
with right:
    stop = st.button(
        "Стоп",
        disabled=not is_running or job is None or job.stop_requested,
        type="secondary",
    )

if stop and job is not None:
    job.cancel_event.set()
    job.stop_requested = True
    job.status = "Остановка запрошена. Текущий фрагмент будет завершен."
    job.messages.append(job.status)

if start and uploaded is not None and not is_running:
    settings = GenerationSettings(
        voice=voice,
        sample_rate=int(sample_rate),
        max_chunk_chars=int(chunk_chars),
        pause_ms=int(pause_ms),
        speech_speed=float(speed),
        target_chapter_minutes=int(chapter_minutes),
        torch_threads=int(threads),
    )
    job = _start_generation_job(
        uploaded_name=uploaded.name,
        uploaded_bytes=uploaded.getvalue(),
        project_name=project_name,
        output_root=output_root,
        add_timestamp=add_timestamp,
        merge=merge,
        settings=settings,
        delete_output_on_cancel=stop_action == "Удалить папку результата",
    )
    st.session_state.generation_job = job
    is_running = True

if job is not None:
    st.progress(job.progress)
    st.write(job.status)
    if job.messages:
        st.text("\n".join(job.messages[-8:]))

    if job.stop_requested and is_running:
        st.info("Остановка запрошена. Генерация завершится на ближайшей безопасной точке.")
    if job.error:
        st.error(job.error)
        if job.error_details:
            with st.expander("Подробности ошибки"):
                st.code(job.error_details)
    elif job.cleanup_error:
        st.error(f"Не удалось удалить папку результата: {job.cleanup_error}")
    elif job.cancelled:
        st.warning("Генерация остановлена.")
    elif job.result:
        st.success(f"Готово: {job.result.output_dir}")
        st.write(f"Глав: {job.result.chapters_count}")
        if job.result.merged_file:
            st.write(f"Общая книга: {job.result.merged_file}")
        st.dataframe(
            {"Файл": [str(path) for path in job.result.chapter_files]},
            use_container_width=True,
            hide_index=True,
        )

if job is not None and (not job.done or (job.thread is not None and job.thread.is_alive())):
    time.sleep(0.5)
    st.rerun()
