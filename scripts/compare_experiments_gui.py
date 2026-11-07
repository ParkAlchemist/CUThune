from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_EVAL_ROOT = "eval"
DEFAULT_RUN_ROOT = "runs/cyclegan_cqt"


def find_preview_dirs(root: Path) -> list:
    if not root.exists():
        return []

    dirs = []

    for path in root.rglob("*"):
        if path.is_dir() and (path / "metrics.csv").exists():
            dirs.append(path)

    return sorted(dirs)


def read_metrics(preview_dir: Path) -> pd.DataFrame:
    path = preview_dir / "metrics.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def preview_png_path(preview_dir: Path, sample_index: int) -> Path:
    return preview_dir / f"preview_{sample_index:04d}.png"


def preview_tensor_path(preview_dir: Path, sample_index: int) -> Path:
    return preview_dir / "tensors" / f"preview_{sample_index:04d}.pt"


def load_preview_tensor(preview_dir: Path, sample_index: int) -> dict[str, Any] | None:
    path = preview_tensor_path(preview_dir, sample_index)

    if not path.exists():
        return None

    return torch.load(path, map_location="cpu")


def tensor_to_cqt_2d(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()

    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")

    return x.numpy().astype(np.float32)


def normalized_cqt_to_amplitude(
    cqt_norm: np.ndarray,
    db_min: float = -80.0,
    db_max: float = 0.0,
    ref_mag: float = 1.0,
) -> np.ndarray:
    db = ((cqt_norm + 1.0) / 2.0) * (db_max - db_min) + db_min
    amp = librosa.db_to_amplitude(db, ref=ref_mag)
    return amp.astype(np.float32)


def rough_icqt_reconstruct(
    cqt_norm: np.ndarray,
    sample_rate: int = 22050,
    hop_length: int = 512,
    bins_per_octave: int = 12,
    db_min: float = -80.0,
    db_max: float = 0.0,
    n_iter: int = 32,
) -> np.ndarray:
    """
    Rough magnitude-only CQT reconstruction.

    This is diagnostic only. It is not expected to be high quality.
    """
    amp = normalized_cqt_to_amplitude(
        cqt_norm=cqt_norm,
        db_min=db_min,
        db_max=db_max,
        ref_mag=1.0,
    )

    try:
        y = librosa.griffinlim_cqt(
            amp,
            sr=sample_rate,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            n_iter=n_iter,
        )
    except Exception:
        # Fallback: zero-phase inverse CQT, usually worse but still sometimes audible.
        complex_cqt = amp.astype(np.complex64)
        y = librosa.icqt(
            complex_cqt,
            sr=sample_rate,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
        )

    y = y.astype(np.float32)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak * 0.95

    return y


def make_audio_file(
    y: np.ndarray,
    sample_rate: int,
    name: str,
) -> Path:
    tmp_dir = Path(tempfile.gettempdir()) / "chiptune_cqt_compare_audio"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    path = tmp_dir / name
    sf.write(path, y, sample_rate)

    return path


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    numeric = df.select_dtypes(include=["number"])

    keep_cols = [
        col for col in numeric.columns
        if col not in {"sample_index", "epoch", "global_step"}
    ]

    summary = []

    for col in keep_cols:
        values = numeric[col].dropna()
        if len(values) == 0:
            continue

        summary.append({
            "metric": col,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
        })

    return pd.DataFrame(summary)


def get_sample_indices(*dfs: pd.DataFrame) -> list:
    indices = set()

    for df in dfs:
        if not df.empty and "sample_index" in df.columns:
            indices.update(int(x) for x in df["sample_index"].dropna().tolist())

    return sorted(indices)


def metric_delta_table(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    sa = summarize_metrics(df_a)
    sb = summarize_metrics(df_b)

    if sa.empty or sb.empty:
        return pd.DataFrame()

    merged = sa[["metric", "mean"]].merge(
        sb[["metric", "mean"]],
        on="metric",
        suffixes=("_left", "_right"),
    )

    merged["delta_right_minus_left"] = merged["mean_right"] - merged["mean_left"]

    preferred_order = [
        "chroma_x_fake_y_l1",
        "cycle_x_l1",
        "cycle_y_l1",
        "identity_y_l1",
        "identity_x_l1",
        "fake_y_std",
        "fake_y_mean",
        "d_y_fake_mean",
        "d_y_real_mean",
    ]

    merged["sort_key"] = merged["metric"].apply(
        lambda x: preferred_order.index(x) if x in preferred_order else 999
    )

    merged = merged.sort_values(["sort_key", "metric"]).drop(columns=["sort_key"])

    return merged


def display_preview_column(label: str, preview_dir: Path, sample_index: int) -> None:
    st.subheader(label)

    png_path = preview_png_path(preview_dir, sample_index)

    if png_path.exists():
        st.image(str(png_path), use_container_width=True)
    else:
        st.warning(f"Missing preview image: {png_path}")

    tensor_data = load_preview_tensor(preview_dir, sample_index)

    if tensor_data is None:
        st.info("No saved tensors found. Run preview export with `--save-tensors` to enable rough iCQT audio.")
        return

    st.markdown("#### Rough iCQT audio")

    choices = [
        "real_x",
        "fake_y",
        "rec_x",
        "real_y",
        "id_y",
        "fake_x",
    ]

    selected = st.selectbox(
        f"Tensor for {label}",
        choices,
        index=choices.index("fake_y") if "fake_y" in choices else 0,
        key=f"{label}_audio_choice",
    )

    if selected not in tensor_data:
        st.warning(f"Tensor `{selected}` not found.")
        return

    col1, col2 = st.columns(2)

    with col1:
        n_iter = st.slider(
            f"Griffin-Lim iterations for {label}",
            min_value=8,
            max_value=96,
            value=32,
            step=8,
            key=f"{label}_n_iter",
        )

    with col2:
        reconstruct = st.button(
            f"Generate audio for {label}",
            key=f"{label}_reconstruct",
        )

    if reconstruct:
        cqt = tensor_to_cqt_2d(tensor_data[selected])
        y = rough_icqt_reconstruct(
            cqt_norm=cqt,
            sample_rate=st.session_state.sample_rate,
            hop_length=st.session_state.hop_length,
            bins_per_octave=st.session_state.bins_per_octave,
            db_min=st.session_state.db_min,
            db_max=st.session_state.db_max,
            n_iter=n_iter,
        )

        audio_path = make_audio_file(
            y,
            sample_rate=st.session_state.sample_rate,
            name=f"{preview_dir.name}_{sample_index:04d}_{selected}.wav",
        )

        st.audio(str(audio_path), format="audio/wav")
        st.caption(str(audio_path))


def main() -> None:
    st.set_page_config(
        page_title="CQT Experiment Comparator",
        layout="wide",
    )

    st.title("CQT Experiment Comparator")
    st.caption(
        "Compare CycleGAN CQT previews, metrics, and rough iCQT reconstructions across experiments."
    )

    with st.sidebar:
        st.header("Folders")

        eval_root = Path(st.text_input("Eval / previews root", DEFAULT_EVAL_ROOT))
        run_root = Path(st.text_input("Runs root", DEFAULT_RUN_ROOT))

        search_roots = [eval_root, run_root]

        st.header("Audio / CQT settings")

        st.session_state.sample_rate = st.number_input(
            "Sample rate",
            min_value=8000,
            max_value=96000,
            value=22050,
            step=1,
        )

        st.session_state.hop_length = st.number_input(
            "Hop length",
            min_value=64,
            max_value=4096,
            value=512,
            step=1,
        )

        st.session_state.bins_per_octave = st.number_input(
            "Bins per octave",
            min_value=1,
            max_value=48,
            value=12,
            step=1,
        )

        st.session_state.db_min = st.number_input(
            "dB min",
            min_value=-160.0,
            max_value=-1.0,
            value=-80.0,
            step=1.0,
        )

        st.session_state.db_max = st.number_input(
            "dB max",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
        )

        refresh = st.button("Refresh preview folders")

    preview_dirs = []
    for root in search_roots:
        preview_dirs.extend(find_preview_dirs(root))

    preview_dirs = sorted(set(preview_dirs))

    if not preview_dirs:
        st.warning("No preview folders with metrics.csv found.")
        st.stop()

    labels = [str(path) for path in preview_dirs]

    st.header("Experiment selection")

    col_a, col_b = st.columns(2)

    with col_a:
        left_label = st.selectbox("Left preview folder", labels, index=0)

    with col_b:
        right_default = min(1, len(labels) - 1)
        right_label = st.selectbox("Right preview folder", labels, index=right_default)

    left_dir = Path(left_label)
    right_dir = Path(right_label)

    df_left = read_metrics(left_dir)
    df_right = read_metrics(right_dir)

    sample_indices = get_sample_indices(df_left, df_right)

    if not sample_indices:
        st.warning("No sample_index values found in metrics.")
        st.stop()

    sample_index = st.select_slider(
        "Sample index",
        options=sample_indices,
        value=sample_indices[0],
    )

    st.divider()

    tab_preview, tab_metrics, tab_sample, tab_notes = st.tabs(
        ["CQT previews", "Aggregate metrics", "Per-sample metrics", "Notes"]
    )

    with tab_preview:
        left_col, right_col = st.columns(2)
        with left_col:
            display_preview_column("Left", left_dir, sample_index)
        with right_col:
            display_preview_column("Right", right_dir, sample_index)

    with tab_metrics:
        st.subheader("Metric deltas")

        delta = metric_delta_table(df_left, df_right)

        if delta.empty:
            st.info("No numeric metrics to compare.")
        else:
            st.dataframe(delta, use_container_width=True)

        st.subheader("Left summary")
        st.dataframe(summarize_metrics(df_left), use_container_width=True)

        st.subheader("Right summary")
        st.dataframe(summarize_metrics(df_right), use_container_width=True)

    with tab_sample:
        st.subheader(f"Metrics for sample {sample_index}")

        left_sample = df_left[df_left["sample_index"] == sample_index] if not df_left.empty else pd.DataFrame()
        right_sample = df_right[df_right["sample_index"] == sample_index] if not df_right.empty else pd.DataFrame()

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Left")
            st.dataframe(left_sample.T, use_container_width=True)

        with c2:
            st.markdown("#### Right")
            st.dataframe(right_sample.T, use_container_width=True)

    with tab_notes:
        st.subheader("Manual comparison notes")

        note_key = f"notes_{left_dir}_{right_dir}_{sample_index}"

        notes = st.text_area(
            "Write subjective notes here",
            key=note_key,
            height=200,
            placeholder="Example: right has cleaner horizontal bands, but slightly worse cycle reconstruction.",
        )

        st.info(
            "This field is session-local for now. If useful, we can add persistent note saving to JSON/Markdown."
        )


if __name__ == "__main__":
    main()

