"""Model path resolution for local and downloaded LiteRT-LM models."""

import os
from pathlib import Path
from typing import Callable

LOCAL_E4B_FILENAME = "gemma-4-E4B-it.litertlm"
FALLBACK_HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
FALLBACK_HF_FILENAME = "gemma-4-E2B-it.litertlm"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def _find_litertlm(path: Path) -> Path | None:
    if path.is_file():
        return path
    if not path.is_dir():
        return None

    preferred = path / LOCAL_E4B_FILENAME
    if preferred.is_file():
        return preferred

    candidates = sorted(path.glob("*.litertlm"))
    return candidates[0] if candidates else None


def resolve_model_path(
    model_path: str | None = None,
    models_dir: Path = DEFAULT_MODELS_DIR,
    downloader: Callable[..., str] | None = None,
) -> str:
    """Resolve the LiteRT-LM model file.

    Resolution order:
    1. MODEL_PATH, accepting either a file or a directory containing a .litertlm.
    2. Project models directory, when it contains a .litertlm.
    3. Hugging Face Gemma 4 E2B fallback download into the project models directory.
    """
    configured = model_path if model_path is not None else os.environ.get("MODEL_PATH", "")
    if configured:
        resolved = _find_litertlm(Path(configured).expanduser())
        if resolved:
            return str(resolved)
        raise FileNotFoundError(f"MODEL_PATH does not point to a .litertlm file: {configured}")

    models_dir = models_dir.expanduser()
    local_model = _find_litertlm(models_dir)
    if local_model:
        return str(local_model)

    should_log_download = downloader is None
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    if should_log_download:
        print(f"Downloading {FALLBACK_HF_REPO}/{FALLBACK_HF_FILENAME} (first run only)...")
    models_dir.mkdir(parents=True, exist_ok=True)
    return downloader(
        repo_id=FALLBACK_HF_REPO,
        filename=FALLBACK_HF_FILENAME,
        local_dir=str(models_dir),
    )
