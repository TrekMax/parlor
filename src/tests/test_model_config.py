import tempfile
import unittest
from pathlib import Path

import model_config


class ResolveModelPathTests(unittest.TestCase):
    def test_model_path_file_is_used_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "custom.litertlm"
            model_file.write_text("model")

            resolved = model_config.resolve_model_path(
                model_path=str(model_file),
                models_dir=Path(tmp) / "models",
                downloader=lambda **_: self.fail("downloader should not be called"),
            )

            self.assertEqual(resolved, str(model_file))

    def test_model_path_directory_resolves_litertlm_inside_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "gemma-4-E4B-it.litertlm"
            model_file.write_text("model")

            resolved = model_config.resolve_model_path(
                model_path=tmp,
                models_dir=Path(tmp) / "models",
                downloader=lambda **_: self.fail("downloader should not be called"),
            )

            self.assertEqual(resolved, str(model_file))

    def test_project_models_directory_is_preferred_before_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            models_dir.mkdir()
            model_file = models_dir / "gemma-4-E4B-it.litertlm"
            model_file.write_text("model")

            resolved = model_config.resolve_model_path(
                model_path="",
                models_dir=models_dir,
                downloader=lambda **_: self.fail("downloader should not be called"),
            )

            self.assertEqual(resolved, str(model_file))

    def test_downloads_e2b_into_project_models_directory_when_no_local_model_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            calls = []

            def downloader(**kwargs):
                calls.append(kwargs)
                return str(models_dir / "gemma-4-E2B-it.litertlm")

            resolved = model_config.resolve_model_path(
                model_path="",
                models_dir=models_dir,
                downloader=downloader,
            )

            self.assertEqual(resolved, str(models_dir / "gemma-4-E2B-it.litertlm"))
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": model_config.FALLBACK_HF_REPO,
                        "filename": model_config.FALLBACK_HF_FILENAME,
                        "local_dir": str(models_dir),
                    }
                ],
            )

    def test_default_models_directory_is_project_relative(self):
        self.assertEqual(
            model_config.DEFAULT_MODELS_DIR,
            Path(model_config.__file__).resolve().parent.parent / "models",
        )


if __name__ == "__main__":
    unittest.main()
