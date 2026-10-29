"""Regression checks for relocated checkouts and frozen-file integrity."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_public_archive import archive_members
from fedcernn.experiments import load_config, run_experiment
from fedcernn.paths import CODE_ROOT, frozen_hash_matches, resolve_project_path


class PortablePathTests(unittest.TestCase):
    def test_relocated_checkout_uses_the_same_output_from_different_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "renamed-checkout"
            (checkout / "configs").mkdir(parents=True)
            shutil.copyfile(CODE_ROOT / "configs/smoke.yaml", checkout / "configs/smoke.yaml")
            elsewhere = Path(directory) / "elsewhere"
            elsewhere.mkdir()
            previous = Path.cwd()
            try:
                with patch("fedcernn.paths.CODE_ROOT", checkout):
                    for cwd in (checkout.parent, checkout, elsewhere):
                        os.chdir(cwd)
                        config = load_config("code/configs/smoke.yaml")
                        # Zero repetitions exercise real output creation without training.
                        run_experiment(config, max_repetitions=0)
                        self.assertEqual(
                            resolve_project_path(config["experiment"]["output_directory"]),
                            checkout / "results/smoke",
                        )
                    self.assertTrue((checkout / "results/smoke").is_dir())
                    self.assertFalse((elsewhere / "code").exists())
                    self.assertEqual(resolve_project_path("custom-output"), Path("custom-output"))
                    self.assertEqual(resolve_project_path(elsewhere), elsewhere)
            finally:
                os.chdir(previous)

    def test_migrated_hash_requires_the_recorded_file_and_exact_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "config.yaml"
            source.write_bytes(b"relative path")
            old = hashlib.sha256(b"original path").hexdigest()
            current = hashlib.sha256(source.read_bytes()).hexdigest()
            (root / "PATH_MIGRATION.json").write_text(json.dumps({"files": {
                "config.yaml": {"original_sha256": old, "relative_path_sha256": current}
            }}))
            with patch("fedcernn.paths.CODE_ROOT", root):
                self.assertTrue(frozen_hash_matches(source, old))
                self.assertTrue(frozen_hash_matches(source, current))
                self.assertFalse(frozen_hash_matches(source, "wrong expected hash"))
                other = root / "other.yaml"
                other.write_bytes(source.read_bytes())
                self.assertFalse(frozen_hash_matches(other, old))
                source.write_bytes(b"unreviewed change")
                self.assertFalse(frozen_hash_matches(source, old))

    def test_public_archive_keeps_the_migration_manifest(self):
        self.assertIn(Path("PATH_MIGRATION.json"), archive_members(CODE_ROOT))


if __name__ == "__main__":
    unittest.main()
