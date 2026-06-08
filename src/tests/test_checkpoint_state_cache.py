import tempfile
import time
import unittest
from pathlib import Path
import sys
from unittest import mock

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from iterative_testing import Base_Agents


class CheckpointStateCacheTests(unittest.TestCase):
    def setUp(self):
        Base_Agents.clear_checkpoint_state_cache()

    def tearDown(self):
        Base_Agents.clear_checkpoint_state_cache()

    def _write_checkpoint_file(self, path: Path, content: bytes) -> None:
        path.write_bytes(content)

    def test_repeated_load_uses_single_torch_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pth"
            self._write_checkpoint_file(checkpoint_path, b"checkpoint-v1")
            expected_state = {"weight": object()}

            with mock.patch.object(Base_Agents.torch, "load", return_value=expected_state) as load_mock:
                first = Base_Agents.load_checkpoint_state_dict_cached(checkpoint_path)
                second = Base_Agents.load_checkpoint_state_dict_cached(checkpoint_path)

            self.assertIs(first, expected_state)
            self.assertIs(second, expected_state)
            self.assertEqual(load_mock.call_count, 1)

    def test_file_change_invalidates_cache_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pth"
            self._write_checkpoint_file(checkpoint_path, b"checkpoint-v1")
            states = [{"version": 1}, {"version": 2}]

            with mock.patch.object(Base_Agents.torch, "load", side_effect=states) as load_mock:
                first = Base_Agents.load_checkpoint_state_dict_cached(checkpoint_path)
                time.sleep(0.001)
                self._write_checkpoint_file(checkpoint_path, b"checkpoint-v2-has-new-size")
                second = Base_Agents.load_checkpoint_state_dict_cached(checkpoint_path)

            self.assertEqual(first, {"version": 1})
            self.assertEqual(second, {"version": 2})
            self.assertEqual(load_mock.call_count, 2)

    def test_torch_load_uses_cpu_map_location(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pth"
            self._write_checkpoint_file(checkpoint_path, b"checkpoint-v1")

            with mock.patch.object(Base_Agents.torch, "load", return_value={}) as load_mock:
                Base_Agents.load_checkpoint_state_dict_cached(checkpoint_path)

            _, kwargs = load_mock.call_args
            self.assertEqual(kwargs.get("map_location"), "cpu")


if __name__ == "__main__":
    unittest.main()
