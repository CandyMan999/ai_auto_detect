"""Contract tests with isolated functions; no model downloads or ML runtime needed."""
import ast
import __future__
import asyncio
import io
import os
from pathlib import Path
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock


ROOT = Path(__file__).resolve().parents[1]


def load_functions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


class ImageNudityTests(unittest.TestCase):
    def test_threshold_labels_and_temporary_file_cleanup(self):
        # Read the production label set and threshold along with the real helpers.
        namespace = dict(os=os, tempfile=tempfile, Optional=Optional, List=List, Dict=Dict, Any=Any)
        tree = ast.parse((ROOT / "nudity_service.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"NUDITY_TAGS", "THRESHOLD"}:
                        namespace[target.id] = ast.literal_eval(node.value)
        detector = Mock()
        namespace["_get_detector"] = Mock(return_value=detector)
        load_functions("nudity_service.py", {"detect_image_nudity", "_analyze_preds"}, namespace)
        image = Mock()
        image.save.side_effect = lambda path, **kwargs: Path(path).write_bytes(b"test image")
        cases = [
            ([], False),
            ([{"class": "FEMALE_BREAST_EXPOSED", "score": 0.5}], True),
            ([{"label": "MALE_GENITALIA_EXPOSED", "score": 0.9}], True),
            ([{"class": "FEMALE_BREAST_EXPOSED", "score": 0.49}], False),
            ([{"class": "FACE_FEMALE", "score": 0.99}], False),
        ]
        for predictions, expected in cases:
            with self.subTest(predictions=predictions):
                detector.detect.return_value = predictions
                self.assertIs(namespace["detect_image_nudity"](image, "custom.onnx"), expected)
                self.assertFalse(Path(detector.detect.call_args.args[0]).exists())
        namespace["_get_detector"].assert_called_with("custom.onnx")
        detector.detect.side_effect = RuntimeError("inference failed")
        with self.assertRaises(RuntimeError):
            namespace["detect_image_nudity"](image)
        self.assertFalse(Path(detector.detect.call_args.args[0]).parent.exists())

    def test_face_checks_and_nudity_results(self):
        for filename in ("app.py", "worker.py"):
            # OpenCV's detection tensor supports tuple indexing.
            predictions = MagicMock()
            predictions.shape = (1, 1, 1, 7)
            predictions.__getitem__.return_value = 0.99
            net = Mock()
            net.forward.return_value = predictions
            nudity = Mock()
            namespace = dict(io=io, os=os, Image=Mock(), np=Mock(), cv2=Mock(), NET=net,
                             CONF_THRESH=0.65, MAX_BYTES=1024, detect_image_nudity=nudity)
            detect = load_functions(filename, {"detect_faces_bytes"}, namespace)["detect_faces_bytes"]
            for flag in (True, False):
                with self.subTest(filename=filename, nudity=flag):
                    nudity.return_value = flag
                    self.assertEqual(detect(b"image"), {
                        "faces": 1, "is_face": True, "nudity": flag, "status": 200})
            nudity.side_effect = RuntimeError("model unavailable")
            self.assertEqual(detect(b"image")["status"], 500)
            nudity.reset_mock()
            predictions.__getitem__.return_value = 0.1
            self.assertEqual(detect(b"image")["status"], 422)
            nudity.assert_not_called()
            self.assertEqual(detect(b"")["status"], 400)
            self.assertEqual(detect(b"123", max_bytes=2)["status"], 413)

    def test_sync_fallback_and_completed_job_responses(self):
        class HTTPError(Exception):
            def __init__(self, status_code, detail):
                self.status_code = status_code

        class Upload:
            async def read(self):
                return b"image"

        result = {"faces": 1, "is_face": True, "nudity": True, "status": 200}
        namespace = dict(UploadFile=Upload, File=Mock(), Header=Mock(), HTTPException=HTTPError,
                         _check_api_key=Mock(), get_queue=Mock(return_value=None),
                         detect_faces_bytes=Mock(return_value=result), CONF_THRESH=0.65, MAX_BYTES=1024)
        load_functions("app.py", {"detect_face_sync", "detect_face_async", "get_result"}, namespace)
        expected = {"faces": 1, "is_face": True, "nudity": True}
        self.assertEqual(asyncio.run(namespace["detect_face_sync"](Upload())), expected)
        self.assertEqual(asyncio.run(namespace["detect_face_async"](Upload())),
                         dict(expected, mode="sync_fallback"))
        queue = Mock()
        job = Mock(is_finished=True, is_failed=False, result=result)
        job.get_id.return_value = "job-123"
        queue.fetch_job.return_value = job
        queue.enqueue.return_value = job
        namespace["get_queue"].return_value = queue
        self.assertEqual(asyncio.run(namespace["detect_face_async"](Upload())),
                         {"job_id": "job-123", "status": "queued"})
        self.assertEqual(namespace["get_result"]("job-123"), dict(expected, status="done"))
        namespace["detect_faces_bytes"].return_value = {"error": "Nudity detection failed", "status": 500}
        with self.assertRaises(HTTPError) as error:
            asyncio.run(namespace["detect_face_sync"](Upload()))
        self.assertEqual(error.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()
