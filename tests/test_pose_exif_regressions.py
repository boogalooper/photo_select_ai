import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import sys
import cv2
import numpy as np

sys.modules.setdefault('exifread', SimpleNamespace())
from app.analysis.face_insightface import _head_pose_metrics
from app.core import scanner


class PoseTests(unittest.TestCase):
    def test_frontal_and_known_turns(self):
        model = np.array([(0,0,0),(0,330,65),(-225,-170,135),
                          (225,-170,135),(-150,150,125),(150,150,125)], dtype=float)
        camera = np.array([[2000,0,1000],[0,2000,500],[0,0,1]], dtype=float)
        for pitch, yaw in [(0,0),(15,0),(-15,0),(0,30),(0,-30)]:
            with self.subTest(pitch=pitch, yaw=yaw):
                points, _ = cv2.projectPoints(model, np.deg2rad([pitch,yaw,0.]),
                    np.array([0.,0.,4000.]), camera, np.zeros(4))
                landmarks = np.zeros((106,2))
                landmarks[[86,0,35,93,52,61]] = points.reshape(-1,2)
                _, confidence, actual_yaw, actual_pitch = _head_pose_metrics(landmarks,(1000,2000,3),{})
                self.assertGreater(confidence, .9)
                self.assertAlmostEqual(actual_yaw,yaw,places=3)
                self.assertAlmostEqual(actual_pitch,pitch,places=3)


class ExifTests(unittest.TestCase):
    def test_call_uses_compatible_arguments(self):
        def process_file(fh, details=True, stop_tag='UNDEF'):
            return {'EXIF DateTimeOriginal':'2026:09:09 12:34:56'}
        with TemporaryDirectory() as directory:
            path = Path(directory)/'1А IMG_1719.CR2'
            path.touch()
            with patch.object(scanner.exifread,'process_file',process_file,create=True):
                self.assertEqual(scanner._read_exif_capture_time(path).year,2026)

    def test_api_error_aborts_instead_of_skipping_photos(self):
        for workers in (1,2):
            with patch.object(scanner,'_read_exif_capture_time',side_effect=TypeError('API mismatch')):
                with self.assertRaises(TypeError):
                    scanner._read_exif_indices(
                        [SimpleNamespace(path=Path('a.CR2')),SimpleNamespace(path=Path('b.CR2'))],
                        {0,1},workers,scanner.ScanReport(),None)
