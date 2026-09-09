from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import sys
import unittest


# scanner imports ExifRead at module load; filename parsing itself does not use
# it, so a tiny stub keeps this pure regression test runnable in a source tree.
sys.modules.setdefault("exifread", SimpleNamespace())

from app.analysis.grouping import _camera_attention_guard, _camera_attention_rank, select_group_series
from app.core.models import FaceAssessment, FrameAssessment, PhotoFile
from app.core.scanner import _sequence_number, _sequence_source
from app.xmp.writer import XmpWriter


def attention_face(score: float, reliable: bool = True):
    return SimpleNamespace(
        camera_attention_score=score,
        camera_attention_reliable=reliable,
    )


class FilenameSequenceTests(unittest.TestCase):
    def test_edit_suffix_does_not_split_camera_sequence(self):
        first = Path("shoot/IMG_0001_edit.jpg")
        second = Path("shoot/IMG_0002_final.jpg")
        self.assertEqual(_sequence_number(first), 1)
        self.assertEqual(_sequence_number(second), 2)
        self.assertEqual(_sequence_source(first), _sequence_source(second))

    def test_long_counter_beats_short_version_suffix(self):
        path = Path("shoot/DSC01234-retouch-v2.jpg")
        self.assertEqual(_sequence_number(path), 1234)

    def test_rightmost_counter_sized_number_is_preferred(self):
        path = Path("shoot/shoot_2026_0042_web.jpg")
        self.assertEqual(_sequence_number(path), 42)

    def test_year_does_not_beat_a_short_explicit_index(self):
        path = Path("shoot/2026_portrait_7_edit.jpg")
        self.assertEqual(_sequence_number(path), 7)

    def test_phone_date_does_not_replace_the_time_counter(self):
        first = Path("shoot/PXL_20260909_120001.jpg")
        second = Path("shoot/PXL_20260909_120002.jpg")
        self.assertEqual(_sequence_number(first), 120001)
        self.assertEqual(_sequence_number(second), 120002)
        self.assertEqual(_sequence_source(first), _sequence_source(second))

    def test_img_date_does_not_beat_the_explicit_frame_counter(self):
        self.assertEqual(_sequence_number(Path("shoot/IMG_20260909_0001.jpg")), 1)


class DependencyPinTests(unittest.TestCase):
    def test_every_requirement_is_an_exact_pin(self):
        requirements = Path("app/requirements.txt").read_text(encoding="utf-8")
        packages = [
            line.strip() for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertTrue(packages)
        self.assertTrue(all(line.count("==") == 1 for line in packages))


class CameraAttentionTests(unittest.TestCase):
    def setUp(self):
        self.track_ids = [1, 2, 3, 4]
        self.config = {"group": {
            "camera_attention_away_threshold": 0.42,
            "camera_attention_acceptable_threshold": 0.52,
            "camera_attention_min_known_fraction": 0.60,
        }}

    def test_protected_gaze_precedes_fbp_inside_same_suitability_tier(self):
        matrix = {
            0: {1: attention_face(0.20), 2: attention_face(0.75), 3: attention_face(0.75), 4: attention_face(0.75)},
            1: {1: attention_face(0.70), 2: attention_face(0.70), 3: attention_face(0.70), 4: attention_face(0.70)},
        }
        better_fbp = (0,) * 10 + (4, 0.90, 0.90, 3.60, 0.80, 4, 4, 0.80, 4)
        weaker_fbp = (0,) * 10 + (3, 0.70, 0.70, 2.80, 0.75, 4, 4, 0.75, 4)
        rank_away = _camera_attention_rank(
            0, 0.80, better_fbp, self.track_ids, matrix, self.config,
            use_preference=True,
        )
        rank_looking = _camera_attention_rank(
            1, 0.75, weaker_fbp, self.track_ids, matrix, self.config,
            use_preference=True,
        )
        self.assertGreater(rank_looking, rank_away)
        # With the UI option disabled select_group_series never calls the gaze
        # rank; the ordinary base order therefore still selects better_fbp.
        self.assertGreater(better_fbp, weaker_fbp)

    def test_unknown_faces_are_not_counted_as_confirmed_away(self):
        matrix = {0: {
            1: attention_face(0.75),
            2: attention_face(0.75),
            3: attention_face(0.50, reliable=False),
            4: attention_face(0.50, reliable=False),
        }}
        tier, acceptable, reliable = _camera_attention_guard(
            0, self.track_ids, matrix, self.config
        )
        self.assertEqual((tier, acceptable, reliable), (1, 2, 2))

    def test_disabled_ui_option_preserves_the_base_fbp_winner(self):
        def full_face(person: int, preference: float, gaze: float, gaze_reliable: bool):
            descriptor = [0.0] * 4
            descriptor[person] = 1.0
            return FaceAssessment(
                bbox=(person * 100, 0, person * 100 + 80, 100),
                center=((person + 0.5) / 4.0, 0.5), size_fraction=0.05,
                eye_open_left=0.9, eye_open_right=0.9, smile=0.5,
                expression=0.5, face_sharpness=0.9, eye_sharpness=0.9,
                technical=0.9, quality=0.9,
                portrait_preference_score=preference,
                portrait_preference_raw=3.0,
                portrait_preference_reliable=True,
                descriptor=descriptor, landmarks_reliable=True,
                detection_confidence=1.0, descriptor_source="insightface",
                camera_attention_score=gaze,
                camera_attention_reliable=gaze_reliable,
                head_pose_confidence=1.0,
            )

        def full_frame(index: int, preference: float, gazes=None):
            photo = PhotoFile(
                Path(f"IMG_{index:04d}.jpg"), datetime(2026, 1, 1), index, ".jpg"
            )
            faces = [
                full_face(person, preference, gazes[person] if gazes else 0.5, gazes is not None)
                for person in range(4)
            ]
            return FrameAssessment(photo=photo, faces=faces, technical=0.9)

        frames = [full_frame(1, 0.9), full_frame(2, 0.6)]
        high_resolution = {
            0: full_frame(1, 0.9, [0.2, 0.8, 0.8, 0.8]),
            1: full_frame(2, 0.6, [0.8, 0.8, 0.8, 0.8]),
        }
        config = {
            "analysis": {"face_min_fraction": 0.0005},
            "group": {
                "min_people": 4, "min_track_presence": 2,
                "match_threshold": 0.32, "portrait_preference_enabled": True,
                "portrait_preference_min_known_fraction": 0.5,
                "portrait_preference_min_relative_span": 0.08,
                "camera_attention_enabled": False,
                "find_headswap_candidates": False,
            },
        }
        disabled, _ = select_group_series(
            frames, config, attention_frames=high_resolution, diagnostic_log=False
        )
        self.assertEqual(disabled.main.photo.path.name, "IMG_0001.jpg")
        config["group"]["camera_attention_enabled"] = True
        enabled, _ = select_group_series(
            frames, config, attention_frames=high_resolution, diagnostic_log=False
        )
        self.assertEqual(enabled.main.photo.path.name, "IMG_0002.jpg")

    def test_successful_second_batch_remains_in_the_final_shortlist(self):
        def full_face(person: int, preference: float, gaze: float, reliable: bool):
            descriptor = [0.0] * 4
            descriptor[person] = 1.0
            return FaceAssessment(
                bbox=(person * 100, 0, person * 100 + 80, 100),
                center=((person + 0.5) / 4.0, 0.5), size_fraction=0.05,
                eye_open_left=0.9, eye_open_right=0.9, smile=0.5,
                expression=0.5, face_sharpness=0.9, eye_sharpness=0.9,
                technical=0.9, quality=0.9,
                portrait_preference_score=preference,
                portrait_preference_raw=3.0,
                portrait_preference_reliable=True,
                descriptor=descriptor, landmarks_reliable=True,
                detection_confidence=1.0, descriptor_source="insightface",
                camera_attention_score=gaze,
                camera_attention_reliable=reliable,
                head_pose_confidence=1.0,
            )

        def frame(index: int, preference: float, gazes=None):
            photo = PhotoFile(
                Path(f"IMG_{index:04d}.jpg"), datetime(2026, 1, 1), index, ".jpg"
            )
            values = gazes if gazes is not None else [0.5] * 4
            return FrameAssessment(
                photo=photo,
                faces=[full_face(i, preference, values[i], gazes is not None) for i in range(4)],
                technical=0.9,
            )

        preferences = [0.99, 0.98, 0.97, 0.96, 0.95, 0.60]
        frames = [frame(i + 1, value) for i, value in enumerate(preferences)]
        high_resolution = {
            i: frame(
                i + 1,
                value,
                [0.8, 0.8, 0.8, 0.8] if i == 5 else [0.2, 0.8, 0.8, 0.8],
            )
            for i, value in enumerate(preferences)
        }
        config = {
            "analysis": {"face_min_fraction": 0.0005},
            "group": {
                "min_people": 4, "min_track_presence": 2,
                "match_threshold": 0.32, "portrait_preference_enabled": True,
                "portrait_preference_min_known_fraction": 0.5,
                "portrait_preference_min_relative_span": 0.08,
                "camera_attention_enabled": True,
                "camera_attention_shortlist": 5,
                "camera_attention_away_threshold": 0.42,
                "camera_attention_acceptable_threshold": 0.52,
                "camera_attention_min_known_fraction": 0.60,
                "find_headswap_candidates": False,
            },
        }

        _first, first_diag = select_group_series(
            frames,
            config,
            attention_frames={i: high_resolution[i] for i in range(5)},
            diagnostic_log=False,
        )
        self.assertIn(5, first_diag.camera_attention_shortlist_indices)

        selected, final_diag = select_group_series(
            frames, config, attention_frames=high_resolution, diagnostic_log=False
        )
        self.assertIn(5, final_diag.camera_attention_shortlist_indices)
        self.assertEqual(selected.main.photo.path.name, "IMG_0006.jpg")


class ForcedYellowTests(unittest.TestCase):
    @staticmethod
    def _frame(index: int, frame_technical: float) -> FrameAssessment:
        faces = []
        for person in range(4):
            descriptor = [0.0] * 4
            descriptor[person] = 1.0
            faces.append(FaceAssessment(
                bbox=(person * 100, 0, person * 100 + 80, 100),
                center=((person + 0.5) / 4.0, 0.5), size_fraction=0.05,
                eye_open_left=0.9, eye_open_right=0.9, smile=0.5,
                expression=0.5, face_sharpness=0.9, eye_sharpness=0.9,
                technical=0.9, quality=0.9,
                portrait_preference_score=0.95 - index * 0.05,
                portrait_preference_raw=3.0,
                portrait_preference_reliable=True,
                descriptor=descriptor, landmarks_reliable=True,
                detection_confidence=1.0, descriptor_source="insightface",
                head_pose_confidence=1.0,
            ))
        return FrameAssessment(
            photo=PhotoFile(
                Path(f"IMG_{index:04d}.jpg"), datetime(2026, 1, 1), index, ".jpg"
            ),
            faces=faces,
            technical=frame_technical,
        )

    @staticmethod
    def _config(minimum: int, maximum: int = 2) -> dict:
        return {
            "analysis": {"face_min_fraction": 0.0005},
            "group": {
                "min_people": 4, "min_track_presence": 2,
                "match_threshold": 0.32, "portrait_preference_enabled": True,
                "portrait_preference_min_known_fraction": 0.5,
                "portrait_preference_min_relative_span": 0.08,
                "camera_attention_enabled": False,
                "find_headswap_candidates": True,
                "min_extra_candidates": minimum,
                "max_extra_candidates": maximum,
            },
        }

    def test_zero_minimum_is_automatic_and_adds_no_unneeded_backup(self):
        frames = [self._frame(1, 0.9), self._frame(2, 0.2), self._frame(3, 0.2)]
        selected, _diag = select_group_series(
            frames, self._config(0), diagnostic_log=False
        )
        self.assertEqual(selected.main.photo.path.name, "IMG_0001.jpg")
        self.assertEqual(selected.extras, [])

    def test_positive_minimum_is_filled_from_best_remaining_frames(self):
        frames = [self._frame(1, 0.9), self._frame(2, 0.2), self._frame(3, 0.2)]
        selected, diag = select_group_series(
            frames, self._config(2), diagnostic_log=False
        )
        self.assertEqual(selected.main.photo.path.name, "IMG_0001.jpg")
        self.assertEqual(len(selected.extras), 2)
        self.assertEqual(diag.backup_extras, 2)
        self.assertTrue(all(extra.label_role == "yellow" for extra in selected.extras))

    def test_maximum_remains_a_hard_cap(self):
        frames = [self._frame(i, 0.9) for i in range(1, 6)]
        selected, _diag = select_group_series(
            frames, self._config(5, maximum=2), diagnostic_log=False
        )
        self.assertEqual(len(selected.extras), 2)


class XmpCleanupTests(unittest.TestCase):
    def test_switching_to_lightroom_clears_old_bridge_labels(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            photo_path = Path(directory) / "IMG_0001.nef"
            photo_path.touch()
            sidecar = photo_path.with_suffix(".xmp")
            sidecar.write_text(
                '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
                'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                '<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
                'xmp:Label="Select"/></rdf:RDF></x:xmpmeta>',
                encoding="utf-8",
            )
            config = {"xmp": {
                "scheme": "lightroom", "bridge_red": "Select",
                "bridge_yellow": "Second", "lightroom_red": "Red",
                "lightroom_yellow": "Yellow", "custom_red": "Select",
                "custom_yellow": "Second", "jpeg_embedded": True,
                "update_existing_jpeg_sidecar": True,
            }}
            photo = PhotoFile(photo_path, datetime(2026, 1, 1), 1, ".nef")
            writer = XmpWriter(config)
            self.assertTrue(writer.clear_label(photo, "red"))
            self.assertNotIn("Select", sidecar.read_text(encoding="utf-8"))

            sidecar.write_text(
                sidecar.read_text(encoding="utf-8").replace(
                    "</rdf:Description>", ""
                ).replace(
                    "/>", ' xmp:Label="Second"/>'
                ),
                encoding="utf-8",
            )
            self.assertTrue(writer.clear_label(photo, "yellow"))
            self.assertNotIn("Second", sidecar.read_text(encoding="utf-8"))

            unrelated = sidecar.read_text(encoding="utf-8").replace(
                "/>", ' xmp:Label="Green"/>'
            )
            sidecar.write_text(unrelated, encoding="utf-8")
            self.assertFalse(writer.clear_label(photo, "red"))
            self.assertFalse(writer.clear_label(photo, "yellow"))
            self.assertIn("Green", sidecar.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
