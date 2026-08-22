from datetime import datetime, timedelta
from pathlib import Path
import unittest

from app.core.models import FaceAssessment, FrameAssessment, PhotoFile
from app.core.series import build_candidate_series, refine_assessments


def fake_frame(name: str, descriptor: list[float] | None, second: int = 0) -> FrameAssessment:
    photo = PhotoFile(Path(name), datetime(2026, 1, 1, 10, 0, second), second, ".jpg")
    if descriptor is None:
        return FrameAssessment(photo, [], 0.7)
    face = FaceAssessment(
        (0, 0, 100, 100), (0.5, 0.5), 0.20,
        0.9, 0.9, 0.2, 0.7, 0.7, 0.7, 0.7, 0.7,
        descriptor,
        descriptor_source="insightface",
    )
    return FrameAssessment(photo, [face], 0.7)


PORTRAIT_CFG = {
    "series": {
        "portrait_algorithm": "dbscan",
        "portrait_dbscan_distance": 0.27,
        "portrait_dbscan_min_samples": 2,
        "portrait_dbscan_noise_attach_factor": 1.15,
        "portrait_no_face_tolerance": 2,
        # Legacy fallback values remain available.
        "portrait_change_sensitivity": 45,
        "portrait_break_confirm_frames": 2,
    }
}


class SeriesTests(unittest.TestCase):
    def test_time_break(self):
        t = datetime(2026, 1, 1, 10, 0, 0)
        photos = [
            PhotoFile(Path("IMG_0001.CR3"), t, 1, ".cr3"),
            PhotoFile(Path("IMG_0002.CR3"), t + timedelta(seconds=1), 2, ".cr3"),
            PhotoFile(Path("IMG_0003.CR3"), t + timedelta(seconds=12), 3, ".cr3"),
        ]
        cfg = {"series": {"max_gap_seconds": 5, "max_filename_gap": 3}}
        groups = build_candidate_series(photos, cfg)
        self.assertEqual([len(x.photos) for x in groups], [2, 1])


    def test_group_and_portrait_candidate_gaps_are_independent(self):
        t = datetime(2026, 1, 1, 10, 0, 0)
        photos = [
            PhotoFile(Path("IMG_0001.JPG"), t, 1, ".jpg"),
            PhotoFile(Path("IMG_0002.JPG"), t + timedelta(seconds=8), 2, ".jpg"),
        ]
        base = {
            "series": {"max_gap_seconds": 5, "max_filename_gap": 3},
            "group": {"max_gap_seconds": 12, "max_filename_gap": 5},
        }
        portrait_cfg = {**base, "runtime": {"mode": "portrait"}}
        group_cfg = {**base, "runtime": {"mode": "group"}}
        self.assertEqual([len(x.photos) for x in build_candidate_series(photos, portrait_cfg)], [1, 1])
        self.assertEqual([len(x.photos) for x in build_candidate_series(photos, group_cfg)], [2])

    def test_missing_face_does_not_split_same_child(self):
        frames = [
            fake_frame("1.jpg", [1.0, 0.0], 1),
            fake_frame("2.jpg", [1.0, 0.0], 2),
            fake_frame("3.jpg", None, 3),
            fake_frame("4.jpg", [0.98, 0.02], 4),
            fake_frame("5.jpg", [1.0, 0.0], 5),
        ]
        groups = refine_assessments(frames, PORTRAIT_CFG)
        self.assertEqual([len(g) for g in groups], [5])

    def test_single_odd_frame_does_not_split_subject(self):
        frames = [
            fake_frame("1.jpg", [1.0, 0.0], 1),
            fake_frame("2.jpg", [1.0, 0.0], 2),
            fake_frame("3.jpg", [0.0, 1.0], 3),
            fake_frame("4.jpg", [1.0, 0.0], 4),
            fake_frame("5.jpg", [1.0, 0.0], 5),
        ]
        groups = refine_assessments(frames, PORTRAIT_CFG)
        self.assertEqual([len(g) for g in groups], [5])

    def test_confirmed_new_subject_splits(self):
        frames = [
            fake_frame("1.jpg", [1.0, 0.0], 1),
            fake_frame("2.jpg", [1.0, 0.0], 2),
            fake_frame("3.jpg", [0.0, 1.0], 3),
            fake_frame("4.jpg", [0.0, 1.0], 4),
            fake_frame("5.jpg", [0.02, 0.98], 5),
        ]
        groups = refine_assessments(frames, PORTRAIT_CFG)
        self.assertEqual([len(g) for g in groups], [2, 3])

    def test_same_child_later_is_new_chronological_series(self):
        # Identity A returns after B. DBSCAN recognises A globally, but the
        # chronological pass must still create A / B / A as three series.
        frames = [
            fake_frame("1.jpg", [1.0, 0.0], 1),
            fake_frame("2.jpg", [0.99, 0.01], 2),
            fake_frame("3.jpg", [0.0, 1.0], 3),
            fake_frame("4.jpg", [0.01, 0.99], 4),
            fake_frame("5.jpg", [1.0, 0.0], 5),
            fake_frame("6.jpg", [0.99, 0.01], 6),
        ]
        groups = refine_assessments(frames, PORTRAIT_CFG)
        self.assertEqual([len(g) for g in groups], [2, 2, 2])


if __name__ == "__main__":
    unittest.main()

class SeriesV025Tests(unittest.TestCase):
    def _cfg(self):
        cfg = {"series": dict(PORTRAIT_CFG["series"]), "analysis": {"face_min_fraction": 0.0005}}
        cfg["series"].update({
            "portrait_segment_merge_distance": 0.32,
            "portrait_min_confirmed_frames": 2,
            "portrait_confirm_det_thresh": 0.45,
            "portrait_confirm_min_face_fraction": 0.0005,
            "portrait_cross_block_merge_seconds": 45.0,
            "portrait_cross_block_merge_distance": 0.30,
        })
        return cfg

    def test_isolated_false_face_is_rejected(self):
        cfg = self._cfg()
        frames = [
            fake_frame("1.jpg", None, 1),
            fake_frame("2.jpg", [0.0, 1.0], 2),
            fake_frame("3.jpg", None, 3),
        ]
        groups = refine_assessments(frames, cfg)
        self.assertEqual(groups, [])

    def test_adjacent_dbscan_fragments_of_same_child_merge(self):
        cfg = self._cfg()
        # Two stable clusters are just outside DBSCAN eps (0.27) but inside the
        # neighbouring-fragment merge distance (0.32).
        frames = [
            fake_frame("1.jpg", [1.0, 0.0], 1),
            fake_frame("2.jpg", [0.999, 0.02], 2),
            fake_frame("3.jpg", [0.71, 0.704], 3),
            fake_frame("4.jpg", [0.70, 0.714], 4),
        ]
        groups = refine_assessments(frames, cfg)
        self.assertEqual([len(g) for g in groups], [4])

    def test_leading_floor_frames_are_not_attached_to_child(self):
        cfg = self._cfg()
        frames = [
            fake_frame("1.jpg", None, 1),
            fake_frame("2.jpg", None, 2),
            fake_frame("3.jpg", [1.0, 0.0], 3),
            fake_frame("4.jpg", [0.99, 0.01], 4),
            fake_frame("5.jpg", None, 5),
        ]
        groups = refine_assessments(frames, cfg)
        self.assertEqual([[f.photo.path.name for f in g] for g in groups], [["3.jpg", "4.jpg"]])

    def test_cross_block_same_child_merges_but_not_through_other_child(self):
        from app.core.series import merge_adjacent_portrait_series
        cfg = self._cfg()
        a1 = [fake_frame("1.jpg", [1.0, 0.0], 1), fake_frame("2.jpg", [0.99, 0.01], 2)]
        a2 = [fake_frame("20.jpg", [0.98, 0.02], 20), fake_frame("21.jpg", [1.0, 0.0], 21)]
        merged, count = merge_adjacent_portrait_series([a1, a2], cfg)
        self.assertEqual(count, 1)
        self.assertEqual([len(g) for g in merged], [4])

        b = [fake_frame("10.jpg", [0.0, 1.0], 10), fake_frame("11.jpg", [0.01, 0.99], 11)]
        merged, count = merge_adjacent_portrait_series([a1, b, a2], cfg)
        self.assertEqual(count, 0)
        self.assertEqual([len(g) for g in merged], [2, 2, 2])


class PortraitBoundaryGuardTests(unittest.TestCase):
    def _cfg(self, algorithm: str = "sequential", enabled: bool = True):
        return {
            "series": {
                "portrait_algorithm": algorithm,
                "portrait_same_person_similarity": 0.42,
                "portrait_break_confirm_frames": 2,
                "portrait_no_face_tolerance": 2,
                "portrait_dbscan_distance": 0.27,
                "portrait_dbscan_min_samples": 2,
                "portrait_dbscan_noise_attach_factor": 1.12,
                "portrait_segment_merge_distance": 0.32,
                "portrait_min_confirmed_frames": 2,
                "portrait_confirm_det_thresh": 0.45,
                "portrait_confirm_min_face_fraction": 0.0005,
                "portrait_cross_block_merge_seconds": 45.0,
                "portrait_cross_block_merge_distance": 0.30,
                "portrait_boundary_guard_enabled": enabled,
                "portrait_boundary_guard_window": 4,
                "portrait_boundary_guard_min_evidence": 2,
                "portrait_boundary_guard_min_confirmed_each": 1,
                "portrait_boundary_guard_min_segment_frames": 2,
                "portrait_boundary_guard_min_centroid_distance": 0.16,
                "portrait_boundary_guard_min_identity_margin": 0.10,
                "portrait_boundary_guard_min_vote_fraction": 0.75,
                "portrait_boundary_guard_min_cohesion": 0.80,
            },
            "analysis": {"face_min_fraction": 0.0005},
        }

    def test_guard_recovers_boundary_missed_by_sequential(self):
        # Similarity A<->B is 0.60: sequential's permissive 0.42 threshold
        # treats the whole run as one child, while DBSCAN proposes a split.
        frames = [
            fake_frame("1.jpg", [1.0, 0.0], 1),
            fake_frame("2.jpg", [0.999, 0.02], 2),
            fake_frame("3.jpg", [0.60, 0.80], 3),
            fake_frame("4.jpg", [0.61, 0.79], 4),
        ]
        without_guard = refine_assessments(frames, self._cfg(enabled=False))
        with_guard = refine_assessments(frames, self._cfg(enabled=True))
        self.assertEqual([len(g) for g in without_guard], [4])
        self.assertEqual([len(g) for g in with_guard], [2, 2])

    def test_guard_does_not_turn_single_bad_embedding_into_child(self):
        frames = [
            fake_frame("1.jpg", [1.0, 0.0], 1),
            fake_frame("2.jpg", [0.99, 0.01], 2),
            fake_frame("3.jpg", [0.0, 1.0], 3),
            fake_frame("4.jpg", [0.99, 0.01], 4),
            fake_frame("5.jpg", [1.0, 0.0], 5),
        ]
        groups = refine_assessments(frames, self._cfg(enabled=True))
        self.assertEqual([len(g) for g in groups], [5])

    def test_guard_preserves_verified_boundary_during_cross_block_merge(self):
        from app.core.series import merge_adjacent_portrait_series
        cfg = self._cfg(enabled=True)
        # Make the normal cross-block merge permissive enough that it would
        # merge these series without the guard, then verify that the local
        # identity evidence vetoes that merge.
        cfg["series"]["portrait_cross_block_merge_distance"] = 0.50
        left = [fake_frame("1.jpg", [1.0, 0.0], 1), fake_frame("2.jpg", [0.99, 0.01], 2)]
        right = [fake_frame("3.jpg", [0.60, 0.80], 3), fake_frame("4.jpg", [0.61, 0.79], 4)]
        merged, count = merge_adjacent_portrait_series([left, right], cfg)
        self.assertEqual(count, 0)
        self.assertEqual([len(g) for g in merged], [2, 2])
