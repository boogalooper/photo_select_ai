from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class PhotoFile:
    path: Path
    capture_time: datetime
    sequence_number: Optional[int]
    extension: str
    capture_time_source: str = "file"
    order_source: str = "time"
    metadata_cached: bool = False
    sequence_source: tuple[str, str] = ("", "")


@dataclass(slots=True)
class FaceAssessment:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    size_fraction: float
    eye_open_left: float
    eye_open_right: float
    smile: float
    expression: float
    face_sharpness: float
    eye_sharpness: float
    technical: float
    quality: float
    descriptor: list[float] = field(default_factory=list)
    landmarks_reliable: bool = True
    detection_confidence: float = 1.0
    descriptor_source: str = "legacy"
    # Experimental group-camera attention metrics.  They are deliberately
    # optional/neutral so older tests and portrait logic remain unchanged.
    camera_attention_score: float = 0.50
    camera_attention_confidence: float = 0.0
    head_frontal_score: float = 0.50
    camera_attention_reliable: bool = False
    # Generic head pose estimated from the same 106 landmarks.  Portrait
    # repeat-pose selection uses this only as a conservative cue that two
    # already-linked series are visibly different; it never affects identity.
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_pose_confidence: float = 0.0

    @property
    def eyes_open_score(self) -> float:
        return min(self.eye_open_left, self.eye_open_right)


@dataclass(slots=True)
class FrameAssessment:
    photo: PhotoFile
    faces: list[FaceAssessment]
    technical: float
    error: Optional[str] = None

    @property
    def primary_face(self) -> Optional[FaceAssessment]:
        if not self.faces:
            return None

        # Portrait shoots sometimes contain a parent/assistant/background face.
        # Prefer a large face, but add a modest centrality bonus so the main
        # subject is less likely to switch between adjacent frames.
        def subject_score(face: FaceAssessment) -> float:
            dx = face.center[0] - 0.5
            dy = face.center[1] - 0.5
            dist = min(1.0, (dx * dx + dy * dy) ** 0.5 / 0.7071)
            centrality = 1.0 - dist
            return face.size_fraction * (0.78 + 0.22 * centrality)

        return max(self.faces, key=subject_score)


@dataclass(slots=True)
class PhotoSeries:
    index: int
    photos: list[PhotoFile]


@dataclass(slots=True)
class Selection:
    photo: PhotoFile
    label_role: str  # red for main selection; yellow for group backups or distinct portrait poses
    score: float
    reason: str


@dataclass(slots=True)
class RunStats:
    run_mode: str = "portrait"
    files_found: int = 0
    files_analyzed: int = 0
    candidate_series: int = 0
    refined_series: int = 0
    series_processed: int = 0
    portrait_series: int = 0
    portrait_selected: int = 0
    portrait_repeat_children: int = 0
    portrait_repeat_links: int = 0
    portrait_repeat_ambiguous: int = 0
    portrait_repeat_yellow_selected: int = 0
    group_series: int = 0
    group_main_selected: int = 0
    group_extra_selected: int = 0
    group_tracks_confirmed: int = 0
    group_blocks_merged: int = 0
    group_identity_splits: int = 0
    group_eye_problems: int = 0
    group_missing_problems: int = 0
    group_sharpness_problems: int = 0
    group_quality_problems: int = 0
    group_camera_attention_known: int = 0
    group_camera_attention_away: int = 0
    group_camera_attention_series: int = 0
    group_problems_covered: int = 0
    group_problems_unresolved: int = 0
    group_backup_candidates: int = 0
    xmp_written: int = 0
    embedded_xmp_written: int = 0
    jpeg_embedded_written: int = 0
    sidecar_xmp_written: int = 0
    skipped_short_series: int = 0
    frames_without_faces: int = 0
    read_errors: int = 0
    analysis_errors: int = 0
    series_without_selection: int = 0
    weak_series_rejected: int = 0
    local_fragments_merged: int = 0
    cross_block_series_merged: int = 0
    portrait_boundary_guard_proposals: int = 0
    portrait_boundary_guard_splits: int = 0
    labels_cleared_before_run: int = 0
    scan_exif_reads: int = 0
    scan_exif_cache_hits: int = 0
    scan_time_from_exif: int = 0
    scan_time_from_file: int = 0
    scan_order_from_name: int = 0
    scan_exif_failures: int = 0
    scan_files_skipped: int = 0
    raw_preview_rawpy: int = 0
    raw_preview_embedded_jpeg: int = 0
    raw_preview_demosaic: int = 0
    raw_jpeg_pairs_collapsed: int = 0
    metadata_errors: int = 0

    @property
    def total_selected(self) -> int:
        return (
            self.portrait_selected
            + self.portrait_repeat_yellow_selected
            + self.group_main_selected
            + self.group_extra_selected
        )
