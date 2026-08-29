from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from app import __version__
from app.core.config import load_ui_state, merged_config, save_ui_state
from app.core.pipeline import AnalysisPipeline, CancelledError
from app.core.scanner import count_supported_photos
from app.gui.tooltip import ToolTip
from app.utils.hardware import detect_hardware


SELECTION_PROFILES: dict[str, dict[str, float | bool]] = {
    "Сбалансированный": {
        "eye_threshold": 0.52,
        "prefer_open_eyes": True,
        "eyes_weight": 1.80,
        "closed_eye_penalty": 0.45,
        "eye_sharpness_weight": 1.40,
        "face_sharpness_weight": 0.90,
        "expression_weight": 0.70,
        "smile_weight": 0.35,
        "technical_weight": 0.35,
    },
    "Глаза и фокус": {
        "eye_threshold": 0.52,
        "prefer_open_eyes": True,
        "eyes_weight": 2.20,
        "closed_eye_penalty": 0.65,
        "eye_sharpness_weight": 1.80,
        "face_sharpness_weight": 1.10,
        "expression_weight": 0.55,
        "smile_weight": 0.20,
        "technical_weight": 0.35,
    },
    "Выражение и улыбка": {
        "eye_threshold": 0.50,
        "prefer_open_eyes": True,
        "eyes_weight": 1.60,
        "closed_eye_penalty": 0.40,
        "eye_sharpness_weight": 1.20,
        "face_sharpness_weight": 0.80,
        "expression_weight": 1.15,
        "smile_weight": 0.80,
        "technical_weight": 0.30,
    },
    "Без приоритета улыбки": {
        "eye_threshold": 0.52,
        "prefer_open_eyes": True,
        "eyes_weight": 1.85,
        "closed_eye_penalty": 0.50,
        "eye_sharpness_weight": 1.45,
        "face_sharpness_weight": 0.95,
        "expression_weight": 0.80,
        "smile_weight": 0.00,
        "technical_weight": 0.35,
    },
}
SELECTION_CUSTOM = "Пользовательский"

SERIES_PROFILES: dict[str, dict[str, object]] = {
    "Сбалансированный": {
        "series_algorithm": "dbscan",
        "dbscan_distance": 0.27,
        "dbscan_min_samples": 2,
        "segment_merge_distance": 0.32,
        "min_confirmed_frames": 2,
        "confirm_det_thresh": 0.45,
        "confirm_face_min_pct": 0.05,
        "no_face_tolerance": 3,
        "gap": 12.0,
        "name_gap": 5,
        "cross_merge_seconds": 45.0,
        "cross_merge_distance": 0.30,
        "min_frames": 2,
        "det_thresh": 0.25,
        "det_size": 640,
        "face_min_pct": 0.05,
    },
    "Меньше повторов": {
        "series_algorithm": "dbscan",
        "dbscan_distance": 0.27,
        "dbscan_min_samples": 2,
        "segment_merge_distance": 0.34,
        "min_confirmed_frames": 2,
        "confirm_det_thresh": 0.45,
        "confirm_face_min_pct": 0.05,
        "no_face_tolerance": 3,
        "gap": 12.0,
        "name_gap": 5,
        "cross_merge_seconds": 60.0,
        "cross_merge_distance": 0.32,
        "min_frames": 2,
        "det_thresh": 0.25,
        "det_size": 640,
        "face_min_pct": 0.05,
    },
    "Строже к случайным кадрам": {
        "series_algorithm": "dbscan",
        "dbscan_distance": 0.27,
        "dbscan_min_samples": 2,
        "segment_merge_distance": 0.32,
        "min_confirmed_frames": 3,
        "confirm_det_thresh": 0.50,
        "confirm_face_min_pct": 0.08,
        "no_face_tolerance": 3,
        "gap": 12.0,
        "name_gap": 5,
        "cross_merge_seconds": 45.0,
        "cross_merge_distance": 0.30,
        "min_frames": 2,
        "det_thresh": 0.25,
        "det_size": 640,
        "face_min_pct": 0.05,
    },
    "Чувствительнее к сложным лицам": {
        "series_algorithm": "dbscan",
        "dbscan_distance": 0.28,
        "dbscan_min_samples": 2,
        "segment_merge_distance": 0.33,
        "min_confirmed_frames": 2,
        "confirm_det_thresh": 0.40,
        "confirm_face_min_pct": 0.03,
        "no_face_tolerance": 4,
        "gap": 16.0,
        "name_gap": 6,
        "cross_merge_seconds": 60.0,
        "cross_merge_distance": 0.31,
        "min_frames": 2,
        "det_thresh": 0.20,
        "det_size": 800,
        "face_min_pct": 0.03,
    },
}
SERIES_CUSTOM = "Пользовательский"

GROUP_RULE_PROFILES: dict[str, dict[str, float | int | bool]] = {
    "Глаза прежде всего": {
        "prioritize_open_eyes_main": True,
        "eye_problem_threshold": 0.62,
        "eye_candidate_threshold": 0.68,
        "eye_improvement_margin": 0.06,
        "good_face_threshold": 0.62,
        "headswap_min_eye_sharpness": 0.35,
        "headswap_candidate_min_quality": 0.52,
        "quality_improvement_margin": 0.10,
        "backup_min_score_ratio": 0.86,
        "backup_person_improvement_margin": 0.05,
        "min_extra_candidates": 1,
        "max_extra_candidates": 3,
    },
    "Сбалансированный": {
        "prioritize_open_eyes_main": True,
        "eye_problem_threshold": 0.58,
        "eye_candidate_threshold": 0.64,
        "eye_improvement_margin": 0.08,
        "good_face_threshold": 0.62,
        "headswap_min_eye_sharpness": 0.35,
        "headswap_candidate_min_quality": 0.52,
        "quality_improvement_margin": 0.12,
        "backup_min_score_ratio": 0.88,
        "backup_person_improvement_margin": 0.06,
        "min_extra_candidates": 1,
        "max_extra_candidates": 3,
    },
    "Больше резервных кадров": {
        "prioritize_open_eyes_main": True,
        "eye_problem_threshold": 0.60,
        "eye_candidate_threshold": 0.66,
        "eye_improvement_margin": 0.06,
        "good_face_threshold": 0.60,
        "headswap_min_eye_sharpness": 0.32,
        "headswap_candidate_min_quality": 0.48,
        "quality_improvement_margin": 0.08,
        "backup_min_score_ratio": 0.80,
        "backup_person_improvement_margin": 0.04,
        "min_extra_candidates": 2,
        "max_extra_candidates": 3,
    },
}
GROUP_PROFILE_DEFAULT = "Глаза прежде всего"

PORTRAIT_REPEAT_MODES = {
    "Обычный — RED на каждую серию": "off",
    "Один ребёнок / разные позы — лучший RED + позы YELLOW": "best_red_pose_yellow",
}
PORTRAIT_REPEAT_CODES = {value: label for label, value in PORTRAIT_REPEAT_MODES.items()}

class MainWindow(tk.Tk):
    def __init__(self, config: dict, initial_folder: str | None = None):
        super().__init__()
        self.title(f"Photo Select AI v{__version__} — портреты и группы")
        # Group quick-start has two additional option rows.  Use a taller
        # default on normal desktop displays, but never force the window beyond
        # the usable height of a smaller screen.
        screen_h = max(600, int(self.winfo_screenheight()))
        window_h = max(560, min(800, screen_h - 100))
        self.geometry(f"1160x{window_h}")
        self.minsize(900, min(600, window_h))
        self.base_config = config
        self.state_data = load_ui_state()
        if int(self.state_data.get("state_version", 1)) < 3:
            self.state_data = {
                key: self.state_data[key]
                for key in ("folder", "scheme", "custom_red")
                if key in self.state_data
            }
        if int(self.state_data.get("state_version", 1)) < 4:
            if str(self.state_data.get("cuda_conv_algo", "")).upper() == "DEFAULT":
                self.state_data["cuda_conv_algo"] = "HEURISTIC"
        if int(self.state_data.get("state_version", 1)) < 5:
            if int(self.state_data.get("min_frames", 1)) <= 1:
                self.state_data["min_frames"] = int(config["series"].get("min_frames", 2))

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self._folder_check_after: str | None = None
        self._folder_check_token = 0
        self._folder_is_valid = False
        self._folder_photo_count = 0
        self._numeric_specs: list[tuple[tk.Variable, str, float, float]] = []
        self._applying_profile = False
        self._advanced_visible = bool(self.state_data.get("advanced_visible", False))
        # Widgets are grouped by applicability so the UI can visibly disable
        # settings that do not affect the current mode/algorithm.
        self._dbscan_only_widgets: list[tk.Widget] = []
        self._sequential_only_widgets: list[tk.Widget] = []
        self._highres_option_widgets: list[tk.Widget] = []
        self._camera_attention_option_widgets: list[tk.Widget] = []
        self._camera_attention_detector_widgets: list[tk.Widget] = []
        self._yellow_option_widgets: list[tk.Widget] = []
        self._custom_red_widgets: list[tk.Widget] = []
        self._custom_yellow_widgets: list[tk.Widget] = []
        self._parallel_face_option_widgets: list[tk.Widget] = []
        self._gpu_memory_safe_option_widgets: list[tk.Widget] = []
        self._gpu_secondary_worker_widgets: list[tk.Widget] = []
        self._portrait_boundary_guard_option_widgets: list[tk.Widget] = []
        self._portrait_repeat_option_widgets: list[tk.Widget] = []
        self._portrait_repeat_basic_option_widgets: list[tk.Widget] = []

        def saved(name: str, default):
            return self.state_data.get(name, default)

        self.folder_var = tk.StringVar(value=initial_folder or saved("folder", ""))
        self.mode_var = tk.StringVar(value=saved("mode", config.get("runtime", {}).get("mode", "portrait")))
        self.preview_var = tk.IntVar(value=int(saved("preview", config["preview"]["portrait_long_edge"])))
        self.group_preview_var = tk.IntVar(value=int(saved("group_preview", config["preview"].get("group_long_edge", 3200))))
        self.scheme_var = tk.StringVar(value=saved("scheme", config["xmp"]["scheme"]))
        self.custom_red_var = tk.StringVar(value=saved("custom_red", config["xmp"]["custom_red"]))
        self.custom_yellow_var = tk.StringVar(value=saved("custom_yellow", config["xmp"].get("custom_yellow", "Second")))
        self.clear_red_var = tk.BooleanVar(value=bool(saved("clear_red_before_run", config["xmp"].get("clear_red_before_run", True))))
        self.clear_yellow_var = tk.BooleanVar(value=bool(saved("clear_yellow_before_run", config["xmp"].get("clear_yellow_before_run", True))))
        self.group_find_candidates_var = tk.BooleanVar(value=bool(saved("group_find_candidates", config.get("group", {}).get("find_headswap_candidates", True))))
        self.group_max_extra_var = tk.IntVar(value=int(saved("group_max_extra", config.get("group", {}).get("max_extra_candidates", 3))))
        self.group_min_extra_var = tk.IntVar(value=int(saved("group_min_extra", config.get("group", {}).get("min_extra_candidates", 1))))
        self.group_min_people_var = tk.IntVar(value=int(saved("group_min_people", config.get("group", {}).get("min_people", 4))))
        self.group_highres_rescue_var = tk.BooleanVar(value=bool(saved("group_highres_rescue", config.get("group", {}).get("highres_rescue_enabled", False))))
        self.group_det_size_var = tk.IntVar(value=int(saved("group_det_size", config.get("analysis", {}).get("insightface_det_size_group", 1024))))
        self.group_det_thresh_var = tk.DoubleVar(value=float(saved("group_det_thresh", config.get("analysis", {}).get("insightface_det_thresh_group", 0.22))))
        self.group_min_face_pct_var = tk.DoubleVar(value=float(saved("group_min_face_pct", config.get("group", {}).get("min_track_face_fraction", 0.00035) * 100.0)))
        self.group_highres_det_size_var = tk.IntVar(value=int(saved("group_highres_det_size", config.get("group", {}).get("highres_rescue_det_size", 1536))))
        self.group_highres_det_thresh_var = tk.DoubleVar(value=float(saved("group_highres_det_thresh", config.get("group", {}).get("highres_rescue_det_thresh", 0.14))))
        self.group_highres_min_face_pct_var = tk.DoubleVar(value=float(saved("group_highres_min_face_pct", config.get("group", {}).get("highres_rescue_min_face_fraction", 0.00018) * 100.0)))
        self.group_highres_min_presence_var = tk.IntVar(value=int(saved("group_highres_min_presence", config.get("group", {}).get("highres_rescue_min_presence", 2))))
        self.group_camera_attention_var = tk.BooleanVar(value=bool(saved("group_camera_attention", config.get("group", {}).get("camera_attention_enabled", False))))
        self.group_camera_attention_shortlist_var = tk.IntVar(value=int(saved("group_camera_attention_shortlist", config.get("group", {}).get("camera_attention_shortlist", 3))))
        self.group_camera_attention_preview_var = tk.IntVar(value=int(saved("group_camera_attention_preview", config.get("group", {}).get("camera_attention_preview_long_edge", 4800))))
        self.group_camera_attention_det_size_var = tk.IntVar(value=int(saved("group_camera_attention_det_size", config.get("group", {}).get("camera_attention_det_size", 1280))))
        self.group_camera_attention_min_eye_px_var = tk.DoubleVar(value=float(saved("group_camera_attention_min_eye_px", config.get("group", {}).get("camera_attention_min_eye_px", 14.0))))
        self.group_camera_attention_away_penalty_var = tk.DoubleVar(value=float(saved("group_camera_attention_away_penalty", config.get("group", {}).get("camera_attention_away_penalty", 0.45))))
        self.group_profile_var = tk.StringVar(value=str(saved("group_profile", GROUP_PROFILE_DEFAULT)))
        if self.group_profile_var.get() not in GROUP_RULE_PROFILES:
            self.group_profile_var.set(GROUP_PROFILE_DEFAULT)
        self.group_profile_desc_var = tk.StringVar()
        self.provider_var = tk.StringVar(value=saved("insightface_provider", config["analysis"].get("insightface_provider", "auto")))
        self.cuda_algo_var = tk.StringVar(value=saved("cuda_conv_algo", config["analysis"].get("insightface_cuda_conv_algo", "HEURISTIC")))
        self.cuda_fallback_var = tk.BooleanVar(value=bool(saved("cuda_fallback", config["analysis"].get("insightface_cuda_fallback_cpu", True))))
        saved_workers = saved("cpu_workers", config.get("runtime", {}).get("cpu_workers", 2))
        try:
            saved_workers = int(saved_workers)
        except (TypeError, ValueError):
            saved_workers = 2
        self.cpu_workers_var = tk.IntVar(value=max(1, min(8, saved_workers or 2)))
        self.parallel_face_analysis_var = tk.BooleanVar(value=bool(saved("parallel_face_analysis", config.get("runtime", {}).get("parallel_face_analysis", False))))
        saved_parallel_workers = saved("parallel_face_workers", config.get("runtime", {}).get("parallel_face_workers", 2))
        try:
            saved_parallel_workers = int(saved_parallel_workers)
        except (TypeError, ValueError):
            saved_parallel_workers = 2
        self.parallel_face_workers_var = tk.IntVar(value=max(2, min(4, saved_parallel_workers)))
        self.gpu_memory_safe_mode_var = tk.BooleanVar(value=bool(saved(
            "gpu_memory_safe_mode", config.get("runtime", {}).get("gpu_memory_safe_mode", True)
        )))
        try:
            saved_mem_limit = float(saved(
                "gpu_session_mem_limit_gb", config.get("runtime", {}).get("gpu_session_mem_limit_gb", 6.0)
            ))
        except (TypeError, ValueError):
            saved_mem_limit = 6.0
        self.gpu_session_mem_limit_gb_var = tk.DoubleVar(value=max(3.0, min(16.0, saved_mem_limit)))
        try:
            saved_secondary_workers = int(saved(
                "group_secondary_face_workers", config.get("runtime", {}).get("group_secondary_face_workers", 2)
            ))
        except (TypeError, ValueError):
            saved_secondary_workers = 2
        self.group_secondary_face_workers_var = tk.IntVar(value=max(1, min(4, saved_secondary_workers)))
        try:
            saved_recycle = int(saved(
                "group_gpu_recycle_every", config.get("runtime", {}).get("group_gpu_recycle_every", 8)
            ))
        except (TypeError, ValueError):
            saved_recycle = 8
        self.group_gpu_recycle_every_var = tk.IntVar(value=max(0, min(50, saved_recycle)))

        # Series / identity.
        self.gap_var = tk.DoubleVar(value=float(saved("gap", config["series"]["max_gap_seconds"])))
        self.name_gap_var = tk.IntVar(value=int(saved("name_gap", config["series"]["max_filename_gap"])))
        self.min_frames_var = tk.IntVar(value=int(saved("min_frames", config["series"]["min_frames"])))
        self.series_algorithm_var = tk.StringVar(value=saved("series_algorithm", config["series"].get("portrait_algorithm", "dbscan")))
        self.sequential_similarity_var = tk.DoubleVar(value=float(saved("sequential_similarity", config["series"].get("portrait_same_person_similarity", 0.42))))
        self.sequential_confirm_frames_var = tk.IntVar(value=int(saved("sequential_confirm_frames", config["series"].get("portrait_break_confirm_frames", 2))))
        self.dbscan_distance_var = tk.DoubleVar(value=float(saved("dbscan_distance", config["series"].get("portrait_dbscan_distance", 0.27))))
        self.dbscan_min_samples_var = tk.IntVar(value=int(saved("dbscan_min_samples", config["series"].get("portrait_dbscan_min_samples", 2))))
        self.segment_merge_distance_var = tk.DoubleVar(value=float(saved("segment_merge_distance", config["series"].get("portrait_segment_merge_distance", 0.32))))
        self.min_confirmed_frames_var = tk.IntVar(value=int(saved("min_confirmed_frames", config["series"].get("portrait_min_confirmed_frames", 2))))
        self.confirm_det_thresh_var = tk.DoubleVar(value=float(saved("confirm_det_thresh", config["series"].get("portrait_confirm_det_thresh", 0.45))))
        self.confirm_face_min_pct_var = tk.DoubleVar(value=float(saved("confirm_face_min_pct", config["series"].get("portrait_confirm_min_face_fraction", 0.0005) * 100.0)))
        self.cross_merge_seconds_var = tk.DoubleVar(value=float(saved("cross_merge_seconds", config["series"].get("portrait_cross_block_merge_seconds", 45.0))))
        self.cross_merge_distance_var = tk.DoubleVar(value=float(saved("cross_merge_distance", config["series"].get("portrait_cross_block_merge_distance", 0.30))))
        self.no_face_tolerance_var = tk.IntVar(value=int(saved("no_face_tolerance", config["series"].get("portrait_no_face_tolerance", 3))))
        self.portrait_boundary_guard_var = tk.BooleanVar(value=bool(saved("portrait_boundary_guard", config["series"].get("portrait_boundary_guard_enabled", False))))
        self.portrait_boundary_guard_window_var = tk.IntVar(value=int(saved("portrait_boundary_guard_window", config["series"].get("portrait_boundary_guard_window", 4))))
        self.portrait_boundary_guard_min_evidence_var = tk.IntVar(value=int(saved("portrait_boundary_guard_min_evidence", config["series"].get("portrait_boundary_guard_min_evidence", 2))))
        self.portrait_boundary_guard_distance_var = tk.DoubleVar(value=float(saved("portrait_boundary_guard_distance", config["series"].get("portrait_boundary_guard_min_centroid_distance", 0.16))))
        self.portrait_boundary_guard_margin_var = tk.DoubleVar(value=float(saved("portrait_boundary_guard_margin", config["series"].get("portrait_boundary_guard_min_identity_margin", 0.10))))
        self.det_thresh_var = tk.DoubleVar(value=float(saved("insightface_det_thresh", config["analysis"].get("insightface_det_thresh", 0.25))))
        self.det_size_portrait_var = tk.IntVar(value=int(saved("det_size_portrait", config["analysis"].get("insightface_det_size_portrait", 640))))
        self.face_min_pct_var = tk.DoubleVar(value=float(saved("face_min_pct", config["analysis"].get("face_min_fraction", 0.0005) * 100.0)))

        # Selection / repeated poses.
        p = config["portrait"]
        saved_repeat_mode = str(saved("repeat_pose_mode", p.get("repeat_pose_mode", "off")))
        if saved_repeat_mode in PORTRAIT_REPEAT_MODES:
            repeat_mode_label = saved_repeat_mode
        else:
            legacy_repeat = {
                "red_yellow": "best_red_pose_yellow",
                "first_red_rest_yellow": "best_red_pose_yellow",
            }.get(saved_repeat_mode.lower(), saved_repeat_mode.lower())
            repeat_mode_label = PORTRAIT_REPEAT_CODES.get(legacy_repeat, PORTRAIT_REPEAT_CODES["off"])
        self.repeat_pose_mode_var = tk.StringVar(value=repeat_mode_label)
        self.repeat_pose_max_series_gap_var = tk.IntVar(value=int(saved("repeat_pose_max_series_gap", p.get("repeat_pose_max_series_gap", 3))))
        self.repeat_pose_max_seconds_var = tk.DoubleVar(value=float(saved("repeat_pose_max_seconds", p.get("repeat_pose_max_seconds", 180.0))))
        self.repeat_pose_min_evidence_var = tk.IntVar(value=int(saved("repeat_pose_min_evidence", p.get("repeat_pose_min_evidence", 2))))
        self.repeat_pose_distance_var = tk.DoubleVar(value=float(saved("repeat_pose_distance", p.get("repeat_pose_max_centroid_distance", 0.24))))
        self.repeat_pose_pair_similarity_var = tk.DoubleVar(value=float(saved("repeat_pose_pair_similarity", p.get("repeat_pose_min_pair_similarity", 0.68))))
        self.repeat_pose_vote_fraction_var = tk.DoubleVar(value=float(saved("repeat_pose_vote_fraction", p.get("repeat_pose_min_vote_fraction", 0.65))))
        self.repeat_pose_cohesion_var = tk.DoubleVar(value=float(saved("repeat_pose_cohesion", p.get("repeat_pose_min_cohesion", 0.78))))
        self.repeat_pose_margin_var = tk.DoubleVar(value=float(saved("repeat_pose_margin", p.get("repeat_pose_min_margin", 0.035))))
        self.repeat_pose_max_yellows_var = tk.IntVar(value=int(saved("repeat_pose_max_yellows", p.get("repeat_pose_max_yellows", 3))))
        self.repeat_pose_min_pose_frames_var = tk.IntVar(value=int(saved("repeat_pose_min_pose_frames", p.get("repeat_pose_min_pose_frames", 2))))
        self.repeat_pose_min_head_conf_var = tk.DoubleVar(value=float(saved("repeat_pose_min_head_conf", p.get("repeat_pose_min_head_confidence", 0.30))))
        self.repeat_pose_yaw_delta_var = tk.DoubleVar(value=float(saved("repeat_pose_yaw_delta", p.get("repeat_pose_min_yaw_delta_deg", 20.0))))
        self.repeat_pose_pitch_delta_var = tk.DoubleVar(value=float(saved("repeat_pose_pitch_delta", p.get("repeat_pose_min_pitch_delta_deg", 16.0))))
        self.repeat_pose_center_shift_var = tk.DoubleVar(value=float(saved("repeat_pose_center_shift", p.get("repeat_pose_min_center_shift", 0.11))))
        self.repeat_pose_scale_change_var = tk.DoubleVar(value=float(saved("repeat_pose_scale_change", p.get("repeat_pose_min_scale_change", 0.32))))
        self.eye_threshold_var = tk.DoubleVar(value=float(saved("eye_threshold", config["analysis"]["eye_open_threshold"])))
        self.prefer_open_eyes_var = tk.BooleanVar(value=bool(saved("prefer_open_eyes", p.get("prefer_open_eyes", True))))
        self.eyes_weight_var = tk.DoubleVar(value=float(saved("eyes_weight", p.get("eyes_weight", 1.8))))
        self.closed_eye_penalty_var = tk.DoubleVar(value=float(saved("closed_eye_penalty", p.get("closed_eye_penalty", 0.45))))
        self.eye_sharpness_weight_var = tk.DoubleVar(value=float(saved("eye_sharpness_weight", p.get("eye_sharpness_weight", 1.4))))
        self.face_sharpness_weight_var = tk.DoubleVar(value=float(saved("face_sharpness_weight", p.get("face_sharpness_weight", 0.9))))
        self.expression_weight_var = tk.DoubleVar(value=float(saved("expression_weight", p.get("expression_weight", 0.7))))
        self.smile_weight_var = tk.DoubleVar(value=float(saved("smile_weight", p.get("smile_weight", 0.35))))
        self.technical_weight_var = tk.DoubleVar(value=float(saved("technical_weight", p.get("technical_weight", 0.35))))

        self.selection_profile_var = tk.StringVar(value=SELECTION_CUSTOM)
        self.series_profile_var = tk.StringVar(value=SERIES_CUSTOM)
        self.advanced_var = tk.BooleanVar(value=self._advanced_visible)
        self.selection_profile_desc_var = tk.StringVar()
        self.series_profile_desc_var = tk.StringVar()

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="0.0%")
        self.status_var = tk.StringVar(value="Готово")
        self.folder_status_var = tk.StringVar(value="Выберите папку съёмки")
        self.hardware_var = tk.StringVar(value="Определение оборудования...")
        self._last_gui_progress = 0.0

        self._build()
        self.mode_var.trace_add("write", lambda *_: self._apply_mode_ui())
        self.series_algorithm_var.trace_add("write", lambda *_: self._apply_context_states())
        self.portrait_boundary_guard_var.trace_add("write", lambda *_: self._apply_context_states())
        self.repeat_pose_mode_var.trace_add("write", lambda *_: self._apply_context_states())
        self.group_highres_rescue_var.trace_add("write", lambda *_: self._apply_context_states())
        self.group_camera_attention_var.trace_add("write", lambda *_: self._apply_context_states())
        self.group_find_candidates_var.trace_add("write", lambda *_: self._apply_context_states())
        self.scheme_var.trace_add("write", lambda *_: self._apply_context_states())
        self.parallel_face_analysis_var.trace_add("write", lambda *_: self._apply_context_states())
        self.gpu_memory_safe_mode_var.trace_add("write", lambda *_: self._apply_context_states())
        self._apply_mode_ui()
        self._apply_context_states()
        self.selection_profile_var.set(self._infer_selection_profile())
        self.series_profile_var.set(self._infer_series_profile())
        self._update_profile_descriptions()
        self._install_profile_traces()
        self.folder_var.trace_add("write", lambda *_: self._schedule_folder_validation())
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._poll_events)
        self.after(100, self._schedule_folder_validation)
        threading.Thread(target=self._detect_hardware, daemon=True).start()

    def _build(self):
        # The former permanent journal pane consumed roughly 40% of the
        # horizontal space and clipped mode-specific controls such as YELLOW
        # min/max. Runtime messages already go to the console and the durable
        # log file, so give the full window width to the settings instead.
        root = ttk.Frame(self, padding=(8, 6))
        root.pack(fill="both", expand=True, padx=10, pady=10)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(4, weight=1)

        title = ttk.Label(root, text="Photo Select AI — портреты и группы", font=("TkDefaultFont", 14, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(root, text="Папка:").grid(row=1, column=0, sticky="w", pady=(8, 2))
        self.folder_entry = ttk.Entry(root, textvariable=self.folder_var)
        self.folder_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(8, 2))
        self.browse_btn = ttk.Button(root, text="Обзор...", command=self._browse)
        self.browse_btn.grid(row=1, column=2, pady=(8, 2))
        ToolTip(self.folder_entry, "Папка одной съёмки. Подкаталоги анализируются автоматически.")
        ttk.Label(root, textvariable=self.folder_status_var).grid(row=2, column=1, columnspan=2, sticky="w", padx=6, pady=(0, 4))

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(4, 6))

        self.basic_tab = ttk.Frame(self.notebook, padding=9)
        self.choice_tab = ttk.Frame(self.notebook, padding=9)
        self.advanced_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(self.basic_tab, text="Быстрый старт")
        self.notebook.add(self.choice_tab, text="Портреты — критерии")
        self.notebook.add(self.advanced_tab, text="Расширенные")
        self._build_basic_tab()
        self._build_choice_tab()
        self._build_advanced_tab()
        if not self._advanced_visible:
            self.notebook.hide(self.advanced_tab)

        ttk.Label(root, textvariable=self.hardware_var).grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 5))

        buttons = ttk.Frame(root)
        buttons.grid(row=6, column=0, columnspan=3, sticky="ew")
        self.start_btn = ttk.Button(buttons, text="НАЧАТЬ АНАЛИЗ", command=self._start, state="disabled")
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(buttons, text="Отмена", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)

        progress_frame = ttk.Frame(root)
        progress_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(7, 2))
        progress_frame.columnconfigure(0, weight=1)
        ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100).grid(row=0, column=0, sticky="ew")
        ttk.Label(progress_frame, textvariable=self.progress_text_var, width=7, anchor="e").grid(row=0, column=1, padx=(6, 0))
        ttk.Label(root, textvariable=self.status_var).grid(row=8, column=0, columnspan=3, sticky="w")

    def _build_basic_tab(self):
        tab = self.basic_tab
        tab.columnconfigure(1, weight=1)
        row = 0

        intro = ttk.LabelFrame(tab, text="Быстрый старт", padding=6)
        intro.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        ttk.Label(
            intro,
            text="Выберите режим и папку; остальные параметры обычно можно оставить по умолчанию.",
            wraplength=800,
        ).pack(anchor="w")
        row += 1

        row = self._combo_row(
            tab,
            row,
            "Режим:",
            self.mode_var,
            ("portrait", "group"),
            "portrait — выбрать лучший портрет. group — выбрать 1 главный групповой кадр и при необходимости дополнительные YELLOW для ручной коррекции.",
        )

        repeat_mode_row = row
        row = self._combo_row(
            tab,
            row,
            "Портреты — режим выбора:",
            self.repeat_pose_mode_var,
            tuple(PORTRAIT_REPEAT_MODES),
            "Обычный режим ставит RED на каждую найденную серию. Режим разных поз связывает близкие серии одного ребёнка по ArcFace: один самый лучший кадр среди всех его серий получает RED, а YELLOW ставится только на лучшие кадры явно и сильно отличающихся поз. Само разделение DBSCAN/sequential не меняется.",
        )
        repeat_clear_row = row
        row = self._check_row(
            tab,
            row,
            "Портреты — снимать старые YELLOW при финальной записи:",
            self.clear_yellow_var,
            "Используется только в режиме «Один ребёнок / разные позы». Старая YELLOW снимается только после полного анализа, одновременно с записью нового плана меток. При отмене анализа XMP не изменяется; остальные XMP/Camera Raw данные сохраняются.",
        )
        self._portrait_repeat_basic_option_widgets.extend(self._grid_row_widgets(tab, repeat_clear_row))

        groups = ttk.LabelFrame(tab, text="Только режим «Группы»", padding=7)
        groups.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(4, 6))
        self.group_settings_frame = groups
        ttk.Label(groups, text="Профиль:").grid(row=0, column=0, sticky="w", pady=1)
        gp = ttk.Combobox(groups, textvariable=self.group_profile_var, values=tuple(GROUP_RULE_PROFILES), state="readonly", width=24)
        gp.grid(row=0, column=1, columnspan=2, sticky="w", padx=(4, 14), pady=1)
        gp.bind("<<ComboboxSelected>>", lambda _e: self._on_group_profile_selected())
        ToolTip(gp, "Рекомендуется: «Глаза прежде всего». Групповые правила хранятся отдельно от портретных и не меняют уже настроенный портретный отбор.")
        ttk.Label(groups, textvariable=self.group_profile_desc_var, wraplength=520).grid(row=0, column=3, columnspan=4, sticky="w", padx=(4, 0))

        ttk.Label(groups, text="Размер изображения для анализа:").grid(row=1, column=0, sticky="w", pady=1)
        e1 = ttk.Entry(groups, textvariable=self.group_preview_var, width=7)
        e1.grid(row=1, column=1, sticky="w", padx=(4, 12), pady=1)
        ToolTip(e1, "Минимум: 1600\nМаксимум: 6000\nРекомендуется: 2800–4000\nПо умолчанию: 3200\n\nБольше → лучше видны маленькие лица, но анализ медленнее.")
        ttk.Label(groups, text="Мин. лиц в группе:").grid(row=1, column=2, sticky="w", pady=1)
        e2 = ttk.Entry(groups, textvariable=self.group_min_people_var, width=5)
        e2.grid(row=1, column=3, sticky="w", padx=(4, 12), pady=1)
        ToolTip(e2, "Минимум: 2\nМаксимум: 80\nРекомендуется: 4–10\nПо умолчанию: 4\n\nМинимум лиц, чтобы серия считалась групповой. Участниками могут быть и дети, и взрослые.")
        ttk.Label(groups, text="YELLOW мин/макс:").grid(row=1, column=4, sticky="w", pady=1)
        emin = ttk.Entry(groups, textvariable=self.group_min_extra_var, width=3)
        emin.grid(row=1, column=5, sticky="w", padx=(4, 2), pady=1)
        emax = ttk.Entry(groups, textvariable=self.group_max_extra_var, width=3)
        emax.grid(row=1, column=6, sticky="w", padx=(2, 0), pady=1)
        ToolTip(emin, "Минимум: 0\nМаксимум: 5\nРекомендуется: 1\nПо умолчанию: 1\n\nМинимальное число YELLOW для группы. Сначала алгоритм анализа ищет целевые кадры, исправляющие конкретные проблемы на RED; если их мало, добирает сильные резервные дубли до этого количества.")
        ToolTip(emax, "Минимум: 0\nМаксимум: 5\nРекомендуется: 3\nПо умолчанию: 3\n\nМожно запросить до 5 YELLOW на одну групповую серию. Этот максимум используется непосредственно во время анализа: сначала выбираются YELLOW, закрывающие проблемы отдельных людей на RED, затем при необходимости добавляются резервные дубли до заданного минимума.")
        self._yellow_option_widgets.extend([emin, emax])
        self._yellow_option_widgets.extend([w for w in groups.grid_slaves(row=1) if int(w.grid_info().get("column", 0)) >= 4])
        c1 = ttk.Checkbutton(groups, text="Искать YELLOW / резервные дубли", variable=self.group_find_candidates_var)
        c1.grid(row=2, column=0, columnspan=4, sticky="w", pady=(3, 0))
        self.group_find_candidates_check = c1
        ToolTip(c1, "Сначала в процессе анализа ищутся целевые YELLOW для конкретных проблем людей на RED. Затем, если их меньше заданного YELLOW минимума, добавляются сильные резервные дубли. Пользователь может разрешить до 5 YELLOW на одну групповую серию.")
        c2 = ttk.Checkbutton(groups, text="Снимать старые YELLOW при финальной записи", variable=self.clear_yellow_var)
        c2.grid(row=2, column=4, columnspan=3, sticky="w", pady=(3, 0))
        self.clear_yellow_check = c2
        ToolTip(c2, "Старая YELLOW снимается только после полного анализа, на финальном этапе записи меток. При отмене анализа XMP не изменяется; остальные XMP/Camera Raw данные сохраняются.")
        c3 = ttk.Checkbutton(
            groups,
            text="Поиск маленьких лиц: улучшенный дополнительный проход (медленнее)",
            variable=self.group_highres_rescue_var,
        )
        c3.grid(row=3, column=0, columnspan=7, sticky="w", pady=(4, 0))
        self.group_highres_check = c3
        ToolTip(c3, "Только для групп. Это соответствует улучшенному поиску маленьких лиц: после обычного прохода программа повторно анализирует уже найденные серии более чувствительным детектором. Может вернуть редкое пропущенное маленькое лицо, но заметно увеличивает время работы. Алгоритм состава группы и значения по умолчанию не изменены.")
        c4 = ttk.Checkbutton(
            groups,
            text="Учитывать взгляд в камеру — финальная проверка лучших дублей (экспериментально)",
            variable=self.group_camera_attention_var,
        )
        c4.grid(row=4, column=0, columnspan=7, sticky="w", pady=(4, 0))
        self.group_camera_attention_check = c4
        ToolTip(c4, "Только для групп. Сначала обычный алгоритм выбирает несколько сильнейших дублей, затем только они перечитываются крупнее. Оценивается поворот головы и положение тёмного центра радужки/зрачка. Сомнительные маленькие глаза считаются unknown и не штрафуются. По умолчанию выключено.")
        row += 1

        series_method_row = row
        row = self._combo_row(
            tab,
            row,
            "Портреты — метод разделения серий:",
            self.series_algorithm_var,
            ("dbscan", "sequential"),
            "dbscan — сопоставляет лица во всём временном блоке и затем разделяет его на последовательные серии. Обычно это наиболее устойчивый вариант. sequential — сравнивает соседние кадры по порядку съёмки и подтверждает смену человека несколькими кадрами. Профиль параметров серии настраивается отдельно.",
        )

        series_profile_row = row
        row = self._combo_row(
            tab,
            row,
            "Портреты — профиль параметров серий:",
            self.series_profile_var,
            tuple(SERIES_PROFILES) + (SERIES_CUSTOM,),
            "Рекомендуется: «Сбалансированный». «Меньше повторов» чаще склеивает соседние фрагменты одного ребёнка. «Строже к случайным кадрам» сильнее отбрасывает стены/пол/отражения. «Чувствительнее к сложным лицам» полезен для дальних, профильных и частично закрытых лиц.",
            command=self._on_series_profile_selected,
        )
        ttk.Label(tab, textvariable=self.series_profile_desc_var, wraplength=720).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
        series_desc_row = row
        row += 1

        selection_profile_row = row
        row = self._combo_row(
            tab,
            row,
            "Портреты — профиль выбора кадра:",
            self.selection_profile_var,
            tuple(SELECTION_PROFILES) + (SELECTION_CUSTOM,),
            "Рекомендуется: «Сбалансированный». Профиль меняет только веса выбора лучшего кадра внутри уже найденной серии и не влияет на распознавание ребёнка.",
            command=self._on_selection_profile_selected,
        )
        ttk.Label(tab, textvariable=self.selection_profile_desc_var, wraplength=720).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        selection_desc_row = row
        row += 1
        boundary_guard_row = row
        row = self._check_row(
            tab, row,
            "Портреты — проверять возможные пропущенные границы вторым методом:",
            self.portrait_boundary_guard_var,
            "Экспериментальная быстрая страховка от редкого слияния двух соседних детей. После выбранного метода второй метод только предлагает подозрительную границу, а программа проверяет её по уже рассчитанным ArcFace embeddings. Повторного анализа фотографий нет."
        )
        self._portrait_only_basic_rows = [repeat_mode_row, repeat_clear_row, series_method_row, series_profile_row, series_desc_row, selection_profile_row, selection_desc_row, boundary_guard_row]
        self._portrait_only_widgets = [
            widget for r in self._portrait_only_basic_rows for widget in tab.grid_slaves(row=r)
        ]

        ttk.Separator(tab).grid(row=row, column=0, columnspan=3, sticky="ew", pady=6); row += 1
        row = self._combo_row(
            tab,
            row,
            "Схема цветовых меток:",
            self.scheme_var,
            ("bridge", "lightroom", "custom"),
            "Рекомендуется: bridge для Adobe Bridge (RED = Select), lightroom для стандартной схемы Lightroom (RED = Red). custom — если вы переименовали цветовые метки.",
        )
        custom_red_row = row
        row = self._entry_row(
            tab,
            row,
            "Custom RED label:",
            self.custom_red_var,
            "Используется только при схеме custom. Введите точное название красной метки из Adobe Bridge Preferences → Labels.",
        )
        self._custom_red_widgets.extend(self._grid_row_widgets(tab, custom_red_row))
        custom_yellow_row = row
        row = self._entry_row(
            tab,
            row,
            "Custom YELLOW label:",
            self.custom_yellow_var,
            "Используется для групп и для портретного режима разных поз. При схеме custom введите точное название жёлтой метки.",
        )
        self._custom_yellow_widgets.extend(self._grid_row_widgets(tab, custom_yellow_row))
        row = self._check_row(
            tab,
            row,
            "Снять старые RED при финальной записи:",
            self.clear_red_var,
            "Рекомендуется для повторных прогонов. Старая RED снимается только после полного анализа, на финальном этапе записи меток. При отмене анализа XMP не изменяется. Остальные XMP/Camera Raw данные сохраняются.",
        )

        ttk.Separator(tab).grid(row=row, column=0, columnspan=3, sticky="ew", pady=8); row += 1
        advanced_check = ttk.Checkbutton(
            tab,
            text="Показать расширенные настройки (отдельно: портреты, группы, GPU)",
            variable=self.advanced_var,
            command=self._toggle_advanced,
        )
        advanced_check.grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        ToolTip(advanced_check, "Для большинства съёмок расширенные настройки менять не требуется. Параметры разделены на портретные, групповые и общие/GPU.")
        row += 1
        ttk.Button(tab, text="Вернуть все рекомендуемые значения", command=self._reset_defaults).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_choice_tab(self):
        tab = self.choice_tab
        tab.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(
            tab,
            text="Только режим «Портреты». Эти параметры определяют, какой кадр победит внутри уже найденной портретной серии. Вес 0 полностью отключает соответствующий критерий.",
            wraplength=790,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10)); row += 1

        row = self._section(tab, row, "Глаза")
        row = self._spin_row(tab, row, "Порог открытых глаз:", self.eye_threshold_var, 0.20, 0.85, 0.01,
            "Рекомендуется: 0.45–0.60; по умолчанию 0.52. Больше → строже к прищуру и полузакрытым глазам. Меньше → мягче.")
        row = self._check_row(tab, row, "Открытые глаза — обязательный фильтр:", self.prefer_open_eyes_var,
            "Рекомендуется: включено. Если в серии есть кадр с надёжно открытыми обоими глазами, закрытые глаза не смогут выиграть только за счёт резкости или улыбки.")
        row = self._spin_row(tab, row, "Вес открытых глаз:", self.eyes_weight_var, 0.0, 3.0, 0.05,
            "Рекомендуется: 1.5–2.2; по умолчанию 1.8. Больше → сильнее приоритет хорошо открытых глаз. 0 → этот вклад отключён.")
        row = self._spin_row(tab, row, "Штраф закрытых глаз:", self.closed_eye_penalty_var, 0.0, 1.5, 0.05,
            "Рекомендуется: 0.35–0.70; по умолчанию 0.45. Больше → сильнее понижается рейтинг кадра с закрытым глазом.")

        row = self._section(tab, row, "Резкость")
        row = self._spin_row(tab, row, "Вес резкости глаз:", self.eye_sharpness_weight_var, 0.0, 3.0, 0.05,
            "Рекомендуется: 1.1–1.8; по умолчанию 1.4. Больше → важнее попадание фокуса именно в глаза. 0 → критерий отключён.")
        row = self._spin_row(tab, row, "Вес резкости лица:", self.face_sharpness_weight_var, 0.0, 3.0, 0.05,
            "Рекомендуется: 0.7–1.2; по умолчанию 0.9. Отвечает за общую детализацию лица независимо от локальной резкости глаз.")

        row = self._section(tab, row, "Выражение")
        row = self._spin_row(tab, row, "Вес выражения лица:", self.expression_weight_var, 0.0, 3.0, 0.05,
            "Рекомендуется: 0.4–1.0; по умолчанию 0.7. Если геометрическая оценка выражения плохо подходит вашей манере съёмки, уменьшите или поставьте 0.")
        row = self._spin_row(tab, row, "Вес улыбки:", self.smile_weight_var, 0.0, 3.0, 0.05,
            "Рекомендуется: 0.2–0.6; по умолчанию 0.35. 0 → улыбка вообще не влияет. Больше → программа чаще предпочитает выраженную улыбку.")
        row = self._spin_row(tab, row, "Вес технического качества:", self.technical_weight_var, 0.0, 3.0, 0.05,
            "Рекомендуется: 0.2–0.7; по умолчанию 0.35. Слишком высокий вес может предпочесть технически чистый, но менее удачный портрет.")

        ttk.Button(tab, text="Сбалансировать критерии", command=lambda: self._apply_selection_profile("Сбалансированный")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_advanced_tab(self):
        outer = self.advanced_tab
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        nested = ttk.Notebook(outer)
        nested.grid(row=0, column=0, sticky="nsew")
        self.advanced_notebook = nested

        # Every advanced page gets its own vertical scroll area.  The notebook
        # page itself remains the stable object used by _apply_context_states(),
        # while all settings are built inside a scrollable content frame.
        portrait_series_tab, portrait_series_content = self._add_scrollable_advanced_page(nested, "Портреты — серии")
        portrait_detector_tab, portrait_detector_content = self._add_scrollable_advanced_page(nested, "Портреты — детектор")
        group_tab, group_content = self._add_scrollable_advanced_page(nested, "Группы — лица")
        runtime_tab, runtime_content = self._add_scrollable_advanced_page(nested, "Общие / GPU")

        self.portrait_series_tab = portrait_series_tab
        self.portrait_detector_tab = portrait_detector_tab
        self.group_advanced_tab = group_tab
        self.runtime_advanced_tab = runtime_tab
        self._build_series_advanced(portrait_series_content)
        self._build_portrait_detector_advanced(portrait_detector_content)
        self._build_group_advanced(group_content)
        self._build_runtime_advanced(runtime_content)

    def _add_scrollable_advanced_page(self, notebook: ttk.Notebook, title: str) -> tuple[ttk.Frame, ttk.Frame]:
        host = ttk.Frame(notebook)
        host.rowconfigure(0, weight=1)
        host.columnconfigure(0, weight=1)

        canvas = tk.Canvas(host, highlightthickness=0, borderwidth=0, takefocus=False)
        scrollbar = ttk.Scrollbar(host, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set, yscrollincrement=18)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        content = ttk.Frame(canvas, padding=12)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def update_scrollregion(_event=None):
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)

        def fit_content_width(event):
            canvas.itemconfigure(window_id, width=max(1, int(event.width)))
            update_scrollregion()

        content.bind("<Configure>", update_scrollregion, add="+")
        canvas.bind("<Configure>", fit_content_width, add="+")
        notebook.add(host, text=title)

        # Bind after the page has been fully populated (via after_idle), so the
        # mouse wheel works over labels, entries, spinboxes and checkbuttons as
        # well as over the empty canvas background.  Returning "break" also
        # prevents a Spinbox under the pointer from changing its value while
        # the user merely wants to scroll the settings page.
        self.after_idle(lambda c=canvas, f=content: self._bind_scroll_wheel_tree(f, c))
        canvas.bind("<MouseWheel>", lambda e, c=canvas: self._scroll_canvas_wheel(e, c), add="+")
        canvas.bind("<Button-4>", lambda e, c=canvas: self._scroll_canvas_wheel(e, c), add="+")
        canvas.bind("<Button-5>", lambda e, c=canvas: self._scroll_canvas_wheel(e, c), add="+")
        return host, content

    def _bind_scroll_wheel_tree(self, widget: tk.Misc, canvas: tk.Canvas):
        try:
            widget.bind("<MouseWheel>", lambda e, c=canvas: self._scroll_canvas_wheel(e, c), add="+")
            widget.bind("<Button-4>", lambda e, c=canvas: self._scroll_canvas_wheel(e, c), add="+")
            widget.bind("<Button-5>", lambda e, c=canvas: self._scroll_canvas_wheel(e, c), add="+")
            for child in widget.winfo_children():
                self._bind_scroll_wheel_tree(child, canvas)
        except tk.TclError:
            pass

    @staticmethod
    def _scroll_canvas_wheel(event, canvas: tk.Canvas):
        if not canvas.winfo_exists() or not canvas.winfo_ismapped():
            return None
        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            delta = int(getattr(event, "delta", 0) or 0)
            if delta == 0:
                return None
            # Windows normally reports +/-120 per wheel notch; high-resolution
            # touchpads may report smaller values, so always move at least one
            # unit in the requested direction.
            steps = max(1, abs(delta) // 120)
            units = -steps if delta > 0 else steps
        canvas.yview_scroll(units, "units")
        return "break"

    def _build_series_advanced(self, tab):
        tab.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(tab, text="Только режим «Портреты». Серые параметры сейчас не участвуют в выбранном методе. Это позволяет видеть все настройки, но сразу понимать, какие реально работают.", wraplength=790).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10)); row += 1

        row = self._section(tab, row, "Метод разделения")
        row = self._combo_row(tab, row, "Портреты — алгоритм разделения:", self.series_algorithm_var, ("dbscan", "sequential"),
            "Оба метода поддерживаются. dbscan — глобальная кластеризация embeddings; sequential — подтверждённая смена ребёнка по соседним кадрам.")

        r = row
        row = self._spin_row(tab, row, "Sequential — порог сходства личности:", self.sequential_similarity_var, 0.20, 0.80, 0.01,
            "Только sequential. По умолчанию 0.42. Выше → легче считать соседние кадры одним ребёнком; слишком высокое значение может склеить похожих детей.")
        self._sequential_only_widgets.extend(self._grid_row_widgets(tab, r))
        r = row
        row = self._spin_row(tab, row, "Sequential — кадров для подтверждения смены:", self.sequential_confirm_frames_var, 1, 6, 1,
            "Только sequential. По умолчанию 2. Больше → устойчивее к одному плохому embedding, но граница серии подтверждается медленнее.")
        self._sequential_only_widgets.extend(self._grid_row_widgets(tab, r))

        row = self._section(tab, row, "Параметры только DBSCAN")
        for label, var, lo, hi, inc, help_text in [
            ("DBSCAN distance:", self.dbscan_distance_var, 0.10, 0.60, 0.01, "Только dbscan. Рекомендуется: 0.25–0.30; по умолчанию 0.27. Меньше → строже разделяет похожих детей, но может дробить одного."),
            ("DBSCAN минимум кадров:", self.dbscan_min_samples_var, 1, 5, 1, "Только dbscan. Рекомендуется: 2. Одиночная embedding не становится устойчивым кластером."),
            ("DBSCAN — объединение соседних фрагментов:", self.segment_merge_distance_var, 0.20, 0.50, 0.01, "Только dbscan. По умолчанию 0.32. Более мягкий второй порог для соседних фрагментов одного ребёнка."),
            ("DBSCAN — мин. уверенных кадров серии:", self.min_confirmed_frames_var, 1, 6, 1, "Только dbscan. Рекомендуется: 2. Отсекает одиночные false-positive."),
            ("DBSCAN — порог подтверждения лица:", self.confirm_det_thresh_var, 0.20, 0.90, 0.05, "Только dbscan. Рекомендуется: 0.40–0.55; по умолчанию 0.45."),
            ("DBSCAN — мин. площадь подтверждающего лица, %:", self.confirm_face_min_pct_var, 0.01, 2.0, 0.01, "Только dbscan. Рекомендуется: 0.05–0.15%; по умолчанию 0.05%."),
        ]:
            r = row
            row = self._spin_row(tab, row, label, var, lo, hi, inc, help_text)
            self._dbscan_only_widgets.extend(self._grid_row_widgets(tab, r))

        row = self._section(tab, row, "Страховка от редкого пропуска ребёнка")
        row = self._check_row(tab, row, "Проверять пропущенные границы вторым методом:", self.portrait_boundary_guard_var,
            "Быстрый этап без повторного InsightFace. Если выбран DBSCAN, sequential предлагает только подозрительные границы; если выбран sequential — предложения делает DBSCAN. Граница принимается лишь после независимой локальной проверки embeddings.")
        for label, var, lo, hi, inc, help_text in [
            ("Страховка — окно кадров с каждой стороны:", self.portrait_boundary_guard_window_var, 2, 8, 1, "По умолчанию 4. Для каждой предложенной границы сравниваются ближайшие пригодные embeddings слева и справа."),
            ("Страховка — мин. embeddings с каждой стороны:", self.portrait_boundary_guard_min_evidence_var, 2, 6, 1, "По умолчанию 2. Не позволяет одному случайному плохому embedding создать нового ребёнка."),
            ("Страховка — мин. дистанция локальных identity:", self.portrait_boundary_guard_distance_var, 0.05, 0.50, 0.01, "По умолчанию 0.16. Меньше → чувствительнее к похожим детям, но выше риск лишнего разделения."),
            ("Страховка — мин. преимущество своей стороны:", self.portrait_boundary_guard_margin_var, 0.02, 0.40, 0.01, "По умолчанию 0.10. Каждый блок должен заметно лучше совпадать со своим локальным центроидом, чем с соседним."),
        ]:
            r = row
            row = self._spin_row(tab, row, label, var, lo, hi, inc, help_text)
            self._portrait_boundary_guard_option_widgets.extend(self._grid_row_widgets(tab, r))

        row = self._section(tab, row, "Повторные позы одного ребёнка")
        row = self._combo_row(
            tab, row, "Портреты — режим выбора:", self.repeat_pose_mode_var, tuple(PORTRAIT_REPEAT_MODES),
            "Тот же переключатель, что на первом экране. После обычного разделения близкие серии одного ребёнка связываются по identity. RED выбирается как лучший кадр вообще среди всех его серий. YELLOW получают только явно отличающиеся позы; небольшие движения не считаются новой позой."
        )
        for label, var, lo, hi, inc, help_text in [
            ("Повторные позы — искать среди ближайших серий:", self.repeat_pose_max_series_gap_var, 1, 10, 1, "По умолчанию 3. Поиск локальный: 3 означает текущая серия может совпасть с тем же ребёнком максимум через две промежуточные серии."),
            ("Повторные позы — макс. пауза, сек:", self.repeat_pose_max_seconds_var, 0.0, 900.0, 10.0, "По умолчанию 180 секунд. 0 отключает ограничение по времени, но это менее безопасно."),
            ("Повторные позы — мин. embeddings серии:", self.repeat_pose_min_evidence_var, 2, 6, 1, "По умолчанию 2. Серия с одним usable embedding не связывается автоматически с другой."),
            ("Повторные позы — макс. дистанция identity:", self.repeat_pose_distance_var, 0.08, 0.45, 0.01, "По умолчанию 0.24. Меньше → безопаснее, но сложный профиль того же ребёнка может не связаться."),
            ("Повторные позы — мин. медианное сходство:", self.repeat_pose_pair_similarity_var, 0.40, 0.95, 0.01, "По умолчанию 0.68. Несколько embeddings двух серий должны устойчиво совпадать, а не только одна пара кадров."),
            ("Повторные позы — доля совпавших пар:", self.repeat_pose_vote_fraction_var, 0.50, 1.00, 0.05, "По умолчанию 0.65. Защита от случайного высокого совпадения одной пары."),
            ("Повторные позы — мин. связность серии:", self.repeat_pose_cohesion_var, 0.50, 1.00, 0.01, "По умолчанию 0.78. Каждая серия должна сама выглядеть как устойчивая identity."),
            ("Повторные позы — запас до второго кандидата:", self.repeat_pose_margin_var, 0.00, 0.20, 0.005, "По умолчанию 0.035. Если два разных ребёнка почти одинаково похожи, совпадение считается неоднозначным и не объединяется."),
            ("Позы — максимум YELLOW на ребёнка:", self.repeat_pose_max_yellows_var, 0, 5, 1, "По умолчанию 3. RED всегда один; каждая YELLOW должна представлять отдельную явно отличающуюся позу."),
            ("Позы — мин. кадров в отдельной позе:", self.repeat_pose_min_pose_frames_var, 1, 6, 1, "По умолчанию 2. Одиночный случайный поворот головы не создаёт YELLOW; отдельная поза должна повториться хотя бы на двух кадрах."),
            ("Позы — мин. уверенность поворота головы:", self.repeat_pose_min_head_conf_var, 0.10, 0.90, 0.05, "По умолчанию 0.30. Слабая оценка head pose не может сама создать новую позу."),
            ("Позы — мин. разница yaw, градусов:", self.repeat_pose_yaw_delta_var, 8.0, 45.0, 1.0, "По умолчанию 20°. Такой поворот головы уже сам по себе считается заметно другой позой."),
            ("Позы — мин. разница pitch, градусов:", self.repeat_pose_pitch_delta_var, 8.0, 40.0, 1.0, "По умолчанию 16°. Небольшой наклон головы отдельной позой не считается."),
            ("Позы — мин. сдвиг лица в кадре:", self.repeat_pose_center_shift_var, 0.03, 0.30, 0.01, "По умолчанию 0.11 ширины/высоты кадра. Сам по себе сдвиг недостаточен: нужен ещё второй сильный признак."),
            ("Позы — мин. изменение масштаба лица:", self.repeat_pose_scale_change_var, 0.10, 1.00, 0.05, "По умолчанию 0.32 (32%). Само кадрирование не создаёт YELLOW без дополнительного отличия."),
        ]:
            r = row
            row = self._spin_row(tab, row, label, var, lo, hi, inc, help_text)
            self._portrait_repeat_option_widgets.extend(self._grid_row_widgets(tab, r))

        row = self._section(tab, row, "Общие для обоих методов")
        row = self._spin_row(tab, row, "Допуск кадров без embedding:", self.no_face_tolerance_var, 0, 8, 1,
            "Используется обоими методами. Рекомендуется: 2–4; по умолчанию 3.")
        row = self._spin_row(tab, row, "Макс. пауза внутри блока, сек:", self.gap_var, 0.3, 60.0, 0.5,
            "Общий грубый временной блок до разделения по личности; по умолчанию 12 с.")
        row = self._spin_row(tab, row, "Макс. пропуск номера файла:", self.name_gap_var, 1, 30, 1,
            "Общий параметр. По умолчанию 5. Увеличьте, если часть кадров удалена.")
        row = self._spin_row(tab, row, "Объединение через паузу, сек:", self.cross_merge_seconds_var, 0.0, 180.0, 5.0,
            "После любого из двух методов соседние серии могут быть снова объединены по identity. 0 отключает этот этап.")
        row = self._spin_row(tab, row, "Дистанция объединения через паузу:", self.cross_merge_distance_var, 0.15, 0.45, 0.01,
            "Общий финальный порог объединения соседних серий; по умолчанию 0.30.")
        row = self._spin_row(tab, row, "Мин. кадров итоговой серии:", self.min_frames_var, 1, 20, 1,
            "Общий финальный фильтр. Рекомендуется 2. Для настоящих одиночных портретов можно поставить 1.")
        ttk.Button(tab, text="Рекомендуемые базовые параметры портретных серий", command=lambda: self._apply_series_profile("Сбалансированный")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_portrait_detector_advanced(self, tab):
        tab.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(tab, text="Только режим «Портреты». Эти параметры управляют разрешением preview и детектором лиц при портретном анализе. На группы они не влияют.", wraplength=790).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10)); row += 1
        row = self._spin_row(tab, row, "Портреты — preview, длинная сторона:", self.preview_var, 1024, 5000, 128,
            "Рекомендуется: 1800–2400 px; по умолчанию 2048.")
        row = self._spin_row(tab, row, "Портреты — порог детектора:", self.det_thresh_var, 0.10, 0.80, 0.05,
            "Рекомендуется: 0.20–0.30; по умолчанию 0.25.")
        row = self._spin_row(tab, row, "Портреты — размер детектора, px:", self.det_size_portrait_var, 320, 1280, 32,
            "Рекомендуется: 640; 800–1024 для ростовых/дальних кадров. Кратно 32.")
        row = self._spin_row(tab, row, "Портреты — мин. площадь лица, %:", self.face_min_pct_var, 0.01, 5.0, 0.01,
            "Рекомендуется: 0.03–0.15%; по умолчанию 0.05%.")

    def _build_group_advanced(self, tab):
        tab.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(tab, text="Только режим «Группы». Основной проход всегда активен. Параметры high-res становятся доступными только после включения дополнительного прохода.", wraplength=790).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10)); row += 1
        row = self._section(tab, row, "Основной проход")
        row = self._spin_row(tab, row, "Группы — preview, длинная сторона:", self.group_preview_var, 1600, 6000, 128,
            "По умолчанию 3200. Больше → лучше маленькие лица, но медленнее декодирование.")
        row = self._spin_row(tab, row, "Группы — размер детектора, px:", self.group_det_size_var, 512, 2048, 32,
            "По умолчанию 1024. Это основной быстрый проход, который формирует stable roster.")
        row = self._spin_row(tab, row, "Группы — порог детектора:", self.group_det_thresh_var, 0.08, 0.60, 0.01,
            "По умолчанию 0.22. Меньше → чувствительнее, но больше ложных лиц.")
        row = self._spin_row(tab, row, "Группы — мин. площадь лица, %:", self.group_min_face_pct_var, 0.005, 1.0, 0.005,
            "По умолчанию 0.035%. Отсекает очень маленькие/фоновые детекции при формировании stable roster.")
        row = self._section(tab, row, "Дополнительный high-res проход")
        row = self._check_row(tab, row, "Включить улучшенный поиск маленьких лиц:", self.group_highres_rescue_var,
            "По умолчанию выключено. Повышает полноту распознавания групп, но заметно увеличивает время работы.")
        for label, var, lo, hi, inc, help_text in [
            ("High-res — размер детектора, px:", self.group_highres_det_size_var, 1024, 2048, 32, "По умолчанию 1536. Больше → медленнее и выше расход VRAM."),
            ("High-res — порог детектора:", self.group_highres_det_thresh_var, 0.05, 0.40, 0.01, "По умолчанию 0.14. Дополнительное лицо всё равно должно подтвердиться на нескольких дублях."),
            ("High-res — мин. площадь лица, %:", self.group_highres_min_face_pct_var, 0.003, 0.50, 0.001, "По умолчанию 0.018%. Меньше разрешает искать более маленькие лица."),
            ("High-res — минимум дублей с лицом:", self.group_highres_min_presence_var, 2, 8, 1, "По умолчанию 2. Новое лицо не добавляется в состав по одиночной детекции."),
        ]:
            r = row
            row = self._spin_row(tab, row, label, var, lo, hi, inc, help_text)
            self._highres_option_widgets.extend(self._grid_row_widgets(tab, r))

        row = self._section(tab, row, "Взгляд в камеру — финальная проверка")
        row = self._check_row(tab, row, "Учитывать взгляд в камеру:", self.group_camera_attention_var,
            "Экспериментально. Не меняет разделение групп и основной рейтинг всех кадров. После обычного отбора перечитывает крупнее только несколько лучших дублей и уточняет RED по направлению головы/глаз.")
        for label, var, lo, hi, inc, help_text in [
            ("Взгляд — сколько лучших дублей проверять:", self.group_camera_attention_shortlist_var, 2, 6, 1, "По умолчанию 3. Больше → выше шанс найти удачный взгляд, но почти линейно растёт время финальной проверки."),
            ("Взгляд — preview, длинная сторона:", self.group_camera_attention_preview_var, 2400, 7000, 128, "По умолчанию 4800. Это применяется только к короткому списку лучших кадров, а не ко всей съёмке."),
            ("Взгляд — размер детектора, px:", self.group_camera_attention_det_size_var, 640, 2048, 32, "По умолчанию 1280. Кратно 32. Если включён общий high-res поиск лиц, используется уже загруженный его InsightFace-анализатор."),
            ("Взгляд — мин. ширина глаза, px:", self.group_camera_attention_min_eye_px_var, 8.0, 40.0, 1.0, "По умолчанию 14 px. Меньше → больше детей получают оценку взгляда, но растёт риск ошибки. Если глаз меньше порога, состояние считается unknown."),
            ("Взгляд — штраф за явно отведённый взгляд:", self.group_camera_attention_away_penalty_var, 0.0, 1.2, 0.05, "По умолчанию 0.45. Это мягкий штраф на долю людей, уверенно смотрящих в сторону; моргание по-прежнему важнее."),
        ]:
            r = row
            row = self._spin_row(tab, row, label, var, lo, hi, inc, help_text)
            row_widgets = self._grid_row_widgets(tab, r)
            self._camera_attention_option_widgets.extend(row_widgets)
            if label.startswith("Взгляд — размер детектора"):
                self._camera_attention_detector_widgets.extend(row_widgets)

    def _build_runtime_advanced(self, tab):
        tab.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(tab, text="Общие настройки производительности и InsightFace/CUDA. Они применяются и к портретам, и к группам.", wraplength=790).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10)); row += 1
        row = self._spin_row(tab, row, "Параллельные задачи EXIF и preview:", self.cpu_workers_var, 1, 8, 1,
            "Рекомендуется: 2. Это число используется для selective EXIF-чтения и предзагрузки изображений. Финальная XMP-запись выполняется отдельно и последовательно внутри каждого логического ресурса RAW+JPEG. На SSD можно попробовать 3–4; для HDD слишком большое значение может не ускорить работу.")
        row = self._section(tab, row, "Экспериментальное ускорение GPU")
        row = self._check_row(tab, row, "Параллельный InsightFace-анализ:", self.parallel_face_analysis_var,
            "По умолчанию выключено. Создаёт несколько независимых InsightFace/ONNX Runtime сессий и анализирует разные кадры одновременно. Это увеличивает расход VRAM. Для 32 ГБ VRAM разумно начать с 2 сессий.")
        r = row
        row = self._spin_row(tab, row, "Параллельных GPU-сеансов InsightFace:", self.parallel_face_workers_var, 2, 4, 1,
            "Действует только при включённом параллельном анализе. Начните с 2. 3–4 могут дать прирост только если одна сессия не загружает GPU полностью; расход VRAM растёт примерно вместе с числом сессий.")
        self._parallel_face_option_widgets.extend(self._grid_row_widgets(tab, r))
        ttk.Label(tab, text="Результаты параллельного анализа собираются обратно в исходном порядке кадров. Если дополнительная CUDA-сессия не создастся, программа продолжит с меньшим числом сессий.", wraplength=790).grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 10)); row += 1

        row = self._section(tab, row, "Защита видеопамяти")
        row = self._check_row(
            tab, row, "Безопасное управление VRAM:", self.gpu_memory_safe_mode_var,
            "Рекомендуется: включено. Ограничивает рост CUDA memory arena, задаёт лимит на одну ORT-сессию, а для тяжёлых Group high-res/взгляд использует отдельный лимит числа GPU-сессий."
        )
        r = row
        row = self._spin_row(
            tab, row, "Бюджет CUDA arena на 1 InsightFace-worker, ГБ:", self.gpu_session_mem_limit_gb_var, 3.0, 16.0, 0.5,
            "Рекомендуется: 6 ГБ. InsightFace-worker содержит 3 ORT-модели (детектор, recognition, landmarks), поэтому этот бюджет делится между тремя CUDA arena. Это не предварительное резервирование и не абсолютный лимит всего процесса, но он не даёт отдельной ORT arena разрастись до всей VRAM."
        )
        self._gpu_memory_safe_option_widgets.extend(self._grid_row_widgets(tab, r))
        r = row
        row = self._spin_row(
            tab, row, "Макс. GPU-сессий для Group high-res/взгляда:", self.group_secondary_face_workers_var, 1, 4, 1,
            "Рекомендуется: 2. Основной проход по сотням файлов по-прежнему может использовать выбранные выше 4 сессии. Но финальный взгляд обычно проверяет только 3 кадра группы, поэтому 4 постоянно загруженные модели расходуют VRAM почти без пользы."
        )
        secondary_widgets = self._grid_row_widgets(tab, r)
        self._gpu_memory_safe_option_widgets.extend(secondary_widgets)
        self._gpu_secondary_worker_widgets.extend(secondary_widgets)
        r = row
        row = self._spin_row(
            tab, row, "Перезапуск gaze-сессий каждые N групп:", self.group_gpu_recycle_every_var, 0, 50, 1,
            "Рекомендуется: 8. Периодически уничтожает и создаёт заново только отдельный пул проверки взгляда, чтобы CUDA/ORT-кэш не накапливался всю съёмку. 0 отключает периодический перезапуск."
        )
        self._gpu_memory_safe_option_widgets.extend(self._grid_row_widgets(tab, r))
        ttk.Label(
            tab,
            text="При выбранных 4 параллельных сессиях safe mode не отнимает ускорение у основного прохода: ограничение до 1–2 сессий применяется только к повторным Group high-res/взгляд этапам.",
            wraplength=790,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 10)); row += 1

        row = self._combo_row(tab, row, "Устройство InsightFace:", self.provider_var, ("auto", "cuda", "cpu"),
            "Рекомендуется: auto. Использует NVIDIA CUDA, если CUDAExecutionProvider доступен.")
        row = self._combo_row(tab, row, "Алгоритм cuDNN Conv:", self.cuda_algo_var, ("HEURISTIC", "EXHAUSTIVE", "DEFAULT"),
            "Рекомендуется: HEURISTIC для RTX 50xx.")
        row = self._check_row(tab, row, "Автопереход CUDA → CPU при ошибке:", self.cuda_fallback_var,
            "Рекомендуется: включено как защита от потери кадров.")

    def _section(self, parent, row: int, title: str) -> int:
        ttk.Label(parent, text=title, font=("TkDefaultFont", 10, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(10 if row else 0, 3))
        return row + 1

    def _help_label(self, parent, row: int, text: str, help_text: str):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=3)
        help_label = ttk.Label(parent, text=" ⓘ ", anchor="center", cursor="hand2")
        help_label.grid(row=row, column=2, sticky="w", padx=(6, 0))
        ToolTip(help_label, help_text)

    @staticmethod
    def _fmt_bound(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:g}"

    def _spin_row(self, parent, row, label, var, from_, to, increment, help_text):
        range_help = (
            f"Минимум: {self._fmt_bound(float(from_))}\n"
            f"Максимум: {self._fmt_bound(float(to))}\n\n"
            f"{help_text}"
        )
        self._help_label(parent, row, label, range_help)
        widget = ttk.Spinbox(parent, from_=from_, to=to, increment=increment, textvariable=var, width=14)
        widget.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
        ToolTip(widget, range_help)
        self._numeric_specs.append((var, label.rstrip(":"), float(from_), float(to)))
        return row + 1

    def _combo_row(self, parent, row, label, var, values, help_text, command=None):
        self._help_label(parent, row, label, help_text)
        widget = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=30)
        widget.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
        ToolTip(widget, help_text)
        if command:
            widget.bind("<<ComboboxSelected>>", lambda _e: command())
        return row + 1

    def _entry_row(self, parent, row, label, var, help_text):
        self._help_label(parent, row, label, help_text)
        widget = ttk.Entry(parent, textvariable=var, width=30)
        widget.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
        ToolTip(widget, help_text)
        return row + 1

    def _check_row(self, parent, row, label, var, help_text):
        self._help_label(parent, row, label, help_text)
        widget = ttk.Checkbutton(parent, variable=var)
        widget.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
        ToolTip(widget, help_text)
        return row + 1

    @staticmethod
    def _grid_row_widgets(parent, row: int) -> list[tk.Widget]:
        return list(parent.grid_slaves(row=row))

    @staticmethod
    def _set_widgets_enabled(widgets, enabled: bool) -> None:
        for widget in widgets:
            try:
                if isinstance(widget, ttk.Widget):
                    widget.state(["!disabled"] if enabled else ["disabled"])
                else:
                    widget.configure(state="normal" if enabled else "disabled")
            except (tk.TclError, AttributeError):
                pass

    def _repeat_pose_mode_code(self) -> str:
        value = str(self.repeat_pose_mode_var.get())
        return PORTRAIT_REPEAT_MODES.get(value, value.lower())

    def _apply_context_states(self):
        if not hasattr(self, "notebook"):
            return
        mode = self.mode_var.get()
        portrait = mode == "portrait"
        group = mode == "group"
        algorithm = self.series_algorithm_var.get().strip().lower()

        # Advanced tabs stay visible as documentation, but irrelevant tabs are
        # disabled so it is immediately obvious which settings can affect a run.
        adv = getattr(self, "advanced_notebook", None)
        if adv is not None:
            for tab, enabled in (
                (getattr(self, "portrait_series_tab", None), portrait),
                (getattr(self, "portrait_detector_tab", None), portrait),
                (getattr(self, "group_advanced_tab", None), group),
                (getattr(self, "runtime_advanced_tab", None), True),
            ):
                if tab is not None:
                    try:
                        adv.tab(tab, state="normal" if enabled else "disabled")
                    except tk.TclError:
                        pass
            try:
                selected = adv.select()
                if selected and str(adv.tab(selected, "state")) == "disabled":
                    adv.select(self.group_advanced_tab if group else self.portrait_series_tab)
            except tk.TclError:
                pass

        repeat_enabled = portrait and self._repeat_pose_mode_code() in {"red_yellow", "first_red_rest_yellow", "best_red_pose_yellow"}
        self._set_widgets_enabled(self._dbscan_only_widgets, portrait and algorithm == "dbscan")
        self._set_widgets_enabled(self._sequential_only_widgets, portrait and algorithm == "sequential")
        self._set_widgets_enabled(self._portrait_boundary_guard_option_widgets, portrait and bool(self.portrait_boundary_guard_var.get()))
        self._set_widgets_enabled(self._portrait_repeat_option_widgets, repeat_enabled)
        self._set_widgets_enabled(self._portrait_repeat_basic_option_widgets, repeat_enabled)
        self._set_widgets_enabled(self._highres_option_widgets, group and bool(self.group_highres_rescue_var.get()))
        camera_enabled = group and bool(self.group_camera_attention_var.get())
        self._set_widgets_enabled(self._camera_attention_option_widgets, camera_enabled)
        # With face-rescue enabled the gaze shortlist reuses that already-loaded
        # InsightFace session, so its detector size comes from the high-res block.
        # Make the otherwise ignored gaze detector field visibly inactive.
        self._set_widgets_enabled(
            self._camera_attention_detector_widgets,
            camera_enabled and not bool(self.group_highres_rescue_var.get()),
        )
        self._set_widgets_enabled(self._yellow_option_widgets, group and bool(self.group_find_candidates_var.get()))
        self._set_widgets_enabled(self._custom_red_widgets, self.scheme_var.get().strip().lower() == "custom")
        self._set_widgets_enabled(
            self._custom_yellow_widgets,
            (group or repeat_enabled) and self.scheme_var.get().strip().lower() == "custom",
        )
        parallel_enabled = bool(self.parallel_face_analysis_var.get())
        self._set_widgets_enabled(self._parallel_face_option_widgets, parallel_enabled)
        safe_vram = bool(self.gpu_memory_safe_mode_var.get())
        self._set_widgets_enabled(self._gpu_memory_safe_option_widgets, safe_vram)
        self._set_widgets_enabled(self._gpu_secondary_worker_widgets, safe_vram and parallel_enabled)
        if hasattr(self, "clear_yellow_check"):
            self._set_widgets_enabled([self.clear_yellow_check], group)

    def _apply_mode_ui(self):
        mode = self.mode_var.get()
        if not hasattr(self, "notebook"):
            return
        if mode == "group":
            if hasattr(self, "group_settings_frame"):
                self.group_settings_frame.grid()
            for widget in getattr(self, "_portrait_only_widgets", []):
                widget.grid_remove()
            try:
                if str(self.notebook.tab(self.choice_tab, "state")) != "hidden":
                    if self.notebook.select() == str(self.choice_tab):
                        self.notebook.select(self.basic_tab)
                    self.notebook.hide(self.choice_tab)
            except tk.TclError:
                pass
        else:
            if hasattr(self, "group_settings_frame"):
                self.group_settings_frame.grid_remove()
            for widget in getattr(self, "_portrait_only_widgets", []):
                widget.grid()
            try:
                if str(self.notebook.tab(self.choice_tab, "state")) == "hidden":
                    self.notebook.add(self.choice_tab, text="Портреты — критерии")
            except tk.TclError:
                pass

        self._apply_context_states()

    def _toggle_advanced(self):
        want = bool(self.advanced_var.get())
        hidden = str(self.notebook.tab(self.advanced_tab, "state")) == "hidden"
        if want and hidden:
            self.notebook.add(self.advanced_tab, text="Расширенные")
            self.notebook.select(self.advanced_tab)
        elif not want and not hidden:
            if self.notebook.select() == str(self.advanced_tab):
                self.notebook.select(self.basic_tab)
            self.notebook.hide(self.advanced_tab)
        self._advanced_visible = want
        self._apply_context_states()

    def _install_profile_traces(self):
        selection_vars = (
            self.eye_threshold_var, self.prefer_open_eyes_var, self.eyes_weight_var,
            self.closed_eye_penalty_var, self.eye_sharpness_weight_var,
            self.face_sharpness_weight_var, self.expression_weight_var,
            self.smile_weight_var, self.technical_weight_var,
        )
        for var in selection_vars:
            var.trace_add("write", lambda *_: self._selection_changed())
        series_vars = (
            self.series_algorithm_var, self.sequential_similarity_var, self.sequential_confirm_frames_var,
            self.dbscan_distance_var, self.dbscan_min_samples_var,
            self.segment_merge_distance_var, self.min_confirmed_frames_var,
            self.confirm_det_thresh_var, self.confirm_face_min_pct_var, self.no_face_tolerance_var,
            self.gap_var, self.name_gap_var, self.cross_merge_seconds_var, self.cross_merge_distance_var,
            self.min_frames_var, self.det_thresh_var, self.det_size_portrait_var, self.face_min_pct_var,
        )
        for var in series_vars:
            var.trace_add("write", lambda *_: self._series_changed())

    def _selection_changed(self):
        if self._applying_profile:
            return
        try:
            self.selection_profile_var.set(self._infer_selection_profile())
        except (tk.TclError, ValueError, TypeError):
            self.selection_profile_var.set(SELECTION_CUSTOM)
        self._update_profile_descriptions()

    def _series_changed(self):
        if self._applying_profile:
            return
        try:
            self.series_profile_var.set(self._infer_series_profile())
        except (tk.TclError, ValueError, TypeError):
            self.series_profile_var.set(SERIES_CUSTOM)
        self._update_profile_descriptions()

    def _infer_selection_profile(self) -> str:
        current = {
            "eye_threshold": float(self.eye_threshold_var.get()),
            "prefer_open_eyes": bool(self.prefer_open_eyes_var.get()),
            "eyes_weight": float(self.eyes_weight_var.get()),
            "closed_eye_penalty": float(self.closed_eye_penalty_var.get()),
            "eye_sharpness_weight": float(self.eye_sharpness_weight_var.get()),
            "face_sharpness_weight": float(self.face_sharpness_weight_var.get()),
            "expression_weight": float(self.expression_weight_var.get()),
            "smile_weight": float(self.smile_weight_var.get()),
            "technical_weight": float(self.technical_weight_var.get()),
        }
        for name, profile in SELECTION_PROFILES.items():
            if _mapping_close(current, profile):
                return name
        return SELECTION_CUSTOM

    def _infer_series_profile(self) -> str:
        current = {
            "dbscan_distance": float(self.dbscan_distance_var.get()),
            "dbscan_min_samples": int(self.dbscan_min_samples_var.get()),
            "segment_merge_distance": float(self.segment_merge_distance_var.get()),
            "min_confirmed_frames": int(self.min_confirmed_frames_var.get()),
            "confirm_det_thresh": float(self.confirm_det_thresh_var.get()),
            "confirm_face_min_pct": float(self.confirm_face_min_pct_var.get()),
            "no_face_tolerance": int(self.no_face_tolerance_var.get()),
            "gap": float(self.gap_var.get()),
            "name_gap": int(self.name_gap_var.get()),
            "cross_merge_seconds": float(self.cross_merge_seconds_var.get()),
            "cross_merge_distance": float(self.cross_merge_distance_var.get()),
            "min_frames": int(self.min_frames_var.get()),
            "det_thresh": float(self.det_thresh_var.get()),
            "det_size": int(self.det_size_portrait_var.get()),
            "face_min_pct": float(self.face_min_pct_var.get()),
        }
        for name, profile in SERIES_PROFILES.items():
            comparable = {k: v for k, v in profile.items() if k != "series_algorithm"}
            if _mapping_close(current, comparable):
                return name
        return SERIES_CUSTOM

    def _on_selection_profile_selected(self):
        name = self.selection_profile_var.get()
        if name != SELECTION_CUSTOM:
            self._apply_selection_profile(name)
        self._update_profile_descriptions()

    def _on_series_profile_selected(self):
        name = self.series_profile_var.get()
        if name != SERIES_CUSTOM:
            self._apply_series_profile(name)
        self._update_profile_descriptions()

    def _apply_selection_profile(self, name: str):
        profile = SELECTION_PROFILES[name]
        self._applying_profile = True
        try:
            self.eye_threshold_var.set(profile["eye_threshold"])
            self.prefer_open_eyes_var.set(profile["prefer_open_eyes"])
            self.eyes_weight_var.set(profile["eyes_weight"])
            self.closed_eye_penalty_var.set(profile["closed_eye_penalty"])
            self.eye_sharpness_weight_var.set(profile["eye_sharpness_weight"])
            self.face_sharpness_weight_var.set(profile["face_sharpness_weight"])
            self.expression_weight_var.set(profile["expression_weight"])
            self.smile_weight_var.set(profile["smile_weight"])
            self.technical_weight_var.set(profile["technical_weight"])
            self.selection_profile_var.set(name)
        finally:
            self._applying_profile = False
        self._update_profile_descriptions()

    def _apply_series_profile(self, name: str):
        profile = SERIES_PROFILES[name]
        self._applying_profile = True
        try:
            self.dbscan_distance_var.set(profile["dbscan_distance"])
            self.dbscan_min_samples_var.set(profile["dbscan_min_samples"])
            self.segment_merge_distance_var.set(profile["segment_merge_distance"])
            self.min_confirmed_frames_var.set(profile["min_confirmed_frames"])
            self.confirm_det_thresh_var.set(profile["confirm_det_thresh"])
            self.confirm_face_min_pct_var.set(profile["confirm_face_min_pct"])
            self.no_face_tolerance_var.set(profile["no_face_tolerance"])
            self.gap_var.set(profile["gap"])
            self.name_gap_var.set(profile["name_gap"])
            self.cross_merge_seconds_var.set(profile["cross_merge_seconds"])
            self.cross_merge_distance_var.set(profile["cross_merge_distance"])
            self.min_frames_var.set(profile["min_frames"])
            self.det_thresh_var.set(profile["det_thresh"])
            self.det_size_portrait_var.set(profile["det_size"])
            self.face_min_pct_var.set(profile["face_min_pct"])
            self.series_profile_var.set(name)
        finally:
            self._applying_profile = False
        self._update_profile_descriptions()

    def _update_profile_descriptions(self):
        selection_text = {
            "Сбалансированный": "Универсальный выбор: открытые глаза и резкость важнее, но выражение и улыбка тоже учитываются.",
            "Глаза и фокус": "Более строгий выбор по открытым глазам и точности фокуса; улыбка влияет меньше.",
            "Выражение и улыбка": "Выражение лица и улыбка влияют заметнее, но открытые глаза всё ещё защищены обязательным фильтром.",
            "Без приоритета улыбки": "Улыбка не влияет на рейтинг; подходят спокойные портреты и нейтральные выражения.",
            SELECTION_CUSTOM: "Пользовательские значения. Точные параметры доступны на вкладке «Критерии выбора».",
        }
        series_text = {
            "Сбалансированный": "Рекомендуемые базовые параметры для обычных серий 3–20 кадров. Метод DBSCAN/sequential выбирается отдельно.",
            "Меньше повторов": "Чуть активнее объединяет соседние фрагменты/серии одного ребёнка. DBSCAN-специфичные части профиля при sequential автоматически не участвуют.",
            "Строже к случайным кадрам": "Усиливает DBSCAN-проверку подтверждений. При sequential DBSCAN-специфичные значения видны, но заблокированы и не влияют на результат.",
            "Чувствительнее к сложным лицам": "Снижает пороги детектора и увеличивает detector input; эти общие параметры полезны и DBSCAN, и sequential.",
            SERIES_CUSTOM: "Изменены базовые параметры серии. Метод DBSCAN/sequential является отдельной настройкой и не определяет имя профиля.",
        }
        group_text = {
            "Глаза прежде всего": "Рекомендуется для групп: сначала максимум детей с открытыми глазами, затем остальные критерии; минимум 1 резервный YELLOW.",
            "Сбалансированный": "Глаза всё ещё важнее, чем в портретах, но общий score влияет немного сильнее.",
            "Больше резервных кадров": "Мягче к альтернативным дублям и старается дать минимум 2 YELLOW для ручного выбора/перестановки голов.",
        }
        self.selection_profile_desc_var.set(selection_text.get(self.selection_profile_var.get(), ""))
        self.series_profile_desc_var.set(series_text.get(self.series_profile_var.get(), ""))
        self.group_profile_desc_var.set(group_text.get(self.group_profile_var.get(), ""))

    def _on_group_profile_selected(self):
        name = self.group_profile_var.get()
        profile = GROUP_RULE_PROFILES.get(name)
        if not profile:
            return
        self.group_min_extra_var.set(int(profile.get("min_extra_candidates", 1)))
        self.group_max_extra_var.set(int(profile.get("max_extra_candidates", 3)))
        self.group_find_candidates_var.set(True)
        self._update_profile_descriptions()
        self._apply_context_states()

    def _browse(self):
        folder = filedialog.askdirectory(initialdir=self.folder_var.get() or None)
        if folder:
            self.folder_var.set(folder)

    def _schedule_folder_validation(self):
        if self._folder_check_after:
            try:
                self.after_cancel(self._folder_check_after)
            except tk.TclError:
                pass
        self._folder_check_after = self.after(220, self._start_folder_validation)

    def _start_folder_validation(self):
        self._folder_check_after = None
        self._folder_check_token += 1
        token = self._folder_check_token
        raw = self.folder_var.get().strip().strip('"')
        self._folder_is_valid = False
        self._folder_photo_count = 0
        self._apply_start_state()
        if not raw:
            self.folder_status_var.set("Выберите папку съёмки")
            return
        folder = Path(raw)
        if not folder.is_dir():
            self.folder_status_var.set("Папка не существует")
            return
        self.folder_status_var.set("Поиск фотографий...")

        def work():
            count = count_supported_photos(folder, self.base_config["scan"]["extensions"], bool(self.base_config["scan"].get("recursive", True)))
            self.events.put(("folder_validation", (token, count)))

        threading.Thread(target=work, daemon=True).start()

    def _validate_settings(self) -> bool:
        inactive_vars = set()
        if not self.gpu_memory_safe_mode_var.get():
            inactive_vars.update(id(v) for v in (
                self.gpu_session_mem_limit_gb_var, self.group_secondary_face_workers_var,
                self.group_gpu_recycle_every_var,
            ))
        elif not self.parallel_face_analysis_var.get():
            inactive_vars.add(id(self.group_secondary_face_workers_var))
        if self.mode_var.get() == "group":
            inactive_vars.update(id(v) for v in (
                self.preview_var, self.det_thresh_var, self.det_size_portrait_var, self.face_min_pct_var,
                self.gap_var, self.name_gap_var, self.min_frames_var, self.sequential_similarity_var,
                self.sequential_confirm_frames_var, self.dbscan_distance_var, self.dbscan_min_samples_var,
                self.segment_merge_distance_var, self.min_confirmed_frames_var, self.confirm_det_thresh_var,
                self.confirm_face_min_pct_var, self.cross_merge_seconds_var, self.cross_merge_distance_var,
                self.no_face_tolerance_var, self.portrait_boundary_guard_window_var, self.portrait_boundary_guard_min_evidence_var,
                self.portrait_boundary_guard_distance_var, self.portrait_boundary_guard_margin_var,
                self.repeat_pose_max_series_gap_var, self.repeat_pose_max_seconds_var, self.repeat_pose_min_evidence_var,
                self.repeat_pose_distance_var, self.repeat_pose_pair_similarity_var, self.repeat_pose_vote_fraction_var,
                self.repeat_pose_cohesion_var, self.repeat_pose_margin_var, self.repeat_pose_max_yellows_var, self.repeat_pose_min_pose_frames_var,
                self.repeat_pose_min_head_conf_var, self.repeat_pose_yaw_delta_var, self.repeat_pose_pitch_delta_var,
                self.repeat_pose_center_shift_var, self.repeat_pose_scale_change_var, self.eye_threshold_var, self.eyes_weight_var,
                self.closed_eye_penalty_var, self.eye_sharpness_weight_var, self.face_sharpness_weight_var,
                self.expression_weight_var, self.smile_weight_var, self.technical_weight_var,
            ))
            if not self.group_highres_rescue_var.get():
                inactive_vars.update(id(v) for v in (self.group_highres_det_size_var, self.group_highres_det_thresh_var, self.group_highres_min_face_pct_var, self.group_highres_min_presence_var))
            if not self.group_camera_attention_var.get():
                inactive_vars.update(id(v) for v in (self.group_camera_attention_shortlist_var, self.group_camera_attention_preview_var, self.group_camera_attention_det_size_var, self.group_camera_attention_min_eye_px_var, self.group_camera_attention_away_penalty_var))
            elif self.group_highres_rescue_var.get():
                inactive_vars.add(id(self.group_camera_attention_det_size_var))
        else:
            inactive_vars.update(id(v) for v in (self.group_preview_var, self.group_det_size_var, self.group_det_thresh_var, self.group_min_face_pct_var, self.group_highres_det_size_var, self.group_highres_det_thresh_var, self.group_highres_min_face_pct_var, self.group_highres_min_presence_var, self.group_camera_attention_shortlist_var, self.group_camera_attention_preview_var, self.group_camera_attention_det_size_var, self.group_camera_attention_min_eye_px_var, self.group_camera_attention_away_penalty_var))
            if self.series_algorithm_var.get().strip().lower() == "dbscan":
                inactive_vars.update(id(v) for v in (self.sequential_similarity_var, self.sequential_confirm_frames_var))
            else:
                inactive_vars.update(id(v) for v in (self.dbscan_distance_var, self.dbscan_min_samples_var, self.segment_merge_distance_var, self.min_confirmed_frames_var, self.confirm_det_thresh_var, self.confirm_face_min_pct_var))
            if not self.portrait_boundary_guard_var.get():
                inactive_vars.update(id(v) for v in (self.portrait_boundary_guard_window_var, self.portrait_boundary_guard_min_evidence_var, self.portrait_boundary_guard_distance_var, self.portrait_boundary_guard_margin_var))
            if self._repeat_pose_mode_code() not in {"red_yellow", "first_red_rest_yellow", "best_red_pose_yellow"}:
                inactive_vars.update(id(v) for v in (
                    self.repeat_pose_max_series_gap_var, self.repeat_pose_max_seconds_var, self.repeat_pose_min_evidence_var,
                    self.repeat_pose_distance_var, self.repeat_pose_pair_similarity_var, self.repeat_pose_vote_fraction_var,
                    self.repeat_pose_cohesion_var, self.repeat_pose_margin_var, self.repeat_pose_max_yellows_var, self.repeat_pose_min_pose_frames_var,
                    self.repeat_pose_min_head_conf_var, self.repeat_pose_yaw_delta_var, self.repeat_pose_pitch_delta_var,
                    self.repeat_pose_center_shift_var, self.repeat_pose_scale_change_var,
                ))

        for var, label, minimum, maximum in self._numeric_specs:
            if id(var) in inactive_vars:
                continue
            try:
                value = float(var.get())
            except (tk.TclError, ValueError, TypeError):
                messagebox.showerror("Некорректная настройка", f"Параметр «{label}» должен быть числом.")
                return False
            if not minimum <= value <= maximum:
                messagebox.showerror(
                    "Некорректная настройка",
                    f"Параметр «{label}» вне допустимого диапазона.\n\nМинимум: {self._fmt_bound(minimum)}\nМаксимум: {self._fmt_bound(maximum)}\nТекущее значение: {value:g}",
                )
                return False
        if self.mode_var.get() == "portrait" and int(self.det_size_portrait_var.get()) % 32 != 0:
            messagebox.showerror("Некорректная настройка", "Портреты: размер детектора должен быть кратен 32.")
            return False
        if self.mode_var.get() == "group":
            if int(self.group_det_size_var.get()) % 32 != 0:
                messagebox.showerror("Некорректная настройка", "Группы: размер основного детектора должен быть кратен 32.")
                return False
            if self.group_highres_rescue_var.get() and int(self.group_highres_det_size_var.get()) % 32 != 0:
                messagebox.showerror("Некорректная настройка", "Группы: high-res размер детектора должен быть кратен 32.")
                return False
            if self.group_camera_attention_var.get() and not self.group_highres_rescue_var.get() and int(self.group_camera_attention_det_size_var.get()) % 32 != 0:
                messagebox.showerror("Некорректная настройка", "Группы: размер детектора проверки взгляда должен быть кратен 32.")
                return False
        try:
            if self.mode_var.get() == "group" and not 1600 <= int(self.group_preview_var.get()) <= 6000:
                raise ValueError("preview")
            if self.mode_var.get() == "group" and not 2 <= int(self.group_min_people_var.get()) <= 80:
                raise ValueError("min_people")
            if self.mode_var.get() == "group" and self.group_find_candidates_var.get() and not 0 <= int(self.group_max_extra_var.get()) <= 5:
                raise ValueError("max_extra")
            if self.mode_var.get() == "group" and self.group_find_candidates_var.get() and not 0 <= int(self.group_min_extra_var.get()) <= 5:
                raise ValueError("min_extra")
            if self.mode_var.get() == "group" and self.group_find_candidates_var.get() and int(self.group_min_extra_var.get()) > int(self.group_max_extra_var.get()):
                raise ValueError("extra_order")
        except Exception as exc:
            labels = {"preview": "Group preview long edge", "min_people": "Мин. детей в группе", "max_extra": "Макс. дополнительных кандидатов"}
            key = str(exc)
            if key == 'preview':
                msg = 'Group preview long edge должен быть в диапазоне 1600–6000.'
            elif key == 'min_people':
                msg = 'Мин. детей в группе должен быть в диапазоне 2–80.'
            elif key == 'max_extra':
                msg = 'YELLOW максимум должен быть в диапазоне 0–5.'
            elif key == 'min_extra':
                msg = 'YELLOW минимум должен быть в диапазоне 0–5.'
            else:
                msg = 'YELLOW минимум не может быть больше YELLOW максимум.'
            messagebox.showerror("Некорректная настройка", msg)
            return False
        yellow_is_used = self.mode_var.get() == "group" or (
            self.mode_var.get() == "portrait"
            and self._repeat_pose_mode_code() in {"red_yellow", "first_red_rest_yellow", "best_red_pose_yellow"}
        )
        if yellow_is_used:
            scheme = self.scheme_var.get().strip().lower()
            if scheme == "custom":
                red = self.custom_red_var.get().strip() or "Select"
                yellow = self.custom_yellow_var.get().strip() or "Second"
                if red.casefold() == yellow.casefold():
                    messagebox.showerror(
                        "Цветовые метки",
                        "RED и YELLOW имеют одинаковое имя. Adobe хранит цветовую метку как одно значение xmp:Label, поэтому они будут выглядеть тем же цветом.\n\nУкажите разные точные названия красной и жёлтой меток Bridge/Lightroom.",
                    )
                    return False
        return True

    def _current_config(self) -> dict:
        group_rules = dict(GROUP_RULE_PROFILES.get(self.group_profile_var.get(), GROUP_RULE_PROFILES[GROUP_PROFILE_DEFAULT]))
        group_rules.update({
            "min_people": int(self.group_min_people_var.get()),
            "find_headswap_candidates": bool(self.group_find_candidates_var.get()),
            "max_extra_candidates": int(self.group_max_extra_var.get()),
            "min_extra_candidates": int(self.group_min_extra_var.get()),
            "track_det_thresh": float(self.group_det_thresh_var.get()),
            "min_track_face_fraction": float(self.group_min_face_pct_var.get()) / 100.0,
            "highres_rescue_enabled": bool(self.group_highres_rescue_var.get()),
            "highres_rescue_det_size": int(self.group_highres_det_size_var.get()),
            "highres_rescue_det_thresh": float(self.group_highres_det_thresh_var.get()),
            "highres_rescue_min_face_fraction": float(self.group_highres_min_face_pct_var.get()) / 100.0,
            "highres_rescue_min_presence": int(self.group_highres_min_presence_var.get()),
            "camera_attention_enabled": bool(self.group_camera_attention_var.get()),
            "camera_attention_shortlist": int(self.group_camera_attention_shortlist_var.get()),
            "camera_attention_preview_long_edge": int(self.group_camera_attention_preview_var.get()),
            "camera_attention_det_size": int(self.group_camera_attention_det_size_var.get()),
            "camera_attention_min_eye_px": float(self.group_camera_attention_min_eye_px_var.get()),
            "camera_attention_away_penalty": float(self.group_camera_attention_away_penalty_var.get()),
        })
        return merged_config(
            self.base_config,
            {
                "series": {
                    "max_gap_seconds": float(self.gap_var.get()),
                    "max_filename_gap": int(self.name_gap_var.get()),
                    "min_frames": int(self.min_frames_var.get()),
                    "portrait_algorithm": self.series_algorithm_var.get(),
                    "portrait_same_person_similarity": float(self.sequential_similarity_var.get()),
                    "portrait_break_confirm_frames": int(self.sequential_confirm_frames_var.get()),
                    "portrait_dbscan_distance": float(self.dbscan_distance_var.get()),
                    "portrait_dbscan_min_samples": int(self.dbscan_min_samples_var.get()),
                    "portrait_segment_merge_distance": float(self.segment_merge_distance_var.get()),
                    "portrait_min_confirmed_frames": int(self.min_confirmed_frames_var.get()),
                    "portrait_confirm_det_thresh": float(self.confirm_det_thresh_var.get()),
                    "portrait_confirm_min_face_fraction": float(self.confirm_face_min_pct_var.get()) / 100.0,
                    "portrait_cross_block_merge_seconds": float(self.cross_merge_seconds_var.get()),
                    "portrait_cross_block_merge_distance": float(self.cross_merge_distance_var.get()),
                    "portrait_no_face_tolerance": int(self.no_face_tolerance_var.get()),
                    "portrait_boundary_guard_enabled": bool(self.portrait_boundary_guard_var.get()),
                    "portrait_boundary_guard_window": int(self.portrait_boundary_guard_window_var.get()),
                    "portrait_boundary_guard_min_evidence": int(self.portrait_boundary_guard_min_evidence_var.get()),
                    "portrait_boundary_guard_min_centroid_distance": float(self.portrait_boundary_guard_distance_var.get()),
                    "portrait_boundary_guard_min_identity_margin": float(self.portrait_boundary_guard_margin_var.get()),
                },
                "analysis": {
                    "insightface_provider": self.provider_var.get(),
                    "insightface_cuda_conv_algo": self.cuda_algo_var.get(),
                    "insightface_cuda_fallback_cpu": bool(self.cuda_fallback_var.get()),
                    "insightface_det_thresh": float(self.det_thresh_var.get()),
                    "insightface_det_size_portrait": int(self.det_size_portrait_var.get()),
                    "face_min_fraction": float(self.face_min_pct_var.get()) / 100.0,
                    "insightface_det_size_group": int(self.group_det_size_var.get()),
                    "insightface_det_thresh_group": float(self.group_det_thresh_var.get()),
                    "eye_open_threshold": float(self.eye_threshold_var.get()),
                },
                "portrait": {
                    "prefer_open_eyes": bool(self.prefer_open_eyes_var.get()),
                    "eyes_weight": float(self.eyes_weight_var.get()),
                    "closed_eye_penalty": float(self.closed_eye_penalty_var.get()),
                    "eye_sharpness_weight": float(self.eye_sharpness_weight_var.get()),
                    "face_sharpness_weight": float(self.face_sharpness_weight_var.get()),
                    "expression_weight": float(self.expression_weight_var.get()),
                    "smile_weight": float(self.smile_weight_var.get()),
                    "technical_weight": float(self.technical_weight_var.get()),
                    "repeat_pose_mode": self._repeat_pose_mode_code(),
                    "repeat_pose_max_series_gap": int(self.repeat_pose_max_series_gap_var.get()),
                    "repeat_pose_max_seconds": float(self.repeat_pose_max_seconds_var.get()),
                    "repeat_pose_min_evidence": int(self.repeat_pose_min_evidence_var.get()),
                    "repeat_pose_profile_frames": max(5, int(self.repeat_pose_min_evidence_var.get())),
                    "repeat_pose_max_centroid_distance": float(self.repeat_pose_distance_var.get()),
                    "repeat_pose_min_pair_similarity": float(self.repeat_pose_pair_similarity_var.get()),
                    "repeat_pose_min_vote_fraction": float(self.repeat_pose_vote_fraction_var.get()),
                    "repeat_pose_min_cohesion": float(self.repeat_pose_cohesion_var.get()),
                    "repeat_pose_min_margin": float(self.repeat_pose_margin_var.get()),
                    "repeat_pose_max_yellows": int(self.repeat_pose_max_yellows_var.get()),
                    "repeat_pose_min_pose_frames": int(self.repeat_pose_min_pose_frames_var.get()),
                    "repeat_pose_min_head_confidence": float(self.repeat_pose_min_head_conf_var.get()),
                    "repeat_pose_min_yaw_delta_deg": float(self.repeat_pose_yaw_delta_var.get()),
                    "repeat_pose_min_pitch_delta_deg": float(self.repeat_pose_pitch_delta_var.get()),
                    "repeat_pose_min_center_shift": float(self.repeat_pose_center_shift_var.get()),
                    "repeat_pose_min_scale_change": float(self.repeat_pose_scale_change_var.get()),
                },
                "preview": {
                    "portrait_long_edge": int(self.preview_var.get()),
                    "group_long_edge": int(self.group_preview_var.get()),
                },
                "group": group_rules,
                "xmp": {
                    "scheme": self.scheme_var.get(),
                    "custom_red": self.custom_red_var.get().strip() or "Select",
                    "custom_yellow": self.custom_yellow_var.get().strip() or "Second",
                    "clear_red_before_run": bool(self.clear_red_var.get()),
                    "clear_yellow_before_run": bool(self.clear_yellow_var.get()),
                },
                "runtime": {
                    "mode": self.mode_var.get(),
                    "cpu_workers": int(self.cpu_workers_var.get()),
                    "parallel_face_analysis": bool(self.parallel_face_analysis_var.get()),
                    "parallel_face_workers": int(self.parallel_face_workers_var.get()),
                    "gpu_memory_safe_mode": bool(self.gpu_memory_safe_mode_var.get()),
                    "gpu_session_mem_limit_gb": float(self.gpu_session_mem_limit_gb_var.get()),
                    "group_secondary_face_workers": int(self.group_secondary_face_workers_var.get()),
                    "group_gpu_recycle_every": int(self.group_gpu_recycle_every_var.get()),
                },
            },
        )

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        folder = Path(self.folder_var.get().strip().strip('"'))
        if not folder.is_dir():
            messagebox.showerror("Photo Select AI", "Укажите существующую папку съёмки.")
            self._schedule_folder_validation(); return
        if not self._folder_is_valid:
            messagebox.showerror("Photo Select AI", "В выбранной папке не найдены поддерживаемые RAW/JPG/TIF/PSD файлы.")
            self._schedule_folder_validation(); return
        if not self._validate_settings():
            return
        self.progress_var.set(0)
        self.progress_text_var.set("0.0%")
        self._last_gui_progress = 0.0
        self.cancel_event.clear()
        self._set_running_state(True)
        mode_name = "групповой" if self.mode_var.get() == "group" else "портретный"
        if self.mode_var.get() == "group":
            profile_text = (
                f"Профиль групп: {self.group_profile_var.get()}\n"
                f"YELLOW минимум/максимум: {self.group_min_extra_var.get()}/{self.group_max_extra_var.get()}\n"
                f"High-res поиск лиц: {'вкл' if self.group_highres_rescue_var.get() else 'выкл'}\n"
                f"Взгляд в камеру: {'вкл' if self.group_camera_attention_var.get() else 'выкл'}\n"
            )
        else:
            profile_text = (
                f"Метод серий: {self.series_algorithm_var.get()}\n"
                f"Режим выбора: {self.repeat_pose_mode_var.get()}\n"
                f"Проверка пропущенных границ: {'вкл' if self.portrait_boundary_guard_var.get() else 'выкл'}\n"
                f"Профиль параметров серий: {self.series_profile_var.get()}\n"
                f"Профиль выбора: {self.selection_profile_var.get()}\n"
            )
        self._append(
            f"\n=== Новый {mode_name} анализ ===\n"
            f"Папка: {folder}\n"
            f"Фотографий: {self._folder_photo_count}\n"
            f"Режим: {self.mode_var.get()}\n"
            f"{profile_text}"
            f"Предзагрузка: {self.cpu_workers_var.get()} поток(а); "
            f"параллельный InsightFace: {'вкл (' + str(self.parallel_face_workers_var.get()) + ')' if self.parallel_face_analysis_var.get() else 'выкл'}\n"
            f"Защита VRAM: {'вкл — ' + format(self.gpu_session_mem_limit_gb_var.get(), 'g') + ' ГБ/сессию, secondary ≤' + str(self.group_secondary_face_workers_var.get()) if self.gpu_memory_safe_mode_var.get() else 'выкл'}\n"
        )
        cfg = self._current_config()
        self.worker = threading.Thread(target=self._worker, args=(folder, cfg), daemon=True)
        self.worker.start()

    def _worker(self, folder: Path, cfg: dict):
        try:
            pipeline = AnalysisPipeline(
                cfg,
                cancel_event=self.cancel_event,
                progress=lambda p, m: self.events.put(("progress", (p, m))),
                message=lambda m: self.events.put(("message", m)),
            )
            stats, selections = pipeline.run(folder)
            self.events.put(("done", (stats, selections)))
        except CancelledError:
            self.events.put(("cancelled", None))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _cancel(self):
        self.cancel_event.set()
        self.status_var.set("Отмена после текущего файла...")

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    p, m = payload
                    try:
                        value = max(0.0, min(100.0, float(p)))
                    except (TypeError, ValueError):
                        value = self._last_gui_progress
                    # The pipeline already guarantees monotonic progress, but
                    # keep a GUI-side guard as well so queued/stale events can
                    # never make the bar jump backwards.
                    if value + 1e-9 < self._last_gui_progress:
                        continue
                    self._last_gui_progress = max(self._last_gui_progress, value)
                    self.progress_var.set(self._last_gui_progress)
                    self.progress_text_var.set(f"{self._last_gui_progress:.1f}%")
                    self.status_var.set(str(m))
                elif kind == "message":
                    self._append(str(payload) + "\n")
                elif kind == "hardware":
                    self.hardware_var.set(str(payload))
                elif kind == "folder_validation":
                    token, count = payload
                    if token == self._folder_check_token:
                        self._folder_photo_count = int(count)
                        self._folder_is_valid = count > 0
                        if count > 0:
                            suffix = _photo_word(int(count))
                            self.folder_status_var.set(f"Найдено {count:,} {suffix} — можно запускать анализ".replace(",", " "))
                        else:
                            self.folder_status_var.set("Поддерживаемые фотографии не найдены")
                        self._apply_start_state()
                elif kind == "done":
                    stats, selections = payload
                    self.worker = None
                    self._set_running_state(False)
                    self.progress_var.set(100)
                    self.progress_text_var.set("100.0%")
                    self.status_var.set("Анализ завершён")
                    if selections:
                        self._append("\nПоследние выбранные кадры:\n")
                        for sel in selections[-20:]:
                            self._append(f"{sel.label_role.upper():6} {sel.photo.path.name} | {sel.score:.3f} | {sel.reason}\n")
                    summary = _stats_text(stats)
                    self._append(summary)
                    if getattr(stats, "run_mode", "portrait") == "group":
                        self.status_var.set(f"Готово: RED {stats.group_main_selected}, YELLOW {stats.group_extra_selected}")
                    else:
                        if stats.portrait_repeat_yellow_selected:
                            self.status_var.set(
                                f"Готово: RED {stats.portrait_selected}, YELLOW поз {stats.portrait_repeat_yellow_selected}"
                            )
                        else:
                            self.status_var.set(f"Готово: выбрано {stats.portrait_selected} портретов")
                elif kind == "cancelled":
                    self.worker = None
                    self._set_running_state(False)
                    self.status_var.set("Отменено")
                    self._append("Анализ отменён пользователем.\n")
                elif kind == "error":
                    self.worker = None
                    self._set_running_state(False)
                    self.status_var.set("Ошибка")
                    self._append(f"ОШИБКА: {payload}\n")
                    messagebox.showerror("Photo Select AI", str(payload))
        except queue.Empty:
            pass
        self.after(80, self._poll_events)

    def _set_running_state(self, running: bool):
        if running:
            self.start_btn.configure(state="disabled")
            self.cancel_btn.configure(state="normal")
            self.folder_entry.configure(state="disabled")
            self.browse_btn.configure(state="disabled")
        else:
            self.cancel_btn.configure(state="disabled")
            self.folder_entry.configure(state="normal")
            self.browse_btn.configure(state="normal")
            self._apply_start_state()

    def _apply_start_state(self, force_disabled: bool = False):
        running = bool(self.worker and self.worker.is_alive())
        enabled = self._folder_is_valid and not running and not force_disabled
        if hasattr(self, "start_btn"):
            self.start_btn.configure(state="normal" if enabled else "disabled")

    def _append(self, text: str):
        # GUI messages remain available in both the launch console and
        # logs/photo_select_ai.log without reserving screen space for a Text
        # widget. Strip only the trailing newline because logging adds its own.
        value = str(text).rstrip("\r\n")
        if value:
            logging.getLogger("photo_select_ai").info(value)

    def _show_result_dialog(self, stats):
        dialog = tk.Toplevel(self)
        dialog.title("Результат анализа — Photo Select AI")
        dialog.transient(self)
        dialog.geometry("600x540")
        dialog.minsize(520, 440)
        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Анализ завершён", font=("TkDefaultFont", 15, "bold")).pack(anchor="w")
        if getattr(stats, "run_mode", "portrait") == "group":
            summary_line = f"Выбрано RED групп: {stats.group_main_selected}; YELLOW кандидатов: {stats.group_extra_selected}; групповых серий: {stats.group_series}."
        else:
            if stats.portrait_repeat_yellow_selected or stats.portrait_repeat_links:
                summary_line = (
                    f"Выбрано RED: {stats.portrait_selected}; YELLOW разных поз: "
                    f"{stats.portrait_repeat_yellow_selected}; обработано серий: {stats.portrait_series}."
                )
            else:
                summary_line = f"Выбрано {stats.portrait_selected} лучших кадров из {stats.portrait_series} обработанных серий."
        ttk.Label(
            frame,
            text=summary_line,
            font=("TkDefaultFont", 11),
        ).pack(anchor="w", pady=(3, 12))
        text_value = _stats_text(stats).strip()
        text = tk.Text(frame, wrap="word", height=18)
        text.pack(fill="both", expand=True)
        text.insert("1.0", text_value)
        text.configure(state="disabled")
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(12, 0))
        ttk.Button(buttons, text="Скопировать статистику", command=lambda: self._copy_to_clipboard(text_value)).pack(side="left")
        ttk.Button(buttons, text="Закрыть", command=dialog.destroy).pack(side="right")
        dialog.lift(); dialog.focus_force()

    def _copy_to_clipboard(self, text: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_idletasks()
        self.status_var.set("Статистика скопирована в буфер обмена")

    def _reset_defaults(self):
        c = self.base_config; p = c["portrait"]
        self._applying_profile = True
        try:
            self.mode_var.set(str(c.get("runtime", {}).get("mode", "portrait")))
            self.provider_var.set(str(c["analysis"].get("insightface_provider", "auto")))
            self.cuda_algo_var.set(str(c["analysis"].get("insightface_cuda_conv_algo", "HEURISTIC")))
            self.cuda_fallback_var.set(bool(c["analysis"].get("insightface_cuda_fallback_cpu", True)))
            self.cpu_workers_var.set(max(1, min(8, int(c.get("runtime", {}).get("cpu_workers", 2) or 2))))
            self.parallel_face_analysis_var.set(bool(c.get("runtime", {}).get("parallel_face_analysis", False)))
            self.parallel_face_workers_var.set(max(2, min(4, int(c.get("runtime", {}).get("parallel_face_workers", 2) or 2))))
            self.gpu_memory_safe_mode_var.set(bool(c.get("runtime", {}).get("gpu_memory_safe_mode", True)))
            self.gpu_session_mem_limit_gb_var.set(max(3.0, min(16.0, float(c.get("runtime", {}).get("gpu_session_mem_limit_gb", 6.0) or 6.0))))
            self.group_secondary_face_workers_var.set(max(1, min(4, int(c.get("runtime", {}).get("group_secondary_face_workers", 2) or 2))))
            self.group_gpu_recycle_every_var.set(max(0, min(50, int(c.get("runtime", {}).get("group_gpu_recycle_every", 8) or 0))))
            self.preview_var.set(int(c["preview"]["portrait_long_edge"]))
            self.gap_var.set(float(c["series"]["max_gap_seconds"])); self.name_gap_var.set(int(c["series"]["max_filename_gap"])); self.min_frames_var.set(int(c["series"]["min_frames"]))
            self.series_algorithm_var.set(str(c["series"].get("portrait_algorithm", "dbscan"))); self.sequential_similarity_var.set(float(c["series"].get("portrait_same_person_similarity", 0.42))); self.sequential_confirm_frames_var.set(int(c["series"].get("portrait_break_confirm_frames", 2))); self.dbscan_distance_var.set(float(c["series"].get("portrait_dbscan_distance", 0.27))); self.dbscan_min_samples_var.set(int(c["series"].get("portrait_dbscan_min_samples", 2))); self.segment_merge_distance_var.set(float(c["series"].get("portrait_segment_merge_distance", 0.32))); self.min_confirmed_frames_var.set(int(c["series"].get("portrait_min_confirmed_frames", 2))); self.confirm_det_thresh_var.set(float(c["series"].get("portrait_confirm_det_thresh", 0.45))); self.confirm_face_min_pct_var.set(float(c["series"].get("portrait_confirm_min_face_fraction", 0.0005)) * 100.0); self.cross_merge_seconds_var.set(float(c["series"].get("portrait_cross_block_merge_seconds", 45.0))); self.cross_merge_distance_var.set(float(c["series"].get("portrait_cross_block_merge_distance", 0.30))); self.no_face_tolerance_var.set(int(c["series"].get("portrait_no_face_tolerance", 3))); self.portrait_boundary_guard_var.set(bool(c["series"].get("portrait_boundary_guard_enabled", False))); self.portrait_boundary_guard_window_var.set(int(c["series"].get("portrait_boundary_guard_window", 4))); self.portrait_boundary_guard_min_evidence_var.set(int(c["series"].get("portrait_boundary_guard_min_evidence", 2))); self.portrait_boundary_guard_distance_var.set(float(c["series"].get("portrait_boundary_guard_min_centroid_distance", 0.16))); self.portrait_boundary_guard_margin_var.set(float(c["series"].get("portrait_boundary_guard_min_identity_margin", 0.10)))
            self.repeat_pose_mode_var.set(PORTRAIT_REPEAT_CODES.get(str(p.get("repeat_pose_mode", "off")).lower(), PORTRAIT_REPEAT_CODES["off"]))
            self.repeat_pose_max_series_gap_var.set(int(p.get("repeat_pose_max_series_gap", 3))); self.repeat_pose_max_seconds_var.set(float(p.get("repeat_pose_max_seconds", 180.0))); self.repeat_pose_min_evidence_var.set(int(p.get("repeat_pose_min_evidence", 2))); self.repeat_pose_distance_var.set(float(p.get("repeat_pose_max_centroid_distance", 0.24))); self.repeat_pose_pair_similarity_var.set(float(p.get("repeat_pose_min_pair_similarity", 0.68))); self.repeat_pose_vote_fraction_var.set(float(p.get("repeat_pose_min_vote_fraction", 0.65))); self.repeat_pose_cohesion_var.set(float(p.get("repeat_pose_min_cohesion", 0.78))); self.repeat_pose_margin_var.set(float(p.get("repeat_pose_min_margin", 0.035))); self.repeat_pose_max_yellows_var.set(int(p.get("repeat_pose_max_yellows", 3))); self.repeat_pose_min_pose_frames_var.set(int(p.get("repeat_pose_min_pose_frames", 2))); self.repeat_pose_min_head_conf_var.set(float(p.get("repeat_pose_min_head_confidence", 0.30))); self.repeat_pose_yaw_delta_var.set(float(p.get("repeat_pose_min_yaw_delta_deg", 20.0))); self.repeat_pose_pitch_delta_var.set(float(p.get("repeat_pose_min_pitch_delta_deg", 16.0))); self.repeat_pose_center_shift_var.set(float(p.get("repeat_pose_min_center_shift", 0.11))); self.repeat_pose_scale_change_var.set(float(p.get("repeat_pose_min_scale_change", 0.32)))
            self.det_thresh_var.set(float(c["analysis"].get("insightface_det_thresh", 0.25))); self.det_size_portrait_var.set(int(c["analysis"].get("insightface_det_size_portrait", 640))); self.face_min_pct_var.set(float(c["analysis"].get("face_min_fraction", 0.0005)) * 100.0)
            self.group_preview_var.set(int(c["preview"].get("group_long_edge", 3200)))
            self.custom_yellow_var.set(str(c["xmp"].get("custom_yellow", "Second")))
            self.group_min_people_var.set(int(c.get("group", {}).get("min_people", 4)))
            self.group_highres_rescue_var.set(bool(c.get("group", {}).get("highres_rescue_enabled", False)))
            self.group_det_size_var.set(int(c.get("analysis", {}).get("insightface_det_size_group", 1024)))
            self.group_det_thresh_var.set(float(c.get("analysis", {}).get("insightface_det_thresh_group", 0.22)))
            self.group_min_face_pct_var.set(float(c.get("group", {}).get("min_track_face_fraction", 0.00035)) * 100.0)
            self.group_highres_det_size_var.set(int(c.get("group", {}).get("highres_rescue_det_size", 1536)))
            self.group_highres_det_thresh_var.set(float(c.get("group", {}).get("highres_rescue_det_thresh", 0.14)))
            self.group_highres_min_face_pct_var.set(float(c.get("group", {}).get("highres_rescue_min_face_fraction", 0.00018)) * 100.0)
            self.group_highres_min_presence_var.set(int(c.get("group", {}).get("highres_rescue_min_presence", 2)))
            self.group_camera_attention_var.set(bool(c.get("group", {}).get("camera_attention_enabled", False)))
            self.group_camera_attention_shortlist_var.set(int(c.get("group", {}).get("camera_attention_shortlist", 3)))
            self.group_camera_attention_preview_var.set(int(c.get("group", {}).get("camera_attention_preview_long_edge", 4800)))
            self.group_camera_attention_det_size_var.set(int(c.get("group", {}).get("camera_attention_det_size", 1280)))
            self.group_camera_attention_min_eye_px_var.set(float(c.get("group", {}).get("camera_attention_min_eye_px", 14.0)))
            self.group_camera_attention_away_penalty_var.set(float(c.get("group", {}).get("camera_attention_away_penalty", 0.45)))
            self.group_profile_var.set(GROUP_PROFILE_DEFAULT)
            self.group_min_extra_var.set(int(GROUP_RULE_PROFILES[GROUP_PROFILE_DEFAULT].get("min_extra_candidates", 1)))
            self.group_max_extra_var.set(int(GROUP_RULE_PROFILES[GROUP_PROFILE_DEFAULT].get("max_extra_candidates", 3)))
            self.group_find_candidates_var.set(bool(c.get("group", {}).get("find_headswap_candidates", True)))
            self.clear_red_var.set(bool(c["xmp"].get("clear_red_before_run", True)))
            self.clear_yellow_var.set(bool(c["xmp"].get("clear_yellow_before_run", True)))
            self.eye_threshold_var.set(float(c["analysis"]["eye_open_threshold"])); self.prefer_open_eyes_var.set(bool(p.get("prefer_open_eyes", True))); self.eyes_weight_var.set(float(p.get("eyes_weight", 1.8))); self.closed_eye_penalty_var.set(float(p.get("closed_eye_penalty", 0.45))); self.eye_sharpness_weight_var.set(float(p.get("eye_sharpness_weight", 1.4))); self.face_sharpness_weight_var.set(float(p.get("face_sharpness_weight", 0.9))); self.expression_weight_var.set(float(p.get("expression_weight", 0.7))); self.smile_weight_var.set(float(p.get("smile_weight", 0.35))); self.technical_weight_var.set(float(p.get("technical_weight", 0.35)))
            self.selection_profile_var.set("Сбалансированный")
            self.series_profile_var.set("Сбалансированный")
        finally:
            self._applying_profile = False
        self._update_profile_descriptions()
        self.status_var.set("Рекомендуемые настройки восстановлены")

    def _detect_hardware(self):
        hw = detect_hardware()
        providers = ", ".join(hw.onnx_providers) if hw.onnx_providers else "ORT unavailable"
        self.events.put(("hardware", f"{hw.summary} | ONNX: {providers}"))

    def _on_close(self):
        save_ui_state({
            "state_version": 14,
            "folder": self.folder_var.get(), "mode": self.mode_var.get(), "preview": self.preview_var.get(), "group_preview": self.group_preview_var.get(), "scheme": self.scheme_var.get(), "custom_red": self.custom_red_var.get(), "custom_yellow": self.custom_yellow_var.get(), "clear_red_before_run": self.clear_red_var.get(), "clear_yellow_before_run": self.clear_yellow_var.get(), "group_profile": self.group_profile_var.get(), "group_min_people": self.group_min_people_var.get(), "group_min_extra": self.group_min_extra_var.get(), "group_max_extra": self.group_max_extra_var.get(), "group_find_candidates": self.group_find_candidates_var.get(), "group_highres_rescue": self.group_highres_rescue_var.get(), "group_det_size": self.group_det_size_var.get(), "group_det_thresh": self.group_det_thresh_var.get(), "group_min_face_pct": self.group_min_face_pct_var.get(), "group_highres_det_size": self.group_highres_det_size_var.get(), "group_highres_det_thresh": self.group_highres_det_thresh_var.get(), "group_highres_min_face_pct": self.group_highres_min_face_pct_var.get(), "group_highres_min_presence": self.group_highres_min_presence_var.get(), "group_camera_attention": self.group_camera_attention_var.get(), "group_camera_attention_shortlist": self.group_camera_attention_shortlist_var.get(), "group_camera_attention_preview": self.group_camera_attention_preview_var.get(), "group_camera_attention_det_size": self.group_camera_attention_det_size_var.get(), "group_camera_attention_min_eye_px": self.group_camera_attention_min_eye_px_var.get(), "group_camera_attention_away_penalty": self.group_camera_attention_away_penalty_var.get(),
            "advanced_visible": bool(self.advanced_var.get()),
            "insightface_provider": self.provider_var.get(), "cuda_conv_algo": self.cuda_algo_var.get(), "cuda_fallback": self.cuda_fallback_var.get(), "cpu_workers": self.cpu_workers_var.get(), "parallel_face_analysis": self.parallel_face_analysis_var.get(), "parallel_face_workers": self.parallel_face_workers_var.get(), "gpu_memory_safe_mode": self.gpu_memory_safe_mode_var.get(), "gpu_session_mem_limit_gb": self.gpu_session_mem_limit_gb_var.get(), "group_secondary_face_workers": self.group_secondary_face_workers_var.get(), "group_gpu_recycle_every": self.group_gpu_recycle_every_var.get(),
            "gap": self.gap_var.get(), "name_gap": self.name_gap_var.get(), "min_frames": self.min_frames_var.get(), "series_algorithm": self.series_algorithm_var.get(), "sequential_similarity": self.sequential_similarity_var.get(), "sequential_confirm_frames": self.sequential_confirm_frames_var.get(), "dbscan_distance": self.dbscan_distance_var.get(), "dbscan_min_samples": self.dbscan_min_samples_var.get(), "segment_merge_distance": self.segment_merge_distance_var.get(), "min_confirmed_frames": self.min_confirmed_frames_var.get(), "confirm_det_thresh": self.confirm_det_thresh_var.get(), "confirm_face_min_pct": self.confirm_face_min_pct_var.get(), "cross_merge_seconds": self.cross_merge_seconds_var.get(), "cross_merge_distance": self.cross_merge_distance_var.get(), "no_face_tolerance": self.no_face_tolerance_var.get(), "portrait_boundary_guard": self.portrait_boundary_guard_var.get(), "portrait_boundary_guard_window": self.portrait_boundary_guard_window_var.get(), "portrait_boundary_guard_min_evidence": self.portrait_boundary_guard_min_evidence_var.get(), "portrait_boundary_guard_distance": self.portrait_boundary_guard_distance_var.get(), "portrait_boundary_guard_margin": self.portrait_boundary_guard_margin_var.get(), "repeat_pose_mode": self._repeat_pose_mode_code(), "repeat_pose_max_series_gap": self.repeat_pose_max_series_gap_var.get(), "repeat_pose_max_seconds": self.repeat_pose_max_seconds_var.get(), "repeat_pose_min_evidence": self.repeat_pose_min_evidence_var.get(), "repeat_pose_distance": self.repeat_pose_distance_var.get(), "repeat_pose_pair_similarity": self.repeat_pose_pair_similarity_var.get(), "repeat_pose_vote_fraction": self.repeat_pose_vote_fraction_var.get(), "repeat_pose_cohesion": self.repeat_pose_cohesion_var.get(), "repeat_pose_margin": self.repeat_pose_margin_var.get(), "repeat_pose_max_yellows": self.repeat_pose_max_yellows_var.get(), "repeat_pose_min_pose_frames": self.repeat_pose_min_pose_frames_var.get(), "repeat_pose_min_head_conf": self.repeat_pose_min_head_conf_var.get(), "repeat_pose_yaw_delta": self.repeat_pose_yaw_delta_var.get(), "repeat_pose_pitch_delta": self.repeat_pose_pitch_delta_var.get(), "repeat_pose_center_shift": self.repeat_pose_center_shift_var.get(), "repeat_pose_scale_change": self.repeat_pose_scale_change_var.get(), "insightface_det_thresh": self.det_thresh_var.get(), "det_size_portrait": self.det_size_portrait_var.get(), "face_min_pct": self.face_min_pct_var.get(),
            "eye_threshold": self.eye_threshold_var.get(), "prefer_open_eyes": self.prefer_open_eyes_var.get(), "eyes_weight": self.eyes_weight_var.get(), "closed_eye_penalty": self.closed_eye_penalty_var.get(), "eye_sharpness_weight": self.eye_sharpness_weight_var.get(), "face_sharpness_weight": self.face_sharpness_weight_var.get(), "expression_weight": self.expression_weight_var.get(), "smile_weight": self.smile_weight_var.get(), "technical_weight": self.technical_weight_var.get(),
        })
        self.cancel_event.set()
        self.destroy()


def _mapping_close(current: dict[str, object], profile: dict[str, object], tol: float = 1e-6) -> bool:
    for key, expected in profile.items():
        actual = current.get(key)
        if isinstance(expected, bool):
            if bool(actual) != expected:
                return False
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                if abs(float(actual) - float(expected)) > tol:
                    return False
            except (TypeError, ValueError):
                return False
        elif actual != expected:
            return False
    return True


def _photo_word(count: int) -> str:
    n = abs(count) % 100
    n1 = n % 10
    if 10 < n < 20:
        return "фотографий"
    if n1 == 1:
        return "фотография"
    if 1 < n1 < 5:
        return "фотографии"
    return "фотографий"


def _stats_text(stats) -> str:
    scan_text = (
        "--- Сканирование ---\n"
        f"EXIF-чтений: {stats.scan_exif_reads}\n"
        f"Попаданий в EXIF-кэш: {stats.scan_exif_cache_hits}\n"
        f"Время из EXIF: {stats.scan_time_from_exif}\n"
        f"Время из файла (mtime): {stats.scan_time_from_file}\n"
        f"Порядок по имени: {stats.scan_order_from_name}\n"
        f"Ошибок EXIF: {stats.scan_exif_failures}\n"
        f"Файлов пропущено при сканировании: {stats.scan_files_skipped}\n"
        f"RAW+JPEG пар обработано как один кадр: {stats.raw_jpeg_pairs_collapsed}\n\n"
        "--- RAW preview ---\n"
        f"rawpy/LibRaw preview: {stats.raw_preview_rawpy}\n"
        f"JPEG из RAW-контейнера: {stats.raw_preview_embedded_jpeg}\n"
        f"demosaic: {stats.raw_preview_demosaic}\n\n"
    )
    if getattr(stats, "run_mode", "portrait") == "group":
        return (
            "===== ИТОГ АНАЛИЗА =====\n"
            f"Файлов найдено: {stats.files_found}\n"
            f"Файлов проанализировано: {stats.files_analyzed}\n\n"
            + scan_text
            + f"Обработано групповых серий: {stats.group_series}\n"
            f"Выбрано RED-групп: {stats.group_main_selected}\n"
            f"Выбрано YELLOW-кандидатов: {stats.group_extra_selected}\n"
            f"Серий без выбора: {stats.series_without_selection}\n\n"
            "--- Диагностика групп ---\n"
            f"Предварительных временных блоков: {stats.candidate_series}\n"
            f"Групповых серий после фильтрации: {stats.refined_series}\n"
            f"Отброшено слабых/негрупповых блоков: {stats.weak_series_rejected}\n"
            f"Людей учтено в составах групп: {stats.group_tracks_confirmed}\n"
            f"Объединено частей одной физической группы: {stats.group_blocks_merged}\n"
            f"Проблем на RED — глаза: {stats.group_eye_problems}\n"
            f"Проблем на RED — лицо не найдено: {stats.group_missing_problems}\n"
            f"Проблем на RED — резкость глаз: {stats.group_sharpness_problems}\n"
            f"Проблем на RED — прочее качество: {stats.group_quality_problems}\n"
            f"Проверка взгляда — серий с надёжными данными: {stats.group_camera_attention_series}\n"
            f"Проверка взгляда — лиц оценено на RED: {stats.group_camera_attention_known}\n"
            f"Проверка взгляда — уверенно смотрят в сторону на RED: {stats.group_camera_attention_away}\n"
            f"Проблем закрыто целевыми YELLOW: {stats.group_problems_covered}\n"
            f"Резервных YELLOW добавлено: {stats.group_backup_candidates}\n"
            f"Проблем без подходящего дубля: {stats.group_problems_unresolved}\n"
            f"Слишком коротких серий пропущено: {stats.skipped_short_series}\n"
            f"Кадров без обнаруженного лица: {stats.frames_without_faces}\n\n"
            "--- Метки ---\n"
            f"Старых меток снято на финальном этапе: {stats.labels_cleared_before_run}\n"
            f"Меток записано: {stats.xmp_written}\n"
            f"  Встроено в исходные файлы: {stats.embedded_xmp_written}\n"
            f"    из них JPG/JPEG: {stats.jpeg_embedded_written}\n"
            f"  Sidecar XMP: {stats.sidecar_xmp_written}\n"
            f"Ошибок финальной записи меток: {stats.metadata_errors}\n"
            f"Ошибок анализа: {stats.analysis_errors}\n"
            "========================="
        )
    return (
        "===== ИТОГ АНАЛИЗА =====\n"
        f"Файлов найдено: {stats.files_found}\n"
        f"Файлов проанализировано: {stats.files_analyzed}\n\n"
        + scan_text
        + f"Обработано портретных серий: {stats.portrait_series}\n"
        f"Выбрано RED-портретов: {stats.portrait_selected}\n"
        f"Выбрано YELLOW разных поз: {stats.portrait_repeat_yellow_selected}\n"
        f"Найдено детей в режиме поз: {stats.portrait_repeat_children}\n"
        f"Связано повторных серий одного ребёнка: {stats.portrait_repeat_links}\n"
        f"Неоднозначных identity-связей отклонено: {stats.portrait_repeat_ambiguous}\n"
        f"Серий без выбранного кадра: {stats.series_without_selection}\n\n"
        "--- Диагностика серий ---\n"
        f"Предварительных временных блоков: {stats.candidate_series}\n"
        f"Серий после распознавания лиц: {stats.refined_series}\n"
        f"Объединено соседних фрагментов: {stats.local_fragments_merged}\n"
        f"Объединено серий через паузу: {stats.cross_block_series_merged}\n"
        f"Страховка — предложено спорных границ: {stats.portrait_boundary_guard_proposals}\n"
        f"Страховка — восстановлено пропущенных границ: {stats.portrait_boundary_guard_splits}\n"
        f"Отброшено слабых/случайных серий: {stats.weak_series_rejected}\n"
        f"Слишком коротких серий пропущено: {stats.skipped_short_series}\n"
        f"Кадров без обнаруженного лица: {stats.frames_without_faces}\n\n"
        "--- Метки ---\n"
        f"Старых меток снято на финальном этапе: {stats.labels_cleared_before_run}\n"
        f"Меток записано: {stats.xmp_written}\n"
        f"  Встроено в исходные файлы: {stats.embedded_xmp_written}\n"
        f"    из них JPG/JPEG: {stats.jpeg_embedded_written}\n"
        f"  Sidecar XMP: {stats.sidecar_xmp_written}\n"
        f"Ошибок финальной записи меток: {stats.metadata_errors}\n"
        f"Ошибок анализа: {stats.analysis_errors}\n"
        "========================="
    )
