from app.analysis.face_insightface import _provider_order


def test_cuda_heuristic_conv_is_recommended_and_session_is_strict_cuda():
    providers = _provider_order(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "auto",
        "HEURISTIC",
    )
    assert providers[0][0] == "CUDAExecutionProvider"
    assert providers[0][1]["cudnn_conv_algo_search"] == "HEURISTIC"
    assert providers[0][1]["cudnn_conv_use_max_workspace"] == "0"
    assert len(providers) == 1


def test_invalid_cuda_algo_falls_back_to_heuristic():
    providers = _provider_order(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda",
        "something-invalid",
    )
    assert providers[0][1]["cudnn_conv_algo_search"] == "HEURISTIC"


def test_cpu_mode_never_adds_cuda():
    providers = _provider_order(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu",
        "EXHAUSTIVE",
    )
    assert providers == ["CPUExecutionProvider"]


def test_auto_uses_cpu_only_when_cuda_is_unavailable():
    providers = _provider_order(["CPUExecutionProvider"], "auto", "HEURISTIC")
    assert providers == ["CPUExecutionProvider"]


def test_cuda_failure_detector_recognises_cudnn_execution_error():
    from app.analysis.face_insightface import _looks_like_cuda_failure
    exc = RuntimeError("CUDNN_FE failure 11: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED")
    assert _looks_like_cuda_failure(exc)


def test_group_highres_detector_size_can_reach_1536():
    from app.analysis.face_insightface import _round_det_size
    assert _round_det_size(1536) == 1536
    assert _round_det_size(2500) == 2048


def test_eye_dark_center_offset_distinguishes_center_from_side():
    import cv2
    import numpy as np
    from app.analysis.face_insightface import _eye_dark_center_offset

    pts = np.asarray([
        [50, 40], [56, 34], [68, 31], [82, 31], [94, 34],
        [100, 40], [94, 46], [82, 49], [68, 49], [56, 46],
    ], dtype=np.float32)

    def image_with_pupil(x: int):
        img = np.full((80, 150, 3), 210, dtype=np.uint8)
        cv2.fillConvexPoly(img, cv2.convexHull(pts.astype(np.int32)), (185, 185, 185))
        cv2.circle(img, (x, 40), 5, (25, 25, 25), -1)
        return img

    centered = _eye_dark_center_offset(image_with_pupil(75), pts, 14.0)
    side = _eye_dark_center_offset(image_with_pupil(91), pts, 14.0)
    assert centered is not None and side is not None
    assert centered[0] < side[0]
    assert centered[2] > 0.2


def test_head_frontal_metric_penalizes_large_yaw():
    import cv2
    import numpy as np
    from app.analysis.face_insightface import _head_frontal_metrics

    h, w = 1000, 1500
    focal = float(max(h, w))
    camera = np.asarray([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    model = np.asarray([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
    ], dtype=np.float64)
    tvec = np.asarray([[0.0], [0.0], [1500.0]], dtype=np.float64)

    def landmarks_for_yaw(deg: float):
        a = np.deg2rad(deg)
        rotation = np.asarray([
            [np.cos(a), 0.0, np.sin(a)],
            [0.0, 1.0, 0.0],
            [-np.sin(a), 0.0, np.cos(a)],
        ], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(model, rvec, tvec, camera, np.zeros((4, 1), dtype=np.float64))
        lm = np.zeros((106, 2), dtype=np.float32)
        for idx, point in zip([86, 0, 35, 93, 52, 61], projected.reshape(-1, 2)):
            lm[idx] = point
        return lm

    front, front_conf = _head_frontal_metrics(landmarks_for_yaw(0.0), (h, w, 3), {"group": {}})
    side, side_conf = _head_frontal_metrics(landmarks_for_yaw(30.0), (h, w, 3), {"group": {}})
    assert front_conf > 0.8 and side_conf > 0.8
    assert front > side


def test_cuda_safe_memory_options_limit_arena_growth():
    limit = 6 * 1024 ** 3
    providers = _provider_order(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda",
        "HEURISTIC",
        gpu_mem_limit_bytes=limit,
        conservative_arena=True,
    )
    options = providers[0][1]
    assert options["gpu_mem_limit"] == str(limit)
    assert options["arena_extend_strategy"] == "kSameAsRequested"
