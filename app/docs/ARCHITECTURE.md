# Photo Select AI — portrait + group selection

Current build supports independent Portrait and Group workflows.

```text
shoot folder
  -> fast path/sequence filesystem index
  -> exact RAW+JPEG pair coalescing
  -> EXIF cache + selective EXIF (Group) / EXIF-first uncached scan (Portrait)
  -> deterministic source/time ordering
  -> embedded RAW preview or exact paired-JPEG fallback
  -> hard chronological/source blocks
  -> InsightFace SCRFD detection
  -> InsightFace recognition embedding
  -> InsightFace 106-point landmarks
  -> existing portrait/group refinement and scoring
  -> Selection objects only (no metadata writes)
  -> build complete logical RED/YELLOW metadata plan
  -> final XMP commit
  -> mirror one logical role to exact RAW+JPEG physical files
  -> temp cleanup
```

## Group workflow

Group selection builds its roster exclusively from the normal primary analysis pass. A face must become part of a stable primary `PersonTrack`; the later high-resolution stage is reserved solely for camera-attention verification of the RED shortlist and cannot add people to the roster.

Group mode does not assume a fixed number of physical groups. It first builds hard chronological blocks, analyses all faces, then splits/merges blocks from identity continuity. A one-frame group-like fragment is retained until the cross-block merge pass; only after that pass is the minimum take count enforced.

Within one physical group, short online identity tracks are followed by a fragment-merge pass. Two fragments can merge only when they never coexist in the same frame and agree on ArcFace identity, approximate position and face size. This prevents one temporarily missed child from becoming two roster entries.

Eye state is not treated as one universal absolute landmark threshold. With at least three reliable observations, each child receives a conservative per-person threshold derived from that child's own upper eye-opening observations. Extremely small/weak faces keep identity information but their fine eye/expression state is marked unknown. RED ranking uses continuous eye deficit plus a stronger penalty for obvious blinks, avoiding threshold-edge flips.

The log records per-frame group diagnostics and accepted split/merge boundaries so failures can be attributed to grouping, identity tracking, eye state, portrait preference, camera attention, or final ranking.

## Portrait-preference ranking

Portrait and Group modes share a small public ResNet-18 facial-beauty regressor through OpenCV DNN. It is deliberately exposed to the rest of the program as a generic `portrait_preference` backend so a later personalised ranker can replace or blend with it without rewriting selection. The model runs on CPU and does not share CUDA/VRAM with InsightFace.

Absolute public-model scores are never used to rank different people against each other. In Group mode, every stable `PersonTrack` is normalised only across that person's own takes. In Portrait mode FBP compares frames only inside one portrait series or one linked child. Faces below the configured source-size floor are treated as unknown/neutral, and portrait mode falls back to the legacy score when there is not enough reliable FBP coverage for a real comparison.

In Portrait mode suitability is evaluated first while head rotation is intentionally allowed. Clearly closed eyes, definite blur and poor technical quality are hard defects. Eye/sharpness readings just below their thresholds and unknown landmark state form a separate uncertain tier: a clean frame always outranks it, but it remains available as fallback when no cleaner take exists. FBP is then the primary ordering key inside the best suitability tier only when at least two candidates have reliable FBP values and the configured minimum coverage is satisfied. Once FBP is usable, reliable FBP candidates form the preferred sub-pool and a legacy fallback value is never compared directly against an FBP value on the same numeric axis. Distinct-pose YELLOW candidates follow the same all-FBP-or-all-fallback rule across clusters.

RED ranking is hierarchical. In Group mode missing faces, clearly closed eyes, excessive head turn, definite blur and poor technical quality are hard defects evaluated first. Borderline eye/sharpness measurements and uncertain fine-state estimates form a lower uncertainty tier rather than an immediate hard reject. When camera attention is enabled and reliable, a protected gaze tier is inserted next; it can reorder only frames from the same complete suitability tier. FBP availability is decided once for the entire best suitability tier by repeated reliable measurements across stable PersonTracks; there is no per-frame usable/not-usable bit. Unknown FBP values are neutral on the same child-relative scale. After the minimum gaze condition, the dominant criteria are: number of people near their own better portrait-preference takes, the lower-tail preference score, mean relative preference, then aggregate preference. The previous group quality score remains a later tie-breaker. This means an excellent beauty score can never compensate for a known unusable face or confirmed away gaze, while disabling camera attention restores the ordinary suitability -> FBP -> quality order exactly.


## One face stack

The portrait build deliberately has one face-analysis stack:

- SCRFD from the InsightFace `buffalo_l` pack for full-frame detection;
- normalized recognition embeddings for identity matching;
- `2d106det.onnx` landmarks for eye geometry, mouth geometry and local eye crops;
- OpenCV/Numpy image metrics for face/eye sharpness and technical quality.

There is no secondary detector/identity/landmark backend in the current build. A detector or
landmark failure therefore cannot create disagreement between two face systems. The
portrait-preference network consumes the already detected crop only; it never creates,
removes, or re-identifies a face.

## CUDA execution

Provider order in `auto` mode is CUDA then CPU. For CUDA convolutions the build
uses `cudnn_conv_algo_search=HEURISTIC` by default. EXHAUSTIVE previously
triggered cuDNN execution-plan failures on the tested RTX 5090, while DEFAULT
maps to cuDNN frontend fallback heuristics in current ORT and may be very slow. A detector smoke test is executed during analyzer startup.
If ONNX Runtime reports a CUDA/cuDNN execution failure, the InsightFace
application is rebuilt with `CPUExecutionProvider` and the same operation/frame
is retried. The goal is correctness first: a GPU failure must not silently turn
into a missing child.

The GUI exposes:

- provider: Auto / CUDA / CPU;
- cuDNN convolution algorithm: HEURISTIC / EXHAUSTIVE / DEFAULT (diagnostic);
- automatic CUDA -> CPU recovery toggle.

## Portrait series

A hard block is created first from capture time and filename continuity. Within
that bounded interval, recognition embeddings are clustered with DBSCAN using
cosine distance. Cluster labels are then traversed in exact shooting order.
Thus:

```text
A A A | B B | A A A
```

always becomes three shooting series, even though the first and last chunks are
recognized as the same person.

Frames without a reliable embedding are bridged conservatively by neighbouring
chronological labels instead of being discarded.


## Series evidence and hysteresis

Detection recall and series acceptance use different thresholds. SCRFD can run
with a low detection threshold (0.25 by default), while a new portrait series
requires at least two detections above the stronger confirmation threshold
(0.45 by default). This prevents a one-off false face on a wall/floor/reflection
from becoming a RED selection.

DBSCAN remains strict (cosine distance 0.27). Only adjacent chronological
fragments are eligible for a looser centroid merge (0.32). Hard temporal blocks
can also be merged when they are adjacent, close in time and have matching
centroids. No merge ever jumps over another accepted portrait series, so A/B/A
remains three series.

Frames without an embedding are bridged only when the same identity is visible
on both sides. Leading/trailing no-face frames are no longer automatically
assigned to the nearest child.

## Portrait fallback scoring

Portrait RED selection is hierarchical: suitability first, then FBP when the best suitability tier has enough reliable FBP coverage. The legacy normalized score is retained only as an internal fallback/tie-breaker when FBP cannot be compared reliably. Its fixed components are open eyes, eye/face sharpness, expression proxy, smile proxy and technical quality; these weights are intentionally not exposed as user profiles or editable selection weights. Eye/smile/expression geometry comes from the 106-point landmarks.

## XMP

Normal Portrait produces only the RED winner. In optional multi-pose Portrait, one global RED plus YELLOW selections for accepted distinct poses may be produced. Group mode likewise produces RED/YELLOW `Selection` objects during analysis, but **none of these stages writes metadata directly**. Group analysis may produce up to 5 YELLOW candidates per series when configured; the cap is applied inside candidate selection, not only at metadata output.

The metadata invariant is:

```text
ANALYZE EVERYTHING FIRST
-> COMPUTE COMPLETE LOGICAL RED/YELLOW PLAN
-> COMMIT METADATA LAST
```

The logical metadata key is the case-insensitive path without extension. Thus `IMG_0001.CR2`, `IMG_0001.JPG` and `IMG_0001.xmp` belong to one resource. A RED/YELLOW conflict is resolved deterministically as RED > YELLOW. Exact RAW+JPEG pairs are analysed once using RAW, while the JPEG is retained as a metadata mirror and exact preview fallback.

Cancellation is checked for the final time immediately before commit. If cancellation or an analysis exception occurs before that point, XMP is untouched. Once commit starts, cancellation is deliberately not checked between files; individual metadata failures are logged and counted while remaining resources continue. This is a deferred commit, not a rollback-capable database transaction.

RED/YELLOW cleanup is unconditional. During final commit, the configured RED and YELLOW label values are first removed from every physical file, including resources that will be selected again and regardless of which application/user created the label. Only after the cleanup pass has completed does the pipeline write the new logical RED/YELLOW plan. There are no user-facing or configuration switches for skipping this cleanup. It remains deferred: no metadata is touched before the final commit begins.

Existing XMP is treated as user data and is never rebuilt with an XML serializer. Only the `xmp:Label` property is surgically changed or removed; Camera Raw settings, rating, crop, masks, keywords, custom namespaces and packet formatting remain untouched. Sidecars are atomically replaced after the surgical edit. Proprietary RAW files always use sidecars; DNG may use embedded XMP. JPEG standard XMP is updated inside its APP1 segment without pixel recompression. PSD/PSB uses Photoshop Image Resource 1060: fixed-size in-place editing is preferred, but when the packet must grow or resource 1060 is absent, only the length-delimited Image Resources section is streamed to an atomic replacement so layers/pixel data and neighboring resources remain byte-for-byte unchanged. Existing sidecars are never deleted.

## Fast scan and capture-time cache

The scan stage first collects path, size, `mtime_ns`, `mtime`, a prioritized numeric filename index and a case-insensitive `(parent folder, prefix before that index)` sequence source. A suffix after the selected counter is ignored, so exported names such as `IMG_0001_edit` and `IMG_0002_final` remain consecutive. A conventional camera prefix and a 3-6 digit counter outrank short edit-version suffixes. No ExifRead call is required for this filesystem index.

Group mode sorts primarily by sequence source and filename sequence. EXIF is probed only around suspicious boundaries: missing/duplicate numbers, source changes, backward/repeated numbers, filename gaps above the configured threshold, or filesystem `mtime` gaps above the configured time threshold. Source changes are hard candidate-series boundaries. `mtime` is a cheap probe trigger/fallback, not a replacement for trustworthy capture time; Windows creation time is intentionally not used.

Portrait mode remains conservative: valid cache entries are reused, then every uncached file is read with ExifRead and sorted EXIF-first. The persistent `runtime/cache/capture_times.json` stores only trustworthy EXIF capture times and validates entries by resolved case-insensitive path + file size + `mtime_ns`. Cache replacement uses a temporary file followed by `os.replace`.

An ExifRead exception is isolated to the affected file. Missing EXIF DateTime is not fatal and falls back to filesystem `mtime`.

RAW preview loading reports whether the successful source was `rawpy_preview`, `embedded_jpeg`, or `demosaic`. Every pipeline preview load uses the common RAW -> exact paired JPEG fallback wrapper.


## Windows CUDA DLL loading

The pip-installed NVIDIA CUDA 12 components live in separate `site-packages/nvidia/*/bin` directories. On Windows, Photo Select AI adds all of these directories to the process DLL search path and `%PATH%` before creating ONNX Runtime sessions, then explicitly preloads NVRTC/nvJitLink/cuBLAS runtime libraries. This is required because cuDNN can load those sublibraries lazily while building/executing convolution graphs.

GPU ORT sessions are CUDA-only at session level. If a genuine CUDA runtime failure still occurs and application-level fallback is enabled, the whole InsightFace application is rebuilt once on CPU instead of allowing every InsightFace model session to independently print CUDA→CPU fallback errors.

## Group camera-attention pass

Camera attention is a protected final-stage Group criterion, not a grouping signal. It is enabled by default. The normal pipeline first builds the stable roster and fixes the complete technical-suitability tier. A roster identity must be observed in at least two frames and at least 25% of the series, which prevents a rare passer-by or assistant from becoming a required group member. Candidates from the best tier are re-read at a larger preview in FBP-ordered batches (default 5). If a batch has no frame with zero confirmed away gazes and sufficient positive coverage, the next batch is analyzed automatically. Once that minimum condition is met, FBP remains primary among the protected candidates. For each matched stable identity, the analyzer combines a soft frontal-head estimate from the 106 landmarks with a conservative dark iris/pupil-centre estimate inside the eye contour. Small/closed/low-contrast eyes remain unknown rather than being treated as looking away. If camera attention is disabled or produces no reliable measurements, the ordinary suitability -> FBP -> quality base ranking is used unchanged.

This remains a conservative geometric/image heuristic rather than a semantic gaze neural network. Logs expose `look_known`, `look_away`, and `look_mean` together with portrait-preference diagnostics.


## Optional parallel InsightFace execution

The default execution path remains conservative: one InsightFace application plus bounded CPU preview prefetch. When `runtime.parallel_face_analysis` is enabled, a stage may own 2–4 independent InsightFace applications. A worker borrows exactly one analyzer for one frame, so no `FaceAnalysis`/ORT session object is called concurrently from multiple threads. Completed assessments are written back to their original frame indices before any grouping or scoring logic runs.

Primary analyzers are closed before the dedicated Group camera-attention stage is created. There is no separate face-rescue analyzer pool. Thus parallelism multiplies VRAM only inside the current stage rather than keeping primary and gaze pools resident together. If creation of an additional analyzer fails, the gaze stage continues with the successfully created pool instead of aborting the shoot.


## Bounded CUDA memory

When `runtime.gpu_memory_safe_mode` is enabled, every CUDAExecutionProvider receives a per-session `gpu_mem_limit` plus `arena_extend_strategy=kSameAsRequested`. The user-requested parallelism still applies to the primary all-files pass, while Group secondary stages are capped independently by `runtime.group_secondary_face_workers` (default 2). Dedicated gaze pools may be recycled every `runtime.group_gpu_recycle_every` groups (default 8), explicitly closing InsightFace model references and forcing a GC pass before recreation. This keeps the throughput benefit of 3–4 primary sessions without leaving 3–4 high-resolution gaze model packs resident for the whole shoot.


## Portrait multi-pose selection

The optional `best_red_pose_yellow` mode runs strictly after normal portrait identity refinement. Nearby refined series may be linked only by robust ArcFace evidence; an ambiguous near-tie is rejected. The linked series remain separate chronological objects, so DBSCAN/sequential boundaries and the boundary guard are not rewritten.

Pose discovery then works across **all frames of the linked child**, because several poses commonly remain inside a single identity series. Each frame already carries a soft yaw/pitch estimate from the same 106 landmarks, plus normalized face centre and scale. Clustering is RED-anchored and conservative: a strong head-angle change is sufficient, while framing/zoom alone is never sufficient. Alternative pose clusters require at least two supporting frames by default.

The RED winner is selected once from the union of every usable frame for that child. Each accepted pose cluster outside the RED pose competes internally for one YELLOW. The number of YELLOW pose alternatives is capped (default 3). This stage does not re-read images or run another neural model. It is intentionally face/head-pose based rather than full-body pose estimation.


## Runtime resilience and installation integrity

The portrait-preference/FBP backend is optional at runtime. `install.bat` still downloads and fully validates it, but `InsightFaceAnalyzer` catches a missing or unloadable FBP model and continues with the established legacy ranking fallback. Normal `run.bat` model validation requires only the three InsightFace networks actually used by Photo Select AI (detection, ArcFace recognition, 2d106 landmarks); the two unused buffalo_l extras and FBP cannot force an unnecessary runtime download.

The FBP download is pinned to upstream commit `e93ff99a3a3bf27694d6fa0b6d66dae5cb651d0c` rather than a moving branch. Installation records SHA-256 values in the model manifest after CPU inference validation. Dependency validation also checks the Microsoft Visual C++ runtime on Windows and executes `pip check`. UI state has an explicit schema version/migration policy so very old algorithm thresholds do not silently override current defaults.

The deferred XMP commit remains intentionally non-transactional, but any clear/write failure now raises a commit error after the complete plan has been attempted. The pipeline never emits the final `Готово` state in that case, the CLI exits non-zero, and the GUI reports an error that metadata may have been partially changed.
