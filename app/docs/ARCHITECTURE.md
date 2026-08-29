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

## Group workflow (v0.4.3)

After physical-group boundaries are established by the primary detector, group selection uses a two-pass roster. The primary pass contributes stable tracks only; one-frame promotion is disabled. A second InsightFace analyzer runs at a larger detector input and lower confidence threshold, and only repeated unmatched tracks that fit the stable group geometry may extend the roster. The primary analyzer is released before the second one is created to limit CUDA VRAM pressure.

Group mode does not assume a fixed number of physical groups. It first builds hard chronological blocks, analyses all faces, then splits/merges blocks from identity continuity. A one-frame group-like fragment is retained until the cross-block merge pass; only after that pass is the minimum take count enforced.

Within one physical group, short online identity tracks are followed by a fragment-merge pass. Two fragments can merge only when they never coexist in the same frame and agree on ArcFace identity, approximate position and face size. This prevents one temporarily missed child from becoming two roster entries.

Eye state is not treated as one universal absolute landmark threshold. With at least three reliable observations, each child receives a conservative per-person threshold derived from that child's own upper eye-opening observations. Extremely small/weak faces keep identity information but their fine eye/expression state is marked unknown. RED ranking uses continuous eye deficit plus a stronger penalty for obvious blinks, avoiding threshold-edge flips.

The log records per-frame group diagnostics and accepted split/merge boundaries so failures can be attributed to grouping, identity tracking, eye state, or final ranking.


## One face stack

The portrait build deliberately has one face-analysis stack:

- SCRFD from the InsightFace `buffalo_l` pack for full-frame detection;
- normalized recognition embeddings for identity matching;
- `2d106det.onnx` landmarks for eye geometry, mouth geometry and local eye crops;
- OpenCV/Numpy image metrics for face/eye sharpness and technical quality.

There is no secondary face-analysis backend in the current build. A detector or
landmark failure therefore cannot create disagreement between two face systems.

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


## v0.2.5 series evidence and hysteresis

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

## Portrait scoring

The GUI maps directly to independent normalized weights:

- open eyes;
- closed-eye penalty;
- eye sharpness;
- face sharpness;
- expression proxy;
- smile proxy;
- technical quality.

A weight of zero disables that criterion. Eye/smile/expression geometry comes
from the 106-point landmarks. These are ranking signals, not semantic emotion
classification.

## XMP

Normal Portrait produces only the RED winner. In optional multi-pose Portrait, one global RED plus YELLOW selections for accepted distinct poses may be produced. Group mode likewise produces RED/YELLOW `Selection` objects during analysis, but **none of these stages writes metadata directly**. Group analysis may produce up to 5 YELLOW candidates per series when configured; the cap is applied inside candidate selection, not only at metadata output.

The v0.5.9 metadata invariant is:

```text
ANALYZE EVERYTHING FIRST
-> COMPUTE COMPLETE LOGICAL RED/YELLOW PLAN
-> COMMIT METADATA LAST
```

The logical metadata key is the case-insensitive path without extension. Thus `IMG_0001.CR2`, `IMG_0001.JPG` and `IMG_0001.xmp` belong to one resource. A RED/YELLOW conflict is resolved deterministically as RED > YELLOW. Exact RAW+JPEG pairs are analysed once using RAW, while the JPEG is retained as a metadata mirror and exact preview fallback.

Cancellation is checked for the final time immediately before commit. If cancellation or an analysis exception occurs before that point, XMP is untouched. Once commit starts, cancellation is deliberately not checked between files; individual metadata failures are logged and counted while remaining resources continue. This is a deferred commit, not a rollback-capable database transaction.

The clear options are literal and apply to the complete discovered resource set. During final commit, enabled configured RED/YELLOW labels are first removed from every physical file, including resources that will be selected again and regardless of which application/user created the label. Only after the cleanup pass has completed does the pipeline write the new logical RED/YELLOW plan. This remains deferred: no metadata is touched before the final commit begins.

Existing XMP is treated as user data and is never rebuilt with an XML serializer. Only the `xmp:Label` property is surgically changed or removed; Camera Raw settings, rating, crop, masks, keywords, custom namespaces and packet formatting remain untouched. Sidecars are atomically replaced after the surgical edit. Proprietary RAW files always use sidecars; DNG may use embedded XMP. JPEG standard XMP is updated inside its APP1 segment without pixel recompression. PSD/PSB uses Photoshop Image Resource 1060: fixed-size in-place editing is preferred, but when the packet must grow or resource 1060 is absent, only the length-delimited Image Resources section is streamed to an atomic replacement so layers/pixel data and neighboring resources remain byte-for-byte unchanged. Existing sidecars are never deleted.

## Fast scan and capture-time cache (v0.5.9)

The scan stage first collects path, size, `mtime_ns`, `mtime`, trailing filename sequence and a case-insensitive `(parent folder, prefix before trailing digits)` sequence source. No ExifRead call is required for this filesystem index.

Group mode sorts primarily by sequence source and filename sequence. EXIF is probed only around suspicious boundaries: missing/duplicate numbers, source changes, backward/repeated numbers, filename gaps above the configured threshold, or filesystem `mtime` gaps above the configured time threshold. Source changes are hard candidate-series boundaries. `mtime` is a cheap probe trigger/fallback, not a replacement for trustworthy capture time; Windows creation time is intentionally not used.

Portrait mode remains conservative: valid cache entries are reused, then every uncached file is read with ExifRead and sorted EXIF-first. The persistent `runtime/cache/capture_times.json` stores only trustworthy EXIF capture times and validates entries by resolved case-insensitive path + file size + `mtime_ns`. Cache replacement uses a temporary file followed by `os.replace`.

An ExifRead exception is isolated to the affected file. Missing EXIF DateTime is not fatal and falls back to filesystem `mtime`.

RAW preview loading reports whether the successful source was `rawpy_preview`, `embedded_jpeg`, or `demosaic`. Every pipeline preview load uses the common RAW -> exact paired JPEG fallback wrapper.


## Windows CUDA DLL loading (v0.2.5)

The pip-installed NVIDIA CUDA 12 components live in separate `site-packages/nvidia/*/bin` directories. On Windows, Photo Select AI adds all of these directories to the process DLL search path and `%PATH%` before creating ONNX Runtime sessions, then explicitly preloads NVRTC/nvJitLink/cuBLAS runtime libraries. This is required because cuDNN can load those sublibraries lazily while building/executing convolution graphs.

GPU ORT sessions are CUDA-only at session level. If a genuine CUDA runtime failure still occurs and application-level fallback is enabled, the whole InsightFace application is rebuilt once on CPU instead of allowing every InsightFace model session to independently print CUDA→CPU fallback errors.

## Experimental group camera-attention pass (v0.4.6)

Camera attention is a final-stage optional group criterion, not a grouping signal. The normal group pipeline first builds the stable roster and ordinary RED ranking. Only the strongest N frames (default 3) are re-read at a larger preview. For each matched stable identity, the analyzer combines a soft frontal-head estimate from the 106 landmarks with a conservative dark iris/pupil-centre estimate inside the eye contour. Small/closed/low-contrast eyes remain unknown and therefore neutral. The final attention score can only re-rank the ordinary shortlist; it cannot promote a weak/blinking frame from outside that shortlist.

This is intentionally an experimental geometric/image heuristic rather than a semantic gaze neural network. Logs expose `look_known`, `look_away`, and `look_mean` so real shoots can be used to validate whether the cue helps before it is given broader influence.


## Optional parallel InsightFace execution (v0.4.7)

The default execution path remains conservative: one InsightFace application plus bounded CPU preview prefetch. When `runtime.parallel_face_analysis` is enabled, a stage may own 2–4 independent InsightFace applications. A worker borrows exactly one analyzer for one frame, so no `FaceAnalysis`/ORT session object is called concurrently from multiple threads. Completed assessments are written back to their original frame indices before any grouping or scoring logic runs.

Primary analyzers are closed before the optional group high-resolution stage is created. The same rule applies to the dedicated gaze stage when high-resolution rescue is not active. Thus parallelism multiplies VRAM only inside the current stage rather than keeping primary + rescue + gaze pools resident together. If creation of an additional analyzer fails, the stage continues with the successfully created pool instead of aborting the shoot.


## Bounded CUDA memory (v0.5.1)

When `runtime.gpu_memory_safe_mode` is enabled, every CUDAExecutionProvider receives a per-session `gpu_mem_limit` plus `arena_extend_strategy=kSameAsRequested`. The user-requested parallelism still applies to the primary all-files pass, while Group secondary stages are capped independently by `runtime.group_secondary_face_workers` (default 2). Dedicated gaze pools may be recycled every `runtime.group_gpu_recycle_every` groups (default 8), explicitly closing InsightFace model references and forcing a GC pass before recreation. This keeps the throughput benefit of 3–4 primary sessions without leaving 3–4 high-resolution gaze model packs resident for the whole shoot.


## Portrait multi-pose selection (v0.5.2)

The optional `best_red_pose_yellow` mode runs strictly after normal portrait identity refinement. Nearby refined series may be linked only by robust ArcFace evidence; an ambiguous near-tie is rejected. The linked series remain separate chronological objects, so DBSCAN/sequential boundaries and the boundary guard are not rewritten.

Pose discovery then works across **all frames of the linked child**, because several poses commonly remain inside a single identity series. Each frame already carries a soft yaw/pitch estimate from the same 106 landmarks, plus normalized face centre and scale. Clustering is RED-anchored and conservative: a strong head-angle change is sufficient, while framing/zoom alone is never sufficient. Alternative pose clusters require at least two supporting frames by default.

The RED winner is selected once from the union of every usable frame for that child. Each accepted pose cluster outside the RED pose competes internally for one YELLOW. The number of YELLOW pose alternatives is capped (default 3). This stage does not re-read images or run another neural model. It is intentionally face/head-pose based rather than full-body pose estimation.
