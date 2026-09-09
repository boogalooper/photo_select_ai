Photo Select AI portrait-preference model

The model files are downloaded by install.bat together with a fresh complete
InsightFace buffalo_l model pack on every installer run:

  beauty_resnet.caffemodel
  beauty_resnet.prototxt

Normal run.bat startup treats Facial Beauty Prediction (FBP) as optional.
If these files are missing, damaged, or fail the startup inference probe,
Photo Select AI continues with the deterministic legacy ranking fallback.
run.bat does not download FBP. Run install.bat to restore/revalidate it.

The installer pins the source to an immutable upstream commit, validates the
model with OpenCV DNN on CPU, and records SHA-256 values in the model manifest.

Source:
  https://github.com/asiryan/HowCuteAmI

The upstream demo loads this ResNet model with OpenCV DNN and uses a 224x224
face crop. Photo Select AI keeps this model behind a generic portrait-preference
interface so it can later be blended with or replaced by a personal ranker.

In Group mode absolute scores are not compared between different people: each
tracked person is ranked against their own takes. In Portrait mode the model
ranks frames only inside one portrait series / linked child after the technical
filter. If reliable FBP coverage is insufficient, the legacy portrait score is
used as a deterministic fallback.

This build is intended only for personal, non-commercial use.
