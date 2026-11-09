"""MCVT - Multi-Camera Vehicle Tracking engine.

Layers (SRS 2.1): detection (YOLOv8) -> tracking (DeepSORT) ->
re-identification + switching (ResNet-IBN, DBN) -> output (FFmpeg, SMTP).
Every component lives in its own module so it can be replaced on its own
(NFR-M-01).
"""

__version__ = "1.0.0"
