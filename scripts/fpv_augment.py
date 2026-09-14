"""Training-time augmentation profiles that model this system's real input.

See ENGINEERING_LOG.md for the measurements behind this.
"""

PROFILES = ("none", "fpv")


def build(profile: str):
    """Return the Albumentations transform list for `profile`, or None.

    None means "let ultralytics use its own defaults", which is the historical
    behaviour (and, with albumentations absent, no augmentation at all).
    """
    if profile in (None, "none"):
        return None
    if profile != "fpv":
        raise ValueError(
            "Unknown augmentation profile {!r}; expected one of {}".format(
                profile, ", ".join(PROFILES)))

    import albumentations as A

    return [
        # --- optics: an airframe in motion, and imperfect focus ---
        # MotionBlur dominates the real failure cases (the night pass on
        # v6.mp4, every handheld clip). Capped at 9 px: a 13+ px kernel
        # removes an 11 px target completely.
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.Defocus(radius=(1, 4), alias_blur=(0.05, 0.3), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.30),

        # --- sensor resolution: an SD analog feed upscaled for inference ---
        # v11.mp4 arrives at 564x480 and is served at imgsz 1280, so the
        # model sees genuinely upsampled, detail-free pixels. Kept mild and
        # infrequent for the small-target reason in the module docstring.
        A.Downscale(scale_range=(0.5, 0.9), p=0.15),

        # --- sensor and transmission noise ---
        # ISONoise for the low-light gain case, MultiplicativeNoise
        # (elementwise) for the speckle an analog downlink adds.
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.07), intensity=(0.1, 0.6), p=1.0),
            A.GaussNoise(std_range=(0.03, 0.15), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=True,
                                  elementwise=True, p=1.0),
        ], p=0.30),

        # --- codec: the blocking/ringing all this footage carries ---
        A.ImageCompression(quality_range=(25, 70), p=0.35),

        # --- exposure: dusk, night, glare, auto-exposure hunting ---
        # Asymmetric on purpose (-0.35 vs +0.25): the operational failures in
        # this project are night and low-light passes, not overexposure.
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.25),
                                   contrast_limit=(-0.30, 0.30), p=0.45),
        A.RandomGamma(gamma_limit=(60, 140), p=0.25),
        A.CLAHE(clip_limit=(1, 4), p=0.10),

        # --- palette: false-colour and non-natural colour mappings ---
        # This is not cosmetic. v10.mp4 is an EO/IR-style feed whose vegetation
        # renders MAGENTA, nothing like the natural greens and greys of every
        # See ENGINEERING_LOG.md for the measurements behind this.
        A.HueSaturationValue(hue_shift_limit=90, sat_shift_limit=40,
                             val_shift_limit=25, p=0.35),
        A.ChannelShuffle(p=0.10),
        # Thermal/IR and washed-out monochrome downlinks.
        A.ToGray(p=0.08),
    ]


def describe(profile: str) -> str:
    transforms = build(profile)
    if not transforms:
        return "augmentation profile 'none' (ultralytics defaults)"
    return "augmentation profile '{}': {}".format(
        profile, ", ".join(type(t).__name__ for t in transforms))
