ENCODING = "ascii"


class Folders:
    IMAGES = "images"
    LABELS = "labels"
    PREDICTIONS = "predictions"
    CENTERLINES = "centerlines"
    CONTOURS = "contours"
    DEFAULT_TRAINER = "nnUNetTrainer"
    DEFAULT_CONFIG = "3d_fullres"
    DEFAULT_PLANS = "nnUNetPlans"
    SEPERATOR = "__"
    CROSS_VALIDATION_RESULTS = "crossval_results_folds_0_1_2_3_4"
    POSTPROCESSED = "postprocessed"


class DatasetInfo:
    FILE_NAME = "dataset.json"
    LABELS = "labels"
    BACKGROUND = "background"
    LUMEN = "Lumen"
    WALL = "Wall"
    IGNORE = "ignore"
    CHANNELS = "channel_names"


class Endings:
    JSON = ".json"
    NIFTI = ".nii.gz"
    CHANNEL_ZERO = "_0000"


class Contours:
    INNER = "inner_contour"
    OUTER = "outer_contour"
    ENDING_NORMAL = "ending_normal"


class EnvironmentVars:
    sparse_vessel_masks_raw = "sparseVesselMasks_raw"
    sparse_vessel_masks_results = "sparseVesselMasks_results"
    nnunet_raw = "nnUNet_raw"
    nnunet_results = "nnUNet_results"
