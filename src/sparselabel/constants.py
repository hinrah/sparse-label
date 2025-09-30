ENCODING = "ascii"


class Folders:
    IMAGES = "images"
    LABELS = "labels"
    CENTERLINES = "centerlines"
    CONTOURS = "contours"


class DatasetInfo:
    FILE_NAME = "dataset.json"
    LABELS = "labels"
    BACKGROUND = "background"
    LUMEN = "Lumen"
    WALL = "Wall"
    IGNORE = "ignore"
    CHANNELS = "channel_names"


class LabelStrategies:
    CONSIDERED_AREA_FACTOR = 1.1


class Endings:
    JSON = ".json"
    NIFTI = ".nii.gz"
    CHANNEL_ZERO = "_0000"


class Contours:
    INNER = "inner_contour"
    OUTER = "outer_contour"
    ENDING_NORMAL = "ending_normal"


class Evaluation:
    SEED = 1337
    SURFACE_SAMPLING = 10000000


class EnvironmentVars:
    sparse_vessel_masks_raw = "sparseVesselMasks_raw"
    sparse_vessel_masks_results = "sparseVesselMasks_results"
    nnunet_raw = "nnUNet_raw"
    nnunet_results = "nnUNet_results"
