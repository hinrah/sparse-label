import os

class Labels:
    BACKGROUND = 0
    WALL = 1
    LUMEN = 2
    UNKNOWN = 3
    UNPROCESSED = -1

class Folders:
    IMAGES = "images"
    LABELS = "labels"
    CENTERLINES = "centerlines"
    CONTOURS = "contours"

class Endings:
    JSON = ".json"
    NIFTI = ".nii.gz"
    CHANNEL_ZERO = "_0000"

class Contours:
    INNER = "inner_contour"
    OUTER = "outer_contour"

data_raw = os.environ.get("sparseVesselMasks_raw")

if data_raw is None:
    print("data_raw is not defined nnUNet_raw is now used.")
    data_raw = os.environ.get('nnUNet_raw')
    if data_raw is None:
        print("nnUNet_raw is not defined as well. Sparse vessel masks can not be created.")
