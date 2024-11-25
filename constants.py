import os

class Labels:
    BACKGROUND = 0
    WALL = 1
    LUMEN = 2
    UNKNOWN = 3
    UNPROCESSED = -1

class Folders:
    IMAGES = "imagesTr"
    LABELS = "labelsTr"
    PREDICTIONS = "predictionsTr"
    CENTERLINES = "centerlinesTr"
    CONTOURS = "contoursTr"
    FULLRES_TRAINER = "nnUNetTrainer__nnUNetPlans__3d_fullres"
    CROSS_VALIDATION_RESULTS = "crossval_results_folds_0_1_2_3_4"
    POSTPROCESSED = "postprocessed"

class Endings:
    JSON = ".json"
    NIFTI = ".nii.gz"
    CHANNEL_ZERO = "_0000"

class Contours:
    INNER = "inner_contour"
    OUTER = "outer_contour"

data_raw = os.environ.get("sparseVesselMasks_raw")

if data_raw is None:
    print("sparseVesselMasks_raw is not defined nnUNet_raw is now used.")
    data_raw = os.environ.get('nnUNet_raw')
    if data_raw is None:
        print("nnUNet_raw is not defined as well. Sparse vessel masks can not be created.")


data_results = os.environ.get("sparseVesselMasks_results")

if data_results is None:
    print("sparseVesselMasks_data_results is not defined nnUNet_data_results is now used.")
    data_results = os.environ.get('nnUNet_results')
    if data_results is None:
        print("nnUNet_results is not defined as well. Sparse vessel masks can not be evaluated.")
