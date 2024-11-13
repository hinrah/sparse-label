import dataclasses


@dataclasses.dataclass
class Metrics:
    dice_coefficients: dict
    hausdorff_distances: dict
    hausdorff_distances_95: dict
    average_contour_distances: dict
