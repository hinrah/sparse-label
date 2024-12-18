import dataclasses
from typing import Union


@dataclasses.dataclass
class Metrics:
    identifier: str
    dice_coefficients: dict
    hausdorff_distances: dict
    hausdorff_distances_95: dict
    average_contour_distances: dict
    centerline_sensitivity: Union[float, None]
    is_correct: bool
