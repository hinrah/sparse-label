import dataclasses
from typing import Union


@dataclasses.dataclass
class Parameters:
    identifier: str
    vessel_wall_thickness: Union[float, None]
    vessel_wall_thickness_manual: Union[float, None]
    lumen_diameter: Union[float, None]
    lumen_diameter_manual: Union[float, None]
    max_vel: Union[float, None]
    max_vel_manual: Union[float, None]
    flow: Union[float, None]
    flow_manual: Union[float, None]
    is_correct: bool
