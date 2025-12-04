from typing import Optional, Literal
from pydantic import BaseModel

class NIOSHParameters(BaseModel):
    weight: Optional[float] = None
    horizontal_origin: Optional[float] = None
    horizontal_destination: Optional[float] = None
    vertical_origin: Optional[float] = None
    vertical_destination: Optional[float] = None
    asymmetry_angle: Optional[float] = None
    frequency: Optional[float] = None
    duration: Optional[str] = None
    coupling: Optional[str] = None
    significant_control: Optional[bool] = None

    gender: Literal["M", "W"] = "M"
    age: int = 25
    judgment: Literal["good", "intermediate", "bad"] = "intermediate"
    etm: float = 1.0
    one_limb_lifting: bool = False
    two_operators_lifting: bool = False
