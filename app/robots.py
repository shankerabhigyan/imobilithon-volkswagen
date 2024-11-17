"""
Robot Types:
- Welding Robot
- Inspection Robot
- Cleaning Robot
- Painting Robot
- Drying Robot
- Assembly Robot
- Packaging Robot
- Testing Robot
- Custom Robot
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum
    
class RobotCapabilities(BaseModel):
    # welding
    spot_welding: Optional[bool] = False
    arc_welding: Optional[bool] = False
    mig_welding: Optional[bool] = False

    # inspection
    inspection: Optional[bool] = False

    # cleaning
    cleaning: Optional[bool] = False

    # paint booth
    painting: Optional[bool] = False
    spraying: Optional[bool] = False
    drying: Optional[bool] = False

    # assembly
    transport: Optional[bool] = False
    assembly: Optional[bool] = False
    pick_and_place: Optional[bool] = False
    packaging: Optional[bool] = False
    screwing: Optional[bool] = False

    # testing
    testing: Optional[bool] = False
    measurement: Optional[bool] = False
    quality_check: Optional[bool] = False

    # option to add custom capabilities
    custom_capabilities: Dict[str, Any] = Field(default_factory=dict)

    core_capabilities:List= [
        "spot_welding",
        "arc_welding",
        "mig_welding",
        "inspection",
        "cleaning",
        "painting",
        "spraying",
        "drying",
        "transport",
        "assembly",
        "pick_and_place",
        "packaging",
        "screwing",
        "testing",
        "measurement",
        "quality_check"
    ]

    def add_custom_capabilities(self, key: str, value: Any):
        if key not in self.core_capabilities:
            self.custom_capabilities[key] = value
        else :
            raise ValueError(f"{key} is a core capability and cannot be added as a custom capability")

    def remove_custom_capabilities(self, key: str):
        self.custom_capabilities.pop(key, None)


class RobotSpecs(BaseModel):
    robot_name: str
    robot_id: str
    robot_capabilities: RobotCapabilities

    payload_capacity: Optional[float] = 0.0 # in kgs
    arm_length: Optional[float] = 0.0 # in meters
    reach: Optional[float] = 0.0 # in meters
    speed: Optional[float] = 0.0 # in m/s
    acceleration: Optional[float] = 0.0 # in m/s^2
    accuracy: Optional[float] = 0.0 # in mm (measure of the robot's ability to reach a specific point)
    repeatability: Optional[float] = 0.0 # in mm (repeatability means the ability of the robot to return to the same position)
    degrees_of_freedom: Optional[int] = 0 # number of joints in the robot - affects the robot's flexibility
    power_consumption: Optional[float] = 0.0 # in kW
    robot_cost: Optional[float] = 0.0 # in USD

    # option to add custom specs
    custom_specs: Dict[str, Any] = Field(default_factory=dict)

    core_specs: List= [
        "robot_name",
        "robot_id",
        "robot_capabilities",
        "payload_capacity",
        "arm_length",
        "reach",
        "speed",
        "acceleration",
        "accuracy",
        "repeatability",
        "degrees_of_freedom",
        "power_consumption",
        "robot_cost"
    ]

    def add_custom_specs(self, key: str, value: Any):
        if key not in self.core_specs:
            self.custom_specs[key] = value
        else :
            raise ValueError(f"{key} is a core spec and cannot be added as a custom spec")

    def remove_custom_specs(self, key: str):
        self.custom_specs.pop(key, None)

class RobotState(BaseModel):
    """
    to be used for predicting throughput, efficiency, lifecycle and failure of the robot
    this is a dummy class to be replaced with sensor data in the actual implementation
    """
    status: bool = False # True if the robot is active, False if the robot is inactive
    timestep: float = 0.5 # time elapsed since the robot was activated
    temperature: List[float] = [0.0, 0.0, 0.0] # current vs min vs max
    humidity: List[float] = [0.0, 0.0, 0.0] # current vs min vs max
    pressure: List[float] = [0.0, 0.0, 0.0] # current vs min vs max
    vibration: List[float] = [0.0, 0.0, 0.0] # current vs min vs max

    def update_temperature(self, current: float, min: float, max: float):
        self.temperature = [current, min, max]

    def update_humidity(self, current: float, min: float, max: float):
        self.humidity = [current, min, max]

    def update_pressure(self, current: float, min: float, max: float):
        self.pressure = [current, min, max]

    def update_vibration(self, current: float, min: float, max: float):
        self.vibration = [current, min, max]

    def update_status(self, status: bool):
        self.status = status

    def update_timestep(self, timestep: float):
        self.timestep = timestep

    def reset_state(self):
        self.status = False
        self.timestep = 0.5
        self.temperature = [0.0, 0.0, 0.0]
        self.humidity = [0.0, 0.0, 0.0]
        self.pressure = [0.0, 0.0, 0.0]
        self.vibration = [0.0, 0.0, 0.0]

class RobotType(BaseModel):
    robot_type: str
    robot_specs: RobotSpecs

# predefined robot types

class WeldingRobotType(RobotType):
    robot_type: str = "Welding Robot"
    robot_specs: RobotSpecs = RobotSpecs(
        robot_name="Welding Robot",
        robot_id="WELDING-ROBOT",
        robot_capabilities=RobotCapabilities(
            spot_welding=True,
            arc_welding=True,
            mig_welding=True
        ),
        payload_capacity=10.0,
        arm_length=1.0,
        reach=1.0,
        speed=0.5,
        acceleration=0.5,
        accuracy=0.1,
        repeatability=0.1,
        degrees_of_freedom=6,
        power_consumption=1.0,
        robot_cost=10000.0
    )
    state: RobotState = RobotState()
    state.update_temperature(25.0, 20.0, 35.0) # current vs min vs max
    state.update_humidity(50.0, 40.0, 60.0) # current vs min vs max
    state.update_pressure(100.0, 90.0, 110.0) # current vs min vs max
    state.update_vibration(0.0, 0.0, 0.0) # current vs min vs max
    state.update_timestep(600) # 10 minutes 

class InspectionRobotType(RobotType):
    robot_type: str = "Inspection Robot"
    robot_specs: RobotSpecs = RobotSpecs(
        robot_name="Inspection Robot",
        robot_id="INSPECTION-ROBOT",
        robot_capabilities=RobotCapabilities(
            inspection=True
        ),
        payload_capacity=5.0,
        arm_length=0.5,
        reach=0.5,
        speed=0.5,
        acceleration=0.5,
        accuracy=0.1,
        repeatability=0.1,
        degrees_of_freedom=4,
        power_consumption=0.5,
        robot_cost=5000.0
    )
    state: RobotState = RobotState()
    state.update_temperature(25.0, 20.0, 35.0) # current vs min vs max
    state.update_humidity(50.0, 40.0, 60.0) # current vs min vs max
    state.update_pressure(100.0, 90.0, 110.0) # current vs min vs max
    state.update_vibration(0.0, 0.0, 0.0) # current vs min vs max
    state.update_timestep(600) # 10 minutes

class CleaningRobotType(RobotType):
    robot_type: str = "Cleaning Robot"
    robot_specs: RobotSpecs = RobotSpecs(
        robot_name="Cleaning Robot",
        robot_id="CLEANING-ROBOT",
        robot_capabilities=RobotCapabilities(
            cleaning=True
        ),
        payload_capacity=5.0,
        arm_length=0.5,
        reach=0.5,
        speed=0.5,
        acceleration=0.5,
        accuracy=0.1,
        repeatability=0.1,
        degrees_of_freedom=4,
        power_consumption=0.5,
        robot_cost=5000.0
    )
    state: RobotState = RobotState()
    state.update_temperature(25.0, 20.0, 35.0) # current vs min vs max
    state.update_humidity(50.0, 40.0, 60.0) # current vs min vs max
    state.update_pressure(100.0, 90.0, 110.0) # current vs min vs max
    state.update_vibration(0.0, 0.0, 0.0) # current vs min vs max
    state.update_timestep(600) # 10 minutes

