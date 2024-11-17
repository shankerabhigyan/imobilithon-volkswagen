from typing import List, Dict, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from app.robots import RobotType, RobotState, WeldingRobotType, InspectionRobotType, CleaningRobotType

class ProcessLoad(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProcessStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class ProcessParameter(BaseModel):
    name: str
    value: Any
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def validate_range(self) -> bool:
        if isinstance(self.value, (int, float)) and self.min_value is not None and self.max_value is not None:
            return self.min_value <= self.value <= self.max_value
        return True

class ProcessStep(BaseModel):
    step_id: str
    name: str
    description: str
    required_capabilities: List[str]
    parameters: List[ProcessParameter]
    load: ProcessLoad
    estimated_duration: timedelta
    predecessor_steps: List[str] = Field(default_factory=list)
    status: ProcessStatus = ProcessStatus.NOT_STARTED

    def validate_prerequisites(self, completed_steps: List[str]) -> bool:
        return all(step in completed_steps for step in self.predecessor_steps)

    def calculate_load_impact(self) -> float:
        """Calculate the impact of this step's load on robot state"""
        load_factors = {
            ProcessLoad.LOW: 0.1,
            ProcessLoad.MEDIUM: 0.25,
            ProcessLoad.HIGH: 0.5,
            ProcessLoad.CRITICAL: 0.75
        }
        return load_factors[self.load]

class RobotStateManager:
    """Manages and updates robot state based on process loads"""
    
    @staticmethod
    def update_robot_state_for_load(robot: RobotType, load_impact: float):
        """Update robot state based on process load impact"""
        # Get current values
        current_temp = robot.state.temperature[0]
        max_temp = robot.state.temperature[2]
        current_pressure = robot.state.pressure[0]
        max_pressure = robot.state.pressure[2]
        
        # Calculate new values based on load impact
        temp_increase = (max_temp - current_temp) * load_impact / 2
        pressure_increase = (max_pressure - current_pressure) * load_impact / 2
        
        # Update temperature
        new_temp = min(current_temp + temp_increase, max_temp)
        robot.state.update_temperature(
            new_temp,
            robot.state.temperature[1],
            robot.state.temperature[2]
        )
        
        # Update pressure
        new_pressure = min(current_pressure + pressure_increase, max_pressure)
        robot.state.update_pressure(
            new_pressure,
            robot.state.pressure[1],
            robot.state.pressure[2]
        )

    @staticmethod
    def check_robot_state_limits(robot: RobotType) -> bool:
        """Check if robot state is within safe limits"""
        temp_margin = 0.95  # 95% of max temperature
        pressure_margin = 0.95  # 95% of max pressure
        
        current_temp = robot.state.temperature[0]
        max_temp = robot.state.temperature[2]
        current_pressure = robot.state.pressure[0]
        max_pressure = robot.state.pressure[2]
        
        return (current_temp <= max_temp * temp_margin and 
                current_pressure <= max_pressure * pressure_margin)

class Process(BaseModel):
    process_id: str
    name: str
    description: str
    steps: List[ProcessStep]
    required_robot_type: RobotType
    status: ProcessStatus = ProcessStatus.NOT_STARTED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def get_total_duration(self) -> timedelta:
        return sum((step.estimated_duration for step in self.steps), timedelta())

    def get_next_executable_steps(self, completed_steps: List[str]) -> List[ProcessStep]:
        return [
            step for step in self.steps 
            if step.status == ProcessStatus.NOT_STARTED 
            and step.validate_prerequisites(completed_steps)
        ]

    def execute_process(self, state_manager: RobotStateManager = RobotStateManager()):
        """Execute the process while managing robot state"""
        print(f"Starting process: {self.name}")
        self.start_time = datetime.now()
        self.status = ProcessStatus.IN_PROGRESS
        completed_steps = []

        try:
            while True:
                executable_steps = self.get_next_executable_steps(completed_steps)
                if not executable_steps:
                    break

                for step in executable_steps:
                    print(f"Executing step: {step.name} with {step.load} load")
                    
                    # Calculate and apply load impact
                    load_impact = step.calculate_load_impact()
                    state_manager.update_robot_state_for_load(self.required_robot_type, load_impact)
                    
                    # Check if robot state is still within safe limits
                    if not state_manager.check_robot_state_limits(self.required_robot_type):
                        raise ValueError(f"Robot state exceeded safe limits during step: {step.name}")
                    
                    step.status = ProcessStatus.COMPLETED
                    completed_steps.append(step.step_id)
                    
                    # Print current robot state
                    print(f"Current robot temperature: {self.required_robot_type.state.temperature[0]:.2f}")
                    print(f"Current robot pressure: {self.required_robot_type.state.pressure[0]:.2f}")

            self.status = ProcessStatus.COMPLETED
            self.end_time = datetime.now()
            print(f"Process {self.name} completed successfully")
            
        except Exception as e:
            self.status = ProcessStatus.FAILED
            print(f"Process failed: {str(e)}")
            raise

def create_welding_process(robot: WeldingRobotType) -> Process:
    return Process(
        process_id="WELD-001",
        name="Pipeline Welding Process",
        description="Automated welding process for pipeline sections",
        required_robot_type=robot,
        steps=[
            ProcessStep(
                step_id="WELD-001-PREP",
                name="Surface Preparation",
                description="Clean and prepare the welding surface",
                required_capabilities=["cleaning"],
                load=ProcessLoad.LOW,
                parameters=[
                    ProcessParameter(
                        name="surface_roughness",
                        value=0.5,
                        unit="mm",
                        min_value=0.1,
                        max_value=1.0
                    )
                ],
                estimated_duration=timedelta(minutes=5)
            ),
            ProcessStep(
                step_id="WELD-001-TACK",
                name="Tack Welding",
                description="Create tack welds to hold pieces in position",
                required_capabilities=["spot_welding"],
                load=ProcessLoad.HIGH,
                parameters=[
                    ProcessParameter(
                        name="weld_current",
                        value=150,
                        unit="A",
                        min_value=100,
                        max_value=200
                    )
                ],
                estimated_duration=timedelta(minutes=10),
                predecessor_steps=["WELD-001-PREP"]
            )
        ]
    )

# Example usage:
if __name__ == "__main__":
    # Create robot instance
    welding_robot = WeldingRobotType()
    
    # Create process
    welding_process = create_welding_process(welding_robot)
    
    # Create state manager
    state_manager = RobotStateManager()
    
    try:
        # Execute process
        welding_process.execute_process(state_manager)
        
        # Print final robot state
        print("\nFinal Robot State:")
        print(f"Temperature: {welding_robot.state.temperature}")
        print(f"Pressure: {welding_robot.state.pressure}")
        
    except Exception as e:
        print(f"Process execution failed: {str(e)}")