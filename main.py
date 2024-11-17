import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import List, Dict
import numpy as np

from app.process import (
    Process, ProcessStep, ProcessParameter, ProcessLoad, 
    ProcessStatus, RobotStateManager
)
from app.robots import (
    WeldingRobotType, InspectionRobotType, CleaningRobotType,
    RobotState, RobotCapabilities
)
from models.bladeanomaly import MachineMonitor  # Import the MachineMonitor class

# Initialize session state
if 'processes' not in st.session_state:
    st.session_state.processes = []
if 'robots' not in st.session_state:
    st.session_state.robots = {
        'Welding Robot': WeldingRobotType(),
        'Inspection Robot': InspectionRobotType(),
        'Cleaning Robot': CleaningRobotType()
    }
if 'blade_monitor' not in st.session_state:
    st.session_state.blade_monitor = MachineMonitor()




def create_process_step() -> ProcessStep:
    """Create a new process step with user input"""
    st.subheader("Add Process Step")
    
    step_id = st.text_input("Step ID", f"STEP-{len(st.session_state.get('current_steps', [])) + 1}")
    name = st.text_input("Step Name")
    description = st.text_area("Step Description")
    
    # Capabilities selection based on robot type
    available_capabilities = []
    if st.session_state.get('selected_robot'):
        robot = st.session_state.robots[st.session_state.selected_robot]
        capabilities = robot.robot_specs.robot_capabilities
        available_capabilities = [cap for cap, enabled in capabilities.__dict__.items() 
                                if enabled and not cap.startswith('_')]
    
    required_capabilities = st.multiselect(
        "Required Capabilities",
        options=available_capabilities
    )
    
    load = st.selectbox(
        "Process Load",
        options=[load.value for load in ProcessLoad]
    )
    
    duration_minutes = st.number_input("Estimated Duration (minutes)", min_value=1)
    
    # Parameter inputs
    st.subheader("Step Parameters")
    parameters = []
    num_params = st.number_input("Number of Parameters", min_value=0, max_value=5, value=1)
    
    for i in range(num_params):
        st.markdown(f"**Parameter {i+1}**")
        param_name = st.text_input(f"Parameter Name", key=f"param_name_{i}")
        param_value = st.number_input(f"Value", key=f"param_value_{i}")
        param_unit = st.text_input(f"Unit", key=f"param_unit_{i}")
        param_min = st.number_input(f"Minimum Value", key=f"param_min_{i}")
        param_max = st.number_input(f"Maximum Value", key=f"param_max_{i}")
        
        if param_name:
            parameters.append(ProcessParameter(
                name=param_name,
                value=param_value,
                unit=param_unit,
                min_value=param_min,
                max_value=param_max
            ))
    
    # Get existing step IDs for predecessor selection
    existing_steps = [step.step_id for step in st.session_state.get('current_steps', [])]
    predecessor_steps = st.multiselect(
        "Predecessor Steps",
        options=existing_steps
    )
    
    if st.button("Add Step"):
        return ProcessStep(
            step_id=step_id,
            name=name,
            description=description,
            required_capabilities=required_capabilities,
            parameters=parameters,
            load=ProcessLoad(load),
            estimated_duration=timedelta(minutes=duration_minutes),
            predecessor_steps=predecessor_steps
        )
    return None

def display_robot_state(robot):
    """Display robot state information using Plotly gauges"""
    fig = go.Figure()

    # Temperature gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=robot.state.temperature[0],
        domain={'x': [0, 0.3], 'y': [0, 1]},
        title={'text': "Temperature"},
        gauge={
            'axis': {'range': [robot.state.temperature[1], robot.state.temperature[2]]},
            'bar': {'color': "darkblue"},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': robot.state.temperature[2] * 0.95
            }
        }
    ))

    # Pressure gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=robot.state.pressure[0],
        domain={'x': [0.35, 0.65], 'y': [0, 1]},
        title={'text': "Pressure"},
        gauge={
            'axis': {'range': [robot.state.pressure[1], robot.state.pressure[2]]},
            'bar': {'color': "darkblue"},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': robot.state.pressure[2] * 0.95
            }
        }
    ))

    # Humidity gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=robot.state.humidity[0],
        domain={'x': [0.7, 1], 'y': [0, 1]},
        title={'text': "Humidity"},
        gauge={
            'axis': {'range': [robot.state.humidity[1], robot.state.humidity[2]]},
            'bar': {'color': "darkblue"},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': robot.state.humidity[2] * 0.95
            }
        }
    ))

    fig.update_layout(height=200)
    st.plotly_chart(fig, use_container_width=True)

def blade_anomaly_tab():
    st.header("Blade Anomaly Detection")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Status")
        status = st.session_state.blade_monitor.get_monitoring_status()
        
        # Create a styled status display
        status_metrics = {
            "Model Status": "Trained ✅" if status['is_trained'] else "Not Trained ❌",
            "Monitoring": "Active ✅" if status['is_monitoring'] else "Inactive ❌",
            "Detected Anomalies": status['anomaly_count'],
            "Latest Error": f"{status['latest_error']:.4f}" if status['latest_error'] > 0 else "N/A"
        }
        
        for metric, value in status_metrics.items():
            st.metric(metric, value)
    
    with col2:
        st.subheader("Controls")
        
        # Model training
        if st.button("Train Model", key="blade_train"):
            with st.spinner("Training model with synthetic data..."):
                n_samples = 1000
                training_data = np.array([
                    [np.random.normal(0.120634, 0.607871) for _ in range(n_samples)],
                    [np.random.normal(-0.000055, 0.121212) for _ in range(n_samples)],
                    [np.random.normal(3.371415e+08, 5.466868e+08) for _ in range(n_samples)],
                    [np.random.normal(1945.794809, 4873.922235) for _ in range(n_samples)],
                    [np.random.normal(1.488169e+08, 2.711355e+08) for _ in range(n_samples)],
                    [np.random.normal(5367.031778, 3382.193664) for _ in range(n_samples)],
                    [np.random.normal(0.984759, 0.343720) for _ in range(n_samples)],
                    [np.random.normal(1927.328330, 655.904709) for _ in range(n_samples)]
                ]).T
                
                history = st.session_state.blade_monitor.train_model(training_data, epochs=5)
                st.success("Model trained successfully!")
        
        # Monitoring controls
        if not status['is_monitoring']:
            if st.button("Start Monitoring", key="blade_start"):
                st.session_state.blade_monitor.start_monitoring()
                st.rerun()
        else:
            if st.button("Stop Monitoring", key="blade_stop"):
                st.session_state.blade_monitor.stop_monitoring()
                st.rerun()
        
        # Simulate data button
        if st.button("Simulate New Reading", key="blade_simulate"):
            sensor_data = {
                'pCut::Motor_Torque': np.random.normal(0.120634, 0.607871),
                'pCut::CTRL_Position_controller::Lag_error': np.random.normal(-0.000055, 0.121212),
                'pCut::CTRL_Position_controller::Actual_position': np.random.normal(3.371415e+08, 5.466868e+08),
                'pCut::CTRL_Position_controller::Actual_speed': np.random.normal(1945.794809, 4873.922235),
                'pSvolFilm::CTRL_Position_controller::Actual_position': np.random.normal(1.488169e+08, 2.711355e+08),
                'pSvolFilm::CTRL_Position_controller::Actual_speed': np.random.normal(5367.031778, 3382.193664),
                'pSvolFilm::CTRL_Position_controller::Lag_error': np.random.normal(0.984759, 0.343720),
                'pSpintor::VAX_speed': np.random.normal(1927.328330, 655.904709)
            }
            
            is_anomaly, error = st.session_state.blade_monitor.process_sensor_data(
                sensor_data, return_prediction=True
            )
            
            if is_anomaly:
                st.error(f"⚠️ Anomaly detected! Error: {error:.4f}")
            else:
                st.success(f"✅ Normal operation (Error: {error:.4f})")
    
    # Display sensor readings
    st.subheader("Current Sensor Readings")
    current_state = status['current_state']
    
    if any(v is not None for v in current_state.values()):
        df = pd.DataFrame([current_state])
        
        # Create a styled dataframe
        st.dataframe(
            df.style.format("{:.4f}")
               .background_gradient(cmap='coolwarm')
        )
    else:
        st.info("No sensor readings available yet.")
    
    # Display anomaly history with plotly chart
    if st.session_state.blade_monitor.anomaly_history:
        st.subheader("Anomaly History")
        
        # Create anomaly history dataframe
        anomaly_df = pd.DataFrame([
            {
                'Timestamp': a['timestamp'],
                'Error': a['error'],
                **{k: v for k, v in a['state'].items()}
            }
            for a in st.session_state.blade_monitor.anomaly_history
        ])
        
        # Create error trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=anomaly_df['Timestamp'],
            y=anomaly_df['Error'],
            mode='lines+markers',
            name='Error',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Anomaly Error Trend",
            xaxis_title="Time",
            yaxis_title="Error",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display anomaly table
        st.dataframe(
            anomaly_df.style.format({
                'Error': "{:.4f}",
                **{col: "{:.4f}" for col in st.session_state.blade_monitor.columns}
            })
        )

def main():
    st.title("Robot Process Management System")
    
    # Sidebar for process creation and robot selection
    with st.sidebar:
        st.header("Process Configuration")
        
        # Robot Selection
        st.session_state.selected_robot = st.selectbox(
            "Select Robot Type",
            options=list(st.session_state.robots.keys())
        )
        
        # Display selected robot specifications
        if st.session_state.selected_robot:
            robot = st.session_state.robots[st.session_state.selected_robot]
            st.subheader("Robot Specifications")
            specs_df = pd.DataFrame({
                "Specification": [
                    "Payload Capacity",
                    "Arm Length",
                    "Reach",
                    "Speed",
                    "Accuracy",
                    "Power Consumption"
                ],
                "Value": [
                    f"{robot.robot_specs.payload_capacity} kg",
                    f"{robot.robot_specs.arm_length} m",
                    f"{robot.robot_specs.reach} m",
                    f"{robot.robot_specs.speed} m/s",
                    f"{robot.robot_specs.accuracy} mm",
                    f"{robot.robot_specs.power_consumption} kW"
                ]
            })
            st.dataframe(specs_df)
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Process Creation", "Process Monitoring", "Blade Anomaly"])

    # Main content area
    tab1, tab2 = st.tabs(["Process Creation", "Process Monitoring"])
    
    with tab1:
        st.header("Create New Process")
        
        # Process basic information
        process_id = st.text_input("Process ID", f"PROC-{len(st.session_state.processes) + 1}")
        process_name = st.text_input("Process Name")
        process_description = st.text_area("Process Description")
        
        # Process steps
        st.subheader("Process Steps")
        if 'current_steps' not in st.session_state:
            st.session_state.current_steps = []
            
        # Add new step
        new_step = create_process_step()
        if new_step:
            st.session_state.current_steps.append(new_step)
            st.success(f"Step {new_step.name} added successfully!")
            
        # Display current steps
        if st.session_state.current_steps:
            st.subheader("Current Steps")
            for idx, step in enumerate(st.session_state.current_steps):
                with st.expander(f"Step {idx + 1}: {step.name}"):
                    st.write(f"ID: {step.step_id}")
                    st.write(f"Description: {step.description}")
                    st.write(f"Load: {step.load}")
                    st.write(f"Duration: {step.estimated_duration}")
                    if st.button(f"Remove Step {step.step_id}"):
                        st.session_state.current_steps.pop(idx)
                        st.rerun()
        
        # Create process
        if st.button("Create Process"):
            if st.session_state.current_steps:
                new_process = Process(
                    process_id=process_id,
                    name=process_name,
                    description=process_description,
                    steps=st.session_state.current_steps,
                    required_robot_type=st.session_state.robots[st.session_state.selected_robot]
                )
                st.session_state.processes.append(new_process)
                st.session_state.current_steps = []
                st.success("Process created successfully!")
                st.rerun()
            else:
                st.error("Please add at least one step to the process.")
    
    with tab2:
        st.header("Process Monitoring")
        
        # Display processes
        for process in st.session_state.processes:
            with st.expander(f"Process: {process.name} ({process.process_id})"):
                st.write(f"Status: {process.status}")
                st.write(f"Robot Type: {process.required_robot_type.robot_type}")
                
                # Display process steps
                steps_df = pd.DataFrame([{
                    "Step": step.name,
                    "Status": step.status,
                    "Load": step.load,
                    "Duration": step.estimated_duration
                } for step in process.steps])
                st.dataframe(steps_df)
                
                # Process execution
                if process.status == ProcessStatus.NOT_STARTED:
                    if st.button(f"Execute Process {process.process_id}"):
                        state_manager = RobotStateManager()
                        try:
                            process.execute_process(state_manager)
                            st.success(f"Process {process.name} completed successfully!")
                        except Exception as e:
                            st.error(f"Process execution failed: {str(e)}")
                
                # Display robot state
                st.subheader("Robot State")
                display_robot_state(process.required_robot_type)

    with tab3:
        blade_anomaly_tab()


if __name__ == "__main__":
    main()