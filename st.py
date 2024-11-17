import streamlit as st
import pandas as pd
import graphviz
from datetime import datetime

# Initialize session state for processes if it doesn't exist
if 'processes' not in st.session_state:
    st.session_state.processes = pd.DataFrame(columns=['id', 'name', 'cycle_time', 'wait_time', 'created_at'])

def add_process(name, cycle_time, wait_time):
    new_process = pd.DataFrame({
        'id': [datetime.now().strftime('%Y%m%d%H%M%S')],
        'name': [name],
        'cycle_time': [float(cycle_time)],
        'wait_time': [float(wait_time)],
        'created_at': [datetime.now()]
    })
    st.session_state.processes = pd.concat([st.session_state.processes, new_process], ignore_index=True)

def delete_process(process_id):
    st.session_state.processes = st.session_state.processes[st.session_state.processes['id'] != process_id]

def create_vsm_graph():
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR')  # Left to Right direction
    
    # Add nodes and edges for each process
    for idx, process in st.session_state.processes.iterrows():
        node_label = f"""<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
            <TR><TD COLSPAN="2">{process['name']}</TD></TR>
            <TR><TD>Cycle Time</TD><TD>{process['cycle_time']} min</TD></TR>
            <TR><TD>Wait Time</TD><TD>{process['wait_time']} min</TD></TR>
        </TABLE>>"""
        
        graph.node(str(process['id']), node_label, shape='none')
        
        # Add edge to next process if it exists
        if idx < len(st.session_state.processes) - 1:
            next_process = st.session_state.processes.iloc[idx + 1]
            graph.edge(str(process['id']), str(next_process['id']))
    
    return graph

# Page title
st.title('Value Stream Mapping')

# Sidebar for adding new processes
with st.sidebar:
    st.header('Add New Process')
    process_name = st.text_input('Process Name')
    cycle_time = st.number_input('Cycle Time (min)', min_value=0.0, step=0.1)
    wait_time = st.number_input('Wait Time (min)', min_value=0.0, step=0.1)
    
    if st.button('Add Process'):
        if process_name:
            add_process(process_name, cycle_time, wait_time)
            st.success(f'Added process: {process_name}')
        else:
            st.error('Please enter a process name')

# Main content area
if not st.session_state.processes.empty:
    # Display VSM diagram
    graph = create_vsm_graph()
    st.graphviz_chart(graph)
    
    # Display process table with delete buttons
    st.header('Process List')
    for idx, process in st.session_state.processes.iterrows():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            st.write(process['name'])
        with col2:
            st.write(f"CT: {process['cycle_time']} min")
        with col3:
            st.write(f"WT: {process['wait_time']} min")
        with col4:
            if st.button('Delete', key=process['id']):
                delete_process(process['id'])
                st.experimental_rerun()

    # Display summary metrics
    st.header('Summary Metrics')
    total_cycle_time = st.session_state.processes['cycle_time'].sum()
    total_wait_time = st.session_state.processes['wait_time'].sum()
    total_lead_time = total_cycle_time + total_wait_time
    process_cycle_efficiency = (total_cycle_time / total_lead_time * 100) if total_lead_time > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Cycle Time', f"{total_cycle_time:.1f} min")
    col2.metric('Total Wait Time', f"{total_wait_time:.1f} min")
    col3.metric('Total Lead Time', f"{total_lead_time:.1f} min")
    col4.metric('Process Cycle Efficiency', f"{process_cycle_efficiency:.1f}%")
else:
    st.info('Add processes using the sidebar to create your Value Stream Map')

# Optional: Add download button for the process data
if not st.session_state.processes.empty:
    st.download_button(
        label="Download Process Data",
        data=st.session_state.processes.to_csv(index=False),
        file_name="vsm_processes.csv",
        mime="text/csv"
    )