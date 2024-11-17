# Value Stream Mapping for OEM Manufacturing Processes
## Setup
1. `pip install -r requirements.txt`
2. pull the data folder from thi google drive and place it in ./datasets
3. pull the remaining fours file from this link and place them in ./models/
4. run `streamlit run main.py` for hoting a local server
5. a sample deployment can be found at : <coming soon>
## Methodology
### OEM Models
- We use a dynamic approach towards adding individual and mututally dependent projects simulating a manufacturing process. 
- The robots and processes base classes can be extended to add more complex processes and robots. This will be made available to the frontend in the future.
### Value Stream Mapping
- Each Robot has several parameters and characteristic values attached to it that can be modified by the process to an extent depending on the load of the particular process.
- The process is a simple linear function that can be modified to a more complex function in the future.
