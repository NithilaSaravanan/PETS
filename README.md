
# PETs
***Python Estimation Toolkits (PETs)*** is an agile and capable estimation library that was developed by Manoj Krishna Venkatesan and Nithilasaravanan Kuppan as part of their Master's thesis at McGill University, Montreal, Canada. **PETs** stands as a testament to the years of intense research conducted by Dr. Michalska and her graduate students - the authors have tried their best to include the essential parts of all this research as a single, *easy-to-use* package.

### What is PETs?
***PETs*** is a Python-based state and parameter estimation library consisting of 4 different methods to perform estimation of SISO LTI systems. A detailed account of all the methods - their working, their logic, and an incisive comparison of their performances under different conditions have been presented in the authors' thesis documents under the *resources* folder.
You can install all the required libraries by simply 
> pip install requirements.txt

 Below is a summary of the options available:

 - `kernel_projection`:  Uses the kernel method to estimate both the system parameters
which are then used by the projection method to estimate the states
 - `kalman_statesonly` : Uses Kalman + RTS filter to estimate the state of the system,
given the system parameters
 - `kalman_ukf` : Uses Unscented Kalman filter + RTS to estimate the state and the system
parameters
 - `kernel_kalman`: Uses the parameters calculated from the kernel algorithm to find the
state of the system using Kalman + RTS filter

These methods can be executed by navigating to `/PETS/scripts/`and then running
> python run_estimation.py -m *method*

Based on the *method* specified (which would be one of the 4 options given above), the script will take the input from `/PETS/src/pets/noisy_input.py` and the running parameters from `/PETS/configs/config_*algorithm*`and execute the associated script. Please refer to the thesis documents for a detailed explanation of how this works.

### What can I do with this?
With ***PETs*** the user can estimate the state and parameters of any SISO LTI system *(the repository currently supports up to 4th order systems!)* with possible future support to MIMO systems. The modularity of this package allows users to leverage specific parts of the script and amend them as per their use-case - theoretical research suggests these algorithms can be successfully used on a multitude of control-related problems. To know more about this repository, the kernel or the projection method in detail, or various use cases of the kernel-projection method, including extensions to non-linear systems, please refer to the `resources`folder. 

Happy PETs-ing!   
