import subprocess
import concurrent.futures
import numpy as np

def run_function(x):
    # Run worker.py as a completely separate process
    process = subprocess.Popen(
        ["python", "tests/worker.py", str(x)],  # Run Python script with x as argument
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.PIPE   # Capture standard error
    )
    stdout, stderr = process.communicate()  # Wait for process to finish
    return(x)

if __name__=='__main__':

    inputs = list(np.arange(300,dtype=np.int32))

    # Limit number of parallel processes
    MAX_PROCESSES = 3  # Set this to the number of desired concurrent subprocesses

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
        results = list(executor.map(run_function, inputs))

    print("Results:", results)
