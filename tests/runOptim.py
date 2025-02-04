import multiprocessing
import time

# Function to be run in parallel
def worker_function(input_value):
    print(f"Processing {input_value} on process {multiprocessing.current_process().name}")
    time.sleep(input_value)  # Simulate work
    return input_value ** 2  # Example computation

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()  # Get number of CPU cores
    num_processes = min(4, num_cores)  # Adjust number of processes (change as needed)
    
    input_values = [1, 3, 7, 15, 1, 3, 7, 15]  # Example different inputs
    
    # Use multiprocessing Pool to parallelize execution
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(worker_function, input_values)  # Distribute tasks
    
    print("Results:", results)
