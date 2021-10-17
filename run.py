"""
Student Name: Alara Dirik
Student Number: 2020800006
Compile Status: NA - No need for compilation
Program Status: Working
Notes: 
Python 3.7
run with mpiexec -n [num process] python run.py --i [path to input file]
"""

from mpi4py import MPI
import numpy as np

# Import relief and utility functions
from relief import *


# Initialize network
comm = MPI.COMM_WORLD   
size = comm.size        
rank = comm.rank        
status = MPI.Status()   

# Master process
if rank == 0:
    # Read input data
    args = vars(get_input_args())
    data, config = read_input(args["path"])
    
    # Number of slaves and data partitions
    num_slaves = size - 1
    subset_size = data.shape[0] // num_slaves
    num_subsets = range(data.shape[0] // subset_size)

    # Number of completed tasks
    completed = 0
    
    # Global list of top features
    top_ts = []
    
    # Wait to receive data from slaves until all subsets are processed
    while completed < num_slaves:

        # Receive data from slaves
        local_data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        
        # Slave process is ready to perform task
        if tag == 0:               
            # Start and end of the data partition
            start = (source - 1) * subset_size
            end = source * subset_size
            
            # Append m and t to data partition
            params = np.zeros((1, data.shape[1]))
            params[0, -2:] = config[3:]
            subset = data[start: end, :]
            subset = np.vstack((subset, params))
            
            # Send corresponding subset to slave process
            comm.send(subset, dest=source, tag=1)
        
        # Received data is top t features
        elif tag == 2:
            top_ts.append(local_data)
            completed += 1

    # Concat local top t features
    top_ts = np.concatenate(top_ts)
    # Remove duplicates and sort
    top_ts = np.sort(np.unique(top_ts))
    # Print result
    top_ts = " ".join([str(x) for x in top_ts])
    print("Master P{} : {}".format(rank, top_ts))

# Slave processes (rank > 0)
else:
    # Send master ready signal
    comm.send(None, dest=0, tag=0)

    # Receive task / subset from master
    task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
    tag = status.Get_tag()
    
    # Slave process received start tag from master
    if tag == 1:

        # Run the Relief algorithm on the subset
        result = run_relief(task)

        # Print local top t features
        top_t = " ".join([str(x) for x in list(result)])
        print("Slave P{} : {}".format(rank, top_t))

        # Send top t features to master with done tag
        comm.send(result, dest=0, tag=2)
        