## Description
Implementation of Relief-based feature selection algorithm parallelized with MPI. The Relief algorithm randomly selects data instances for feature weight calculation and updates the weights based on the instance’s significance, determined by the its distance to the nearest instances of the same and opposite classes.

## Installation and Usage
```
conda create -n [env-name] python=3.7 
conda activate [env-name]
cd [project-folder]
pip install -r requirements.txt
```

Start the parallel program with mpiexec with the number of processes and the path to the input data file as arguments. The program creates n-1 slave processes:

```
mpiexec -n [number of processes] python run.py −−i [path to input file]
```

The source code includes functions to read both text and tsv files, please refer to the testcases to see the expected input format:
```
mpiexec -n 4 python run.py −−i testcases/input0.txt 
mpiexec -n 4 python run.py −−i testcases/input0.tsv
```

An example output with is given below. Slave processes printing their results in no particular order, the master process merges the output of the slave processes and prints the final result:
```
SlaveP3: 12
SlaveP1: 23
SlaveP2: 13 
MasterP0: 123
```
