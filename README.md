# Sudoku solvers.

Including sequential and parallel version.

# Compile and run the code

## Sequential

```shell
cd sequential

module load openmpi/intel/4.1.1

make

./sequential_avg
```
## Openmp
```shell
cd openmp

make

./openmp <num of threads> <rate of mask>
./opemmp 16 0.6
```

## Cuda

```shell
cd cuda

module load openmpi/intel/4.1.1

make

./sequential_avg
```
