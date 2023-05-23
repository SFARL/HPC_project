# Sudoku solvers.

Including sequential and parallel version.

# Compile and run the code

## Sequential - on CIMS

```shell
cd sequential

module load gcc-9.2

make

./sequential_avg
```
## Openmp - on CIMS
```shell
cd openmp

module load gcc-9.2

make

./openmp <num of threads> <rate of mask>
./opemmp 16 0.6
```

## Cuda - on Greene

```shell
cd cuda

module load openmpi/intel/4.1.1

make

./sequential_avg
```
