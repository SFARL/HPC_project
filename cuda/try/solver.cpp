#include <stdio.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <algorithm>
#include <curand.h>
#include "gpu.cu"
const int bfs_size = 128;
using namespace std;

// Function to prlong the Sudoku grid
void read_grid(int *grid, int n, string filename)
{
    ifstream file(filename);
    if (file.is_open())
    {
        for (int i = 0; i < n * n; i++)
        {
            file >> grid[i];
        }
        file.close();
    }
    else
    {
        cout << "Error: Unable to open input file." << endl;
    }
}

// Function to prlong the Sudoku grid
void print_grid(int *grid, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        cout << std::setw(2) << grid[i] << " ";
        if ((i + 1) % n == 0)
            cout << endl;
    }
}
int main(int argc, char **argv)
{
    if (argc < 6)
    {
        printf("Usage: ./sequential <int:n> <string:input_file> <string:answer_file> <int:gridDim> <int:blockDim> \n");
        abort();
    }
    int root = atoi(argv[1]);
    string input_filename = argv[2];
    string answer_filename = argv[3];
    int gridDim = atoi(argv[4]);
    int blockDim = atoi(argv[5]);
    int n = root * root;
    int *grid = (int *)malloc(n * n * sizeof(int));
    read_grid(grid, n, input_filename);
    cout << "Input grid: " << endl;
    print_grid(grid, n);
    // Prepare for the gpu parallel
    int *new_grid, *old_grid;
    int *empty_space, *empty_cnt;
    int *grid_index;
    int grid_total = 1;

    int bfs_array_size = bfs_size * n * n * sizeof(int); // initial bfs search 128 grid

    // cuda allocate memory
    cudaMalloc(&new_grid, bfs_array_size);
    cudaMalloc(&old_grid, bfs_array_size);
    cudaMalloc(&empty_space, bfs_array_size);
    cudaMalloc(&empty_cnt, bfs_array_size / n / n + 1);
    cudaMalloc(&grid_index, sizeof(int));

    // initialize the grid to 0
    cudaMemset(grid_index, 0, sizeof(int));
    cudaMemset(new_grid, 0, bfs_array_size);
    cudaMemset(old_grid, 0, bfs_array_size);

    cudaMemcpy(old_grid, grid, n * n * sizeof(int), cudaMemcpyHostToDevice);

    cudaBFS<<<gridDim, blockDim>>>(old_grid, new_grid, grid_total, grid_index, empty_space, empty_cnt, n);

    int *tempPointer = old_grid;
    while (grid_total < bfs_size)
    {
        cudaMemcpy(&grid_total, grid_index, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(grid_index, 0, sizeof(int));

        callCudaBFS(gridDim, blockDim, old_grid, new_grid, grid_total, grid_index, empty_space, empty_cnt, n);

        tempPointer = old_grid;
        old_grid = new_grid;
        new_grid = tempPointer;
    }
    cudaMemcpy(&grid_total, grid_index, sizeof(int), cudaMemcpyDeviceToHost);

    bool *finished;
    int *solved;
    cudaMalloc(&solved, n * n * sizeof(int));
    cudaMalloc(&finished, sizeof(bool));

    cudaMemcpy(solved, old_grid, n * n * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemset(finished, false, sizeof(bool));

    callCudaBackTrack(gridDim, blockDim, old_grid, grid_total, empty_space, empty_cnt, finished, solved, n);
    cudaStreamSynchronize(0);
    print_grid(solved, n);

    delete[] grid;
    delete[] solved;
    cudaFree(new_grid);
    cudaFree(old_grid);
    cudaFree(empty_space);
    cudaFree(empty_cnt);
    cudaFree(grid_index);
    cudaFree(finished);
    cudaFree(solved);
    return 0;
}