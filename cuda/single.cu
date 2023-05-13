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
#include <string>
#include <string.h>
using namespace std;

const int bfs_size = 128;
// nvcc single.cu -o single.o
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
__global__ void
cudaBFS(int *old_grid, int *new_grid, int grid_total, int *grid_index, int *empty_space, int *empty_cnt, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < grid_total)
    {
        bool found = false;
        int start = index * n * n;
        // from the start of the grid to find next avaiable space
        for (int i = start; i < (start + n * n) && !found; i++)
        {
            if (old_grid[i] == 1)
            {
                continue;
            }
            found = true;
            int row = (i - start) / n;
            int col = (i - start) % n;

            // check avaible number
            for (int ava = 1; ava <= n; ava++)
            {

                // check row and col
                for (int j = 0; j < n; j++)
                {
                    if (old_grid[start + row * n + j] == ava)
                    {
                        break;
                    }
                    if (old_grid[start + j * n + col] == ava)
                    {
                        break;
                    }
                }

                // check board
                for (int j = 0; j < sqrtf(n); j++)
                {
                    for (int k = 0; k < sqrtf(n); k++)
                    {
                        if (old_grid[start + (row / n + j) * n + (col / n + k)] == ava)
                        {
                            break;
                        }
                    }
                }

                // if avaible, add to new grid
                int next_grid_index = atomicAdd(grid_index, 1);
                if (next_grid_index >= bfs_size)
                {
                    return;
                }
                int empty_index = 0;
                // This step maybe refact to using old gird index information
                for (int r = 0; r < n; row++)
                {
                    for (int c = 0; c < n; col++)
                    {
                        new_grid[next_grid_index * n * n + r * n + c] = old_grid[index * n * n + r * n + c];

                        // calculate the new grid empty space.
                        if (old_grid[index * n * n + r * n + c] == 0 && (r != row || c != col))
                        {
                            empty_space[next_grid_index * n * n + empty_index] = r * n + c;
                            empty_index++;
                        }
                    }
                }
                empty_cnt[next_grid_index] = empty_index;
                new_grid[next_grid_index * n * n + row * n + col] = ava;
            }
        }
        index += blockDim.x * gridDim.x;
    }
}

__device__ bool is_safe(int *grid, int n, int index)
{
    int row = index / n;
    int col = index % n;
    int num = grid[index];
    // check row and col
    for (int i = 0; i < n; i++)
    {
        if (grid[row * n + i] == num && i != col)
        {
            return false;
        }
        if (grid[i * n + col] == num && i != row)
        {
            return false;
        }
    }

    // check board
    for (int i = 0; i < sqrtf(n); i++)
    {
        for (int j = 0; j < sqrtf(n); j++)
        {
            if (grid[(row / n + i) * n + (col / n + j)] == num && (row / n + i) * n + (col / n + j) != index)
            {
                return false;
            }
        }
    }
    return true;
}

__global__ void cudaBackTrack(int *grid, int grid_total, int *emptySpaces, int *empty_cnt, bool *finished, int *solved, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int *grid_now, *empty_now, empty_cnt_now;
    while (!(*finished) && index < grid_total)
    {
        int empty_index = 0;
        grid_now = grid + index * n * n;
        empty_now = emptySpaces + index * n * n;
        empty_cnt_now = empty_cnt[index];
        while (empty_index < empty_cnt_now)
        {
            grid_now[empty_now[empty_index]]++;
            if (is_safe(grid_now, n, empty_now[empty_index]))
            {
                empty_index++;
            }
            else
            {
                if (grid_now[empty_now[empty_index]] == n)
                {
                    grid_now[empty_now[empty_index]] = 0;
                    empty_index--;
                }
            }
        }
        if (empty_index == empty_cnt_now)
        {
            *finished = true;
            for (int i = 0; i < n * n; i++)
            {
                solved[i] = grid_now[i];
            }
        }
        index += blockDim.x * gridDim.x;
    }
}
void callCudaBFS(int gridDim, int blockDim, int *old_grid, int *new_grid, int grid_total, int *grid_index, int *empty_space, int *empty_cnt, int n)
{
    cudaBFS<<<gridDim, blockDim>>>(old_grid, new_grid, grid_total, grid_index, empty_space, empty_cnt, n);
}

void callCudaBackTrack(int gridDim, int blockDim, int *grid, int grid_total, int *emptySpaces, int *empty_cnt, bool *finished, int *solved, int n)
{
    cudaBackTrack<<<gridDim, blockDim>>>(grid, grid_total, emptySpaces, empty_cnt, finished, solved, n);
}
int main()
{
    // if (argc < 6)
    // {
    //     printf("Usage: ./sequential <int:n> <string:input_file> <string:answer_file> <int:gridDim> <int:blockDim> \n");
    //     abort();
    // }
    // int root = atoi(argv[1]);
    // string input_filename = argv[2];
    // string answer_filename = argv[3];
    // int gridDim = atoi(argv[4]);
    // int blockDim = atoi(argv[5]);
    int root = 3;
    string input_filename = "../9.txt";
    string answer_filename = "../9_answer.txt";
    int gridDim = 4;
    int blockDim = 4;
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