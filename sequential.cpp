#include <stdio.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>

// sequential backtrack sudoku solution
// Sample run ./sequential 4
// Sample test ./sequential 9 ./9.txt ./9_ans.txt
// Sample test ./sequential 16 ./16.txt ./16_ans.txt
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

// Function to check the Sudoku solve grid is correct or not
bool check_grid(int *grid, int n, int *answer)
{
    for (int i = 0; i < n * n; i++)
    {
        if (grid[i] != answer[i])
        {
            return false;
            break;
        }
    }
    return true;
}

// Function to check if a number is safe to be placed in a cell
bool is_safe(int *grid, int n, int row, int col, int num)
{
    int idx;
    // Check if the number is already in the row
    for (int i = 0; i < n; i++)
    {
        idx = row * n + i;
        if (grid[idx] == num)
        {
            return false;
        }
    }
    // Check if the number is already in the column, can be optimized by data locality when size increase
    for (int i = 0; i < n; i++)
    {
        idx = i * n + col;
        if (grid[idx] == num)
        {
            return false;
        }
    }

    // Check if the number is already in the sqrt(n) box
    int root = sqrt(n);
    int box_row = row - row % root;
    int box_col = col - col % root;
    for (int i = box_row; i < box_row + root; i++)
    {
        for (int j = box_col; j < box_col + root; j++)
        {
            idx = i * n + j;
            if (grid[idx] == num)
            {
                return false;
            }
        }
    }
    return true;
}

// Function to solve the Sudoku puzzle using backtracking
bool solve_sudoku(int *grid, int n, int row, int col)
{
    // cout << "row: " << row << " col: " << col << endl;
    // If all rows and columns are processed, the puzzle is solved
    if (row == n - 1 && col == n)
    {
        return true;
    }
    // If all columns of a row are processed, move to the next row
    if (col == n)
    {
        row++;
        col = 0;
    }

    int idx = row * n + col;
    // If the current cell is not empty, skip to the next cell
    if (grid[idx] != 0)
    {
        return solve_sudoku(grid, n, row, col + 1);
    }
    // Try to place a number in the current cell
    for (int num = n; num > 0; num--)
    {
        if (is_safe(grid, n, row, col, num))
        {
            // cout << "row: " << row << " col: " << col << " num: " << num << endl;
            // print_grid(grid, n);
            // cout << endl;
            grid[idx] = num;
            if (solve_sudoku(grid, n, row, col + 1))
            {
                return true;
            }
            grid[idx] = 0; // backtrack
        }
    }
    return false;
}

void build_sudoku(int *grid, int n)
{
    std::mt19937 rng(42);
    int root = sqrt(n);
    for (int i = 0; i < n; i += root)
    {
        int num = 1;
        for (int row = 0; row < root; row++)
        {
            for (int col = 0; col < root; col++)
            {
                while (!is_safe(grid, n, row + i, col + i, num))
                {
                    num = std::uniform_int_distribution<int>{1, n}(rng);
                }
                grid[(row + i) * n + col + i] = num;
            }
        }
    }
    solve_sudoku(grid, n, 0, 0);
}

// Remove numbers from the grid while ensuring the puzzle remains solvable with unique solution
void remove_unique(int *grid, int n, int cnt)
{
    int idx;
    int row;
    int col;
    int solution;
    srand(time(NULL));
    while (cnt > 0)
    {
        col = rand() % n;
        row = rand() % n;
        idx = row * n + col;
        cnt--;
        solution = 0;
        int temp = grid[idx];
        if (temp != 0)
        {
            grid[idx] = 0;
            for (int num = 1; num <= n; num++)
            {
                if (is_safe(grid, n, row, col, num) && solve_sudoku(grid, n, row, col))
                {
                    solution++;
                }
                if (solution >= 2)
                {
                    grid[idx] = temp;
                    cnt--;
                    break;
                }
            }
        }
    }
}

void remove_standard(int *grid, int n, int cnt)
{
    int idx;
    int row;
    int col;
    int solution;
    srand(time(NULL));
    while (cnt > 0)
    {
        col = rand() % n;
        row = rand() % n;
        idx = row * n + col;

        int temp = grid[idx];
        if (temp != 0)
        {
            grid[idx] = 0;
            cnt--;
        }
    }
}

int main(int argc, char **argv)
// int main_standard(int argc, char **argv)
{
    srand(4);
    if (argc < 2)
    {
        printf("Usage: ./sequential <int:root> \n");
        abort();
    }
    int root = atoi(argv[1]);
    int n = root * root;
    auto t_start = std::chrono::high_resolution_clock::now();
    int *grid = (int *)malloc(n * n * sizeof(int));
    // int *ans = (int *)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; i++)
    {
        grid[i] = 0;
    }
    // int rand_num = rand() % n;
    // for (int i = 0; i < n; i++)
    // {
    //     grid[i * n + i] = (rand_num + i) % n + 1;
    //     cout << "i: " << i << " value: " << grid[i * n + i] << endl;
    // }
    build_sudoku(grid, n);
    // grid[0] = rand() % n + 1;
    // solve_sudoku(grid, n, 0, 0);
    // for (int i = 0; i < n * n; i++)
    // {
    //     ans[i] = grid[i];
    // }
    int rm_cnt = n * n * 0.3;
    remove_standard(grid, n, rm_cnt);
    cout << "Input grid: " << endl;
    print_grid(grid, n);
    // cout << "Answer grid: " << endl;
    // print_grid(ans, n);
    auto t_end = std::chrono::high_resolution_clock::now();
    cout << "Build time cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9 << " s" << endl;

    t_start = std::chrono::high_resolution_clock::now();
    if (solve_sudoku(grid, n, 0, 0))
    {
        // if (check_grid(grid, n, ans))
        printf("Solved grid! \n");
        // else
        //     printf("Solved grid: Wrong! \n");
        // free(ans);
    }
    else
    {
        cout << "No solution exists or can't find the solution" << endl;
    }
    print_grid(grid, n);
    t_end = std::chrono::high_resolution_clock::now();
    cout << "Solve time cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9 << " s" << endl;
    free(grid);
    return 0;
}

int main_unqiue(int argc, char **argv)
{
    srand(42);
    if (argc < 2)
    {
        printf("Usage: ./sequential <int:root> \n");
        abort();
    }
    int root = atoi(argv[1]);
    int n = root * root;
    auto t_start = std::chrono::high_resolution_clock::now();
    int *grid = (int *)malloc(n * n * sizeof(int));
    int *ans = (int *)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; i++)
    {
        grid[i] = 0;
    }
    build_sudoku(grid, n);
    int rm_cnt = n * n * 0.5;
    remove_unique(grid, n, rm_cnt);
    cout << "Input grid: " << endl;
    print_grid(grid, n);
    cout << "Answer grid: " << endl;
    print_grid(ans, n);
    auto t_end = std::chrono::high_resolution_clock::now();
    cout << "Build time cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9 << " s" << endl;

    t_start = std::chrono::high_resolution_clock::now();
    if (solve_sudoku(grid, n, 0, 0))
    {
        if (check_grid(grid, n, ans))
            printf("Solved grid: Correct! \n");
        else
            printf("Solved grid: Wrong! \n");
        free(ans);
    }
    else
    {
        cout << "No solution exists or can't find the solution" << endl;
    }
    print_grid(grid, n);
    t_end = std::chrono::high_resolution_clock::now();
    cout << "Solve time cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9 << " s" << endl;
    free(grid);
    return 0;
}

int main_test(int argc, char **argv)
{
    srand(42);
    if (argc < 4)
    {
        printf("Usage: ./sequential <int:n> <string:input_file> <string:answer_file> \n");
        abort();
    }
    int n = atoi(argv[1]);
    if (sqrt(n) * sqrt(n) != n)
    {
        printf("Error: n must be a perfect square. \n");
        abort();
    }
    string input_filename = argv[2];
    string answer_filename = argv[3];

    int *grid = (int *)malloc(n * n * sizeof(int));
    read_grid(grid, n, input_filename);

    cout << "Input grid: " << endl;
    print_grid(grid, n);
    auto t_start = std::chrono::high_resolution_clock::now();
    if (solve_sudoku(grid, n, 0, 0))
    {
        int *ans = (int *)malloc(n * n * sizeof(int));
        read_grid(ans, n, answer_filename);
        if (check_grid(grid, n, ans))
            printf("Solved grid: Correct! \n");
        else
            printf("Solved grid: Wrong! \n");
        free(ans);
    }
    else
    {
        cout << "No solution exists or can't find the solution" << endl;
    }
    print_grid(grid, n);
    auto t_end = std::chrono::high_resolution_clock::now();
    cout << "Solve time cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9 << " s" << endl;
    free(grid);
    return 0;
}
