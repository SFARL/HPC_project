#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "sudoku.h"
#include <string.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>




//returns a 2D array from a file containing the Sudoku in space separated format (empty cells are 0)
int ** readInput(char *filename){
	FILE *infile;
	infile = fopen(filename, "r");
	int i, j;
	char dummyline[SIZE+1];
	char dummy;  //to eliminate the eol in txt file
	int value;
	//initialising double dimensional array of n*n
	int **sudokuGrid = (int**)malloc(sizeof(int*)*SIZE);
	for (i=0;i<SIZE;i++)
	{
		sudokuGrid[i] = (int*)malloc(sizeof(int)*SIZE);
		for (j=0;j<SIZE;j++)
			sudokuGrid[i][j] = 0;
	}
	//assigning values
	for (i = 0; i < SIZE; i++)
	{
		for (j = 0; j < SIZE; j++)
		{
			/* Checking if number of rows is less than SIZE */
			if (feof(infile))
			{
				if (i != SIZE)
				{
					printf("The input puzzle has less number of rows than %d. Exiting.\n", SIZE);
					exit(-1);
				}
    		}

        		fscanf(infile, "%d", &value);
        		if(value >= 0 && value <= SIZE)//valid sudoku
				sudokuGrid[i][j] = value;
			else
			{
				printf("The input puzzle is not a grid of numbers (0 <= n <= %d) of size %dx%d. Exiting.\n", SIZE, SIZE, SIZE);
				exit(-1);
			}
		}
		fscanf(infile, "%c", &dummy); /* To remove stray \0 at the end of each line */

		/* Checking if row has more elements than SIZE */
		if (j > SIZE)
		{
			printf("Row %d has more number of elements than %d. Exiting.\n", i+1, SIZE);
			exit(-1);
		}
	}
	return sudokuGrid;
}


/*checks if solution is a valid solution to original 
i.e. all originally filled cells match, and that solution is a valid grid*/
int isValid(int **original, int **solution){
	int valuesSeen[SIZE],i,j,k;

	//check all rows
	for (i=0;i<SIZE;i++)
	{
		for (k=0;k<SIZE;k++) 
			valuesSeen[k] = 0;

		for (j=0;j<SIZE;j++)
		{
			if (solution[i][j]==0) 
				return 0;
			if ((original[i][j])&&(solution[i][j] != original[i][j])) 
				return 0;
			int v = solution[i][j];
			if (valuesSeen[v-1]==1){   //checking if values are distinct in the horizontal big lines
				return 0;
			}
			valuesSeen[v-1] = 1; 
		}
	}

	//check all columns
	for (i=0;i<SIZE;i++){
		for (k=0;k<SIZE;k++) 
			valuesSeen[k] = 0;
		for (j=0;j<SIZE;j++)
		{
			int v = solution[j][i];
			if (valuesSeen[v-1]==1)    //checking if values are distinct in the vertical big lines
			{
				return 0;
			}
			valuesSeen[v-1] = 1;
		}
	}

	//check all minigrids
	//check all rows
	for (i=0;i<SIZE;i=i+MINIGRIDSIZE)
	{
		for (j=0;j<SIZE;j=j+MINIGRIDSIZE)
		{
			for (k=0;k<SIZE;k++) 
				valuesSeen[k] = 0;

			//checking if values are distinct in each sub box
			int r,c;
			for (r=i;r<i+MINIGRIDSIZE;r++)
				for (c=j;c<j+MINIGRIDSIZE;c++)
				{
					int v = solution[r][c];
					if (valuesSeen[v-1]==1) 
					{
						return 0;
					}
					valuesSeen[v-1] = 1;
				}
		}
	}
	return 1;
}


int main(int argc, char *argv[])
{
	srand(4);
	if (argc < 2)
	{
		printf("Usage: ./sudoku <int:num_threads> <float: mask>\n");
		abort();
	}
	int num_threads = atoi(argv[1]);
	omp_set_num_threads(num_threads);
	double mask = atof(argv[2]);
	printf("mask is %f\n", mask);
	auto t_start = std::chrono::high_resolution_clock::now();
	int **grid = (int **)malloc(SIZE*sizeof(int*));
	for (int i = 0; i < SIZE; i++){
		grid[i] = (int *)malloc(SIZE*sizeof(int*));
		for (int j = 0; j < SIZE; j++){
			grid[i][j] = 0;
		}
	}
	build_sudoku(grid);
	int rm_cnt = SIZE * SIZE *mask;
	remove_standard(grid, rm_cnt);
	auto t_end = std::chrono::high_resolution_clock::now();
	std::cout<< "Build time cost:" <<std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9 << " s" <<std::endl;
	print_grid(grid);
	int **test_grid = (int **)malloc(SIZE*sizeof(int*));
    for (int i = 0; i < SIZE; i++){
        test_grid[i] = (int *)malloc(SIZE*sizeof(int*));
		for (int j = 0; j < SIZE; j++){
			test_grid[i][j] = grid[i][j];
		}
    }
	
	printf("parallel starts\n");
	double start = omp_get_wtime();//Elapsed wall clock time in seconds. The time is measured per thread, no guarantee can be made that two distinct threads measure the same time
	//double omp_get_wtime(void);
	int **outputGrid = solveSudoku(grid);
	double finish = omp_get_wtime();
	printf("************************OUTPUT GRID***********************\n");

	print_grid(outputGrid);
	printf("*********************************************************\n");
	if (isValid(grid,outputGrid)){
		printf("SOLUTION FOUND\nTIME = %lf\n",(finish-start));
	}
	else{
		printf("NO SOLUTION FOUND\nTIME =%lf\n",(finish-start));
	}
	printf("************************SEQ OUTPUT GRID***********************\n");
	auto t_start1 = std::chrono::high_resolution_clock::now();
	Sudoku_seq(test_grid, 0, 0);
	auto t_end1 = std::chrono::high_resolution_clock::now();
	print_grid(test_grid);
	std::cout << "Seq solve time cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end1 - t_start1).count() * 1e-9 << " s" << std::endl;
	free(grid);
	free(test_grid);
	return 0;
}

int main_test(int argc, char *argv[]){
	if (argc<3){
		printf("Usage: ./sudoku <thread_count> <inputFile>\n");
		exit(0);
	}
	int **originalGrid = readInput(argv[2]);
	
	int thread_count = atoi(argv[1]);
	if (thread_count<=0){
		printf("Usage: Thread Count should be positive\n");
	}
	omp_set_num_threads(thread_count);  //The omp_set_num_threads function sets the default number of threads to use for subsequent parallel regions.
	//void omp_set_num_threads(int num_threads);

	//int i,j;
	printf("************************INPUT GRID***********************\n");
	/*for (i=0;i<SIZE;i++){
		for (j=0;j<SIZE;j++){
			printf("%d ",originalGrid[i][j]);
		}
		printf("\n");
	}*/
	print_grid(originalGrid);
	int **test_grid1 = (int **)malloc(SIZE*sizeof(int*));
    for (int i = 0; i < SIZE; i++){
        test_grid1[i] = (int *)malloc(SIZE*sizeof(int*));
		for (int j = 0; j < SIZE; j++){
			test_grid1[i][j] = originalGrid[i][j];
		}
    }
	/*
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			printf("************************SEQ OUTPUT GRID***********************\n");
			auto t_start = std::chrono::high_resolution_clock::now();
			Sudoku_seq(test_grid1, 0, 0);
			auto t_end = std::chrono::high_resolution_clock::now();
			std::cout << "Seq solve time cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9 << " s" << std::endl;
			print_grid(test_grid1);
		}
		#pragma omp section
		{
			double start = omp_get_wtime();
			int **outputGrid = solveSudoku(originalGrid);
			double finish = omp_get_wtime();
			printf("************************OUTPUT GRID***********************\n");

			print_grid(outputGrid);
			printf("*********************************************************\n");
			if (isValid(originalGrid,outputGrid)){
				printf("SOLUTION FOUND\nTIME = %lf\n",(finish-start));
			}
			else{
				printf("NO SOLUTION FOUND\nTIME =%lf\n",(finish-start));
			}

		}
	}*/
	/*
	printf("************************SEQ OUTPUT GRID***********************\n");
	auto t_start = std::chrono::high_resolution_clock::now();
	Sudoku_seq(test_grid1, 0, 0);
	auto t_end = std::chrono::high_resolution_clock::now();
	print_grid(test_grid1);
	std::cout << "Seq solve time cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9 << " s" << std::endl;*/
	printf("parallel starts\n");
	double start = omp_get_wtime();//Elapsed wall clock time in seconds. The time is measured per thread, no guarantee can be made that two distinct threads measure the same time
	//double omp_get_wtime(void);
	int **outputGrid = solveSudoku(originalGrid);
	double finish = omp_get_wtime();
	printf("************************OUTPUT GRID***********************\n");

	print_grid(outputGrid);
	printf("*********************************************************\n");
	if (isValid(originalGrid,outputGrid)){
		printf("SOLUTION FOUND\nTIME = %lf\n",(finish-start));
	}
	else{
		printf("NO SOLUTION FOUND\nTIME =%lf\n",(finish-start));
	}
	return 0;
}
