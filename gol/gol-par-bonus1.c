/***********************
Author: Foivos Papapanagiotakis-Bousy (fpy700)

Conway's Game of Life - Parallel Impementation using MPI

Based on https://web.cs.dal.ca/~arc/teaching/CS4125/2014winter/Assignment2/Assignment2.html

************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <assert.h>
#include "StopWatch/StopWatch.h"


typedef struct {
   int height, width;
   int** cells;
} world;

static world worlds[2];
static world *cur_world;
static world* next_world;

static int print_cells = 0;
static int print_world = 0;

StopWatch* process_sw;
StopWatch* comms_sw;
static double elaps = 0.0;
static double total_elaps = 0.0;

MPI_Status status;

// use fixed world or random world?
#ifdef FIXED_WORLD
static int random_world = 0;
#else
static int random_world = 1;
#endif

static const int top_row_tag = 2;
static const int bottom_row_tag = 3;

static char* start_world[] = {
    /* Gosper glider gun */
    /* example from https://bitstorm.org/gameoflife/ */
    "..........................................",
    "........................OO.........OO.....",
    ".......................O.O.........OO.....",
    ".OO.......OO...........OO.................",
    ".OO......O.O..............................",
    ".........OO......OO.......................",
    ".................O.O......................",
    ".................O........................",
    "....................................OO....",
    "....................................O.O...",
    "....................................O.....",
    "..........................................",
    "..........................................",
    ".........................OOO..............",
    "..........................................",
    ".........................OOO..............",
    ".........................O................",
    "..........................O...............",
    "..........................................",
};

static void world_init_fixed(world* world) {
    int** cells = world->cells;
    int i, j;

    /* use predefined start_world */

    for (i = 1; i <= world->height; i++) {
        for (j = 1; j <= world->width; j++) {
            if ((i <= sizeof(start_world) / sizeof(char*)) &&
                (j <= strlen(start_world[i - 1]))) {
                cells[i][j] = (start_world[i - 1][j - 1] != '.');
            } else {
                cells[i][j] = 0;
            }
        }
    }
}

static void world_init_random(world* world) {
    int** cells = world->cells;
    int i, j;

    // Note that rand() implementation is platform dependent.
    // At least make it reprodible on this platform by means of srand()
    srand(1);

    for (i = 1; i <= world->height; i++) {
        for (j = 1; j <= world->width; j++) {
            float x = rand() / ((float)RAND_MAX + 1);
            if (x < 0.5) {
                cells[i][j] = 0;
            } else {
                cells[i][j] = 1;
            }
        }
    }
}

static void world_print(world* world) {
    int** cells = world->cells;
    int i, j;

    for (i = 1; i <= world->height; i++) {
        for (j = 1; j <= world->width; j++) {
            if (cells[i][j]) {
                printf("O");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
}

static int world_count(world* world) {
    int** cells = world->cells;
    int isum;
    int i, j;

    isum = 0;
    for (i = 1; i <= world->height; i++) {
        for (j = 1; j <= world->width; j++) {
            isum = isum + cells[i][j];
        }
    }

    return isum;
}

static int partial_world_count(world* world, int top_row, int bottom_row){
    int** cells = world->cells;
    int isum;
    int i, j;

    isum = 0;
    for (i = top_row; i <= bottom_row; i++) {
        for (j = 1; j <= world->width; j++) {
            isum = isum + cells[i][j];
        }
    }

    return isum;
}

/* Take world wrap-around into account: */
static void world_border_wrap(world *world) {
    int** cells = world->cells;
    int i, j;

    /* left-right boundary conditions */
    for (i = 1; i <= world->height; i++) {
        cells[i][0] = cells[i][world->width];
        cells[i][world->width + 1] = cells[i][1];
    }

    /* top-bottom boundary conditions */
    for (j = 0; j <= world->width + 1; j++) {
        cells[0][j] = cells[world->height][j];
        cells[world->height + 1][j] = cells[1][j];
    }
}

/*  This function should be called once, before the main loop. 
    This will remove the need for extra communication between the first and the last process before the computation part starts. */
static void world_top_bottom_border_wrap(world* world){
    int** cells = world->cells;
    int i;

    /* top-bottom boundary conditions */
    for (i = 0; i <= world->width + 1; i++) {
        cells[0][i] = cells[world->height][i];
        cells[world->height + 1][i] = cells[1][i];
    }
}

/*  This function is not required to wrap top-bottom boundaries since these are communicated by other processes. 
    However, the caller should wrap the communicated rows as well. */
static void world_partial_left_right_border_wrap(world* world, int start_row, int end_row){
    int** cells = world->cells;
    int i;

    /* left-right boundary conditions */
    for (i = start_row; i <= end_row; i++) {
        cells[i][0] = cells[i][world->width];
        cells[i][world->width + 1] = cells[i][1];
    }
}

static int world_cell_newstate(world* world, int row, int col) {
    int** cells = world->cells;
    int row_m, row_p, col_m, col_p, nsum;
    int newval;

    // sum surrounding cells
    row_m = row - 1;
    row_p = row + 1;
    col_m = col - 1;
    col_p = col + 1;

    nsum = cells[row_p][col_m] + cells[row_p][col] + cells[row_p][col_p]
         + cells[row  ][col_m]                     + cells[row  ][col_p]
         + cells[row_m][col_m] + cells[row_m][col] + cells[row_m][col_p];

    switch (nsum) {
    case 3:
        // a new cell is born
        newval = 1;
        break;
    case 2:
        // nothing happens
        newval = cells[row][col];
        break;
    default:
        // the cell, if any, dies
        newval = 0;
    }

    return newval;
}


// update board for next timestep
// height/width params are the base height/width
// excluding the surrounding 1-cell wraparound border
static void world_timestep(world *old, world *new) {
    int i, j;

    // update board
    for (i = 1; i <= new->height; i++) {
        for (j = 1; j <= new->width; j++) {
            new->cells[i][j] = world_cell_newstate(old, i, j);
        }
    }
}

// update board partially. [top_row, bottom_row]
static void world_partial_timestep(world *old, world *new, int top_row, int bottom_row){
    int i, j;

    for(i = top_row; i <= bottom_row; i++){
        for(j = 1; j <= new->width; j++){
            new->cells[i][j] = world_cell_newstate(old, i, j);
        }
    }
}

static int** alloc_2d_int_array(int nrows, int ncolumns) {
    int** array;
    int i;

    /* version that keeps the 2d data contiguous, can help caching and slicing across dimensions */
    array = malloc(nrows * sizeof(int*));
    if (array == NULL) {
       fprintf(stderr, "out of memory\n");
       exit(1);
    }

    array[0] = malloc(nrows * ncolumns * sizeof(int));
    if (array[0] == NULL) {
       fprintf(stderr, "out of memory\n");
       exit(1);
    }

    for (i = 1; i < nrows; i++) {
	    array[i] = array[0] + i * ncolumns;
    }

    return array;
}

static void init_mpi(int* total_processes, int* process_rank){
    assert(total_processes && process_rank);                // args are references, they cannot be null.
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, total_processes);         // init total_processes
	MPI_Comm_rank(MPI_COMM_WORLD, process_rank);            // init process_rank (different for every process)
}

static void create_worlds(int bwidth, int bheight){
    worlds[0].height = bheight;
    worlds[0].width = bwidth;
    worlds[0].cells = alloc_2d_int_array(bheight + 2, bwidth + 2);

    worlds[1].height = bheight;
    worlds[1].width = bwidth;
    worlds[1].cells = alloc_2d_int_array(bheight + 2, bwidth + 2);

    cur_world = &worlds[0];
    next_world = &worlds[1];
}

static void world_init(){                               
    if (random_world)
        world_init_random(cur_world);
    else
        world_init_fixed(cur_world);
}

/*  Function which dictates how the world (elements) will be divided across processes. 
    It initializes both arrays which are needed for the MPI_scatterv function.
    Caller receives ownership of both arrays. */
static void world_distribution_init(int total_processes, int** distribution, int** displacement){
    assert(distribution && displacement);   // distribution & displacement are references, they cannot be null
    *distribution = malloc(total_processes * sizeof(int));
    *displacement = malloc(total_processes * sizeof(int));
    int elements_per_row = cur_world->width + 2;
    int next_idx, idx = 1; // index of the first element to be sent (skipping first row)
    double idx_d = (double)idx;
    int chunk_size = (cur_world->height / total_processes);
    double chunk_size_d = (double)cur_world->height / (double)total_processes;

    
    for(int i=0; i<total_processes-1; i++){
        next_idx = idx_d > (double)idx ? idx + chunk_size + 1 : idx + chunk_size;
        (*distribution)[i] = next_idx - idx;
        (*displacement)[i] = idx;
        idx = next_idx;
        idx_d += chunk_size_d;
    }

    (*distribution)[total_processes - 1] = cur_world->height + 1 - idx;
    (*displacement)[total_processes - 1] = idx;

    // printf("Row distribution and displacement: \n");
    // for (int i=0; i < total_processes; i++)
    //     printf("process %d: %d - %d\n", i, (*distribution)[i], (*displacement)[i]);

    for(int i=0; i<total_processes; i++){
        (*distribution)[i] *= elements_per_row;
        (*displacement)[i] *= elements_per_row;
    }
}

static void world_process_row_range_init(int process_rank, int total_processes, int* top_row, int* bottom_row){
    assert(top_row && bottom_row);
    int chunk_size = cur_world->height / total_processes;
    *top_row = (chunk_size * process_rank) + 1;
    if(process_rank == total_processes - 1)
        *bottom_row = (chunk_size * (process_rank + 1)) + (cur_world->height % total_processes);
    else
        *bottom_row = (chunk_size * (process_rank + 1));
}

static inline void send_top_row(int process_rank, int total_processes){
    if(process_rank == 0)
        MPI_Ssend(cur_world->cells[1], cur_world->width + 2, MPI_INT, total_processes - 1, top_row_tag, MPI_COMM_WORLD);
    else
        MPI_Ssend(cur_world->cells[1], cur_world->width + 2, MPI_INT, process_rank - 1, top_row_tag, MPI_COMM_WORLD);
}

static inline void send_bottom_row(int process_rank, int total_processes, int bottom_row){
    if(process_rank == total_processes - 1)
        MPI_Ssend(cur_world->cells[bottom_row], cur_world->width + 2, MPI_INT, 0, bottom_row_tag, MPI_COMM_WORLD);
    else
        MPI_Ssend(cur_world->cells[bottom_row], cur_world->width + 2, MPI_INT, process_rank + 1, bottom_row_tag, MPI_COMM_WORLD);
}

static inline void receive_adjacent_top_row(int process_rank, int total_processes){
    if(process_rank == 0)
        MPI_Recv(cur_world->cells[0], cur_world->width + 2, MPI_INT, total_processes - 1, bottom_row_tag, MPI_COMM_WORLD, &status);
    else 
        MPI_Recv(cur_world->cells[0], cur_world->width + 2, MPI_INT, process_rank - 1, bottom_row_tag, MPI_COMM_WORLD, &status);
}

static inline void receive_adjacent_bottom_row(int process_rank, int total_processes, int adj_bottom_row){
    if(process_rank == total_processes - 1)
        MPI_Recv(cur_world->cells[adj_bottom_row], cur_world->width + 2, MPI_INT, 0, top_row_tag, MPI_COMM_WORLD, &status);
    else
        MPI_Recv(cur_world->cells[adj_bottom_row], cur_world->width + 2, MPI_INT, process_rank + 1, top_row_tag, MPI_COMM_WORLD, &status);
}

static inline void exchange_rows(int process_rank, int total_processes, int process_rows){
    if(total_processes == 1){
        world_top_bottom_border_wrap(cur_world);
        return;
    }

    if(process_rank % 2 == 0){
        send_top_row(process_rank, total_processes);
        receive_adjacent_bottom_row(process_rank, total_processes, process_rows + 1);
        send_bottom_row(process_rank, total_processes, process_rows);
        receive_adjacent_top_row(process_rank, total_processes);
    }
    else {
        receive_adjacent_bottom_row(process_rank, total_processes, process_rows + 1);
        send_top_row(process_rank, total_processes);
        receive_adjacent_top_row(process_rank, total_processes);
        send_bottom_row(process_rank, total_processes, process_rows);
    }
}

static void gather_world(int process_rows, int* distribution, int* displacement){
    int elements_to_send = process_rows * (cur_world->width + 2);
    MPI_Gatherv(cur_world->cells[1], elements_to_send, MPI_INT, cur_world->cells[0], distribution, displacement, MPI_INT, 0, MPI_COMM_WORLD);
}

static int reduce_live_cells(int process_rank, int top_row, int bottom_row){
    int live_cells, partial_world_live_cells = partial_world_count(cur_world, top_row, bottom_row);
    MPI_Reduce(&partial_world_live_cells, &live_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    return live_cells;
}

static void parallel_gol_loop(int nsteps, int total_processes, int process_rank, int* distribution, int* displacement, int process_rows){
    assert((nsteps > 0) && (process_rows > 0));
    double sssss;
    world* tmp_world;
    int n, live_cells;

    /*  time steps */
    for (n = 0; n < nsteps; n++) {
        StopWatch_resume(comms_sw);
        sssss = MPI_Wtime();
        exchange_rows(process_rank, total_processes, process_rows);
        StopWatch_pause(comms_sw);
        elaps += MPI_Wtime() - sssss;

        world_partial_left_right_border_wrap(cur_world, 0, process_rows + 1);
        world_partial_timestep(cur_world, next_world, 1, process_rows);

        /* swap old and new worlds */
        tmp_world = cur_world;
        cur_world = next_world;
        next_world = tmp_world;

        /* Print world and population according to main arguments*/
        if (print_cells > 0 && (n % print_cells) == (print_cells - 1)) {
            live_cells = reduce_live_cells(process_rank, 1, process_rows);
            if(process_rank == 0)
                printf("%d: %d live cells\n", n, live_cells);
        }

        if (print_world > 0 && (n % print_world) == (print_world - 1)) {
            gather_world(process_rows, distribution, displacement);
            if(process_rank == 0){
                printf("\nafter time step %d:\n\n", n);
                world_print(cur_world);
            }
        }
    }
}

static void parallel_gol(int bwidth, int bheight, int nsteps){
    int total_processes, process_rank;
    int process_assigned_elements, process_rows;
    int live_cells;
    double process_elapsed_sec, max_elapsed_sec, comms_elapsed_sec, max_comms_elapsed_sec;
    int* distribution = NULL;
    int* displacement = NULL;
    double aaaaa;

	init_mpi(&total_processes, &process_rank);
    

    /* root initializes & prints board. */
    if(process_rank == 0){
        create_worlds(bwidth, bheight); // root creates the full world
        world_init();
        world_distribution_init(total_processes, &distribution, &displacement);
        if (print_world > 0) {
            printf("\ninitial world:\n\n");
            world_print(cur_world);
        }
    }

    // root scatters the distribution for each process to know the size of their world
    MPI_Scatter(distribution, 1, MPI_INT, &process_assigned_elements, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    process_rows = process_assigned_elements / (bwidth + 2);

    if(process_rank != 0)
        create_worlds(bwidth, process_assigned_elements / process_rows);

    // root scatters the world according to the distribution and displacements previously calculated 
    MPI_Scatterv(cur_world->cells[0], distribution, displacement, MPI_INT, cur_world->cells[1], process_assigned_elements, MPI_INT, 0, MPI_COMM_WORLD);

    /* Start measuring execution time (for each process). Run main loop. Stop measuring execution time. */
    comms_sw = StopWatch_new();
    StopWatch_pause(comms_sw);
    aaaaa = MPI_Wtime();
    process_sw = StopWatch_new();
    parallel_gol_loop(nsteps, total_processes, process_rank, distribution, displacement, process_rows);
    process_elapsed_sec = StopWatch_elapsed_sec(process_sw);
    total_elaps = MPI_Wtime() - aaaaa;
    comms_elapsed_sec = StopWatch_elapsed_sec(comms_sw);

    //printf("Process %d (Wtime) elapsed: %f\n", process_rank, elaps);
    //printf("Process %d (StopWatch) elapsed: %f\n", process_rank, comms_elapsed_sec);
    printf("Process %d (Wtime) elapsed: %f\n", process_rank, total_elaps);

    /* Iterations are done. Print max elapsed time & number of live cells. */
    MPI_Reduce(&process_elapsed_sec, &max_elapsed_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comms_elapsed_sec, &max_comms_elapsed_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //fprintf(stderr, "Process %d: Game of Life took %10.3f seconds\n", process_rank, process_elapsed_sec);
    live_cells = reduce_live_cells(process_rank, 1, process_rows);
    if(process_rank == 0){
        printf("Number of live cells = %d\n", live_cells);
        fflush(stdout);
        fprintf(stderr, "Game of Life took %f seconds\n", max_elapsed_sec);
        fprintf(stderr, "Communications took %f seconds\n", max_comms_elapsed_sec);
    }

    /* Free resources */
    StopWatch_destroy(comms_sw);
    StopWatch_destroy(process_sw);
    MPI_Finalize();

    if(process_rank == 0){
        free(distribution);
        free(displacement);
    }
}


int main(int argc, char* argv[]) {
    int nsteps;
    int bwidth, bheight;

    /* Get Parameters */
    if (argc != 6) {
        fprintf(stderr, "Usage: %s width height steps print_world print_cells\n", argv[0]);
        exit(1);
    }
    bwidth = atoi(argv[1]);
    bheight = atoi(argv[2]);
    nsteps = atoi(argv[3]);
    print_world = atoi(argv[4]);
    print_cells = atoi(argv[5]);

    if((bwidth <= 0) || (bheight <= 0) || (nsteps <= 0)){
        fprintf(stderr, "bwidth, bheight and nsteps must be positive integers");
        exit(1);
    }

    parallel_gol(bwidth, bheight, nsteps);

    return 0;
}
