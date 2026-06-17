/* MPI divide-and-conquer bubble sort  --  LOAD-BALANCED version
   ------------------------------------------------------------------
   Trabalho 3 - Programacao Paralela (PUCRS) - grupo cp12.

   This is the load-balanced counterpart of mpi_bubblesort.c.

   Model (divisao e conquista, Figura 1 do enunciado):
     - The root holds the whole vector. Each internal node of the binary
       process-tree splits its sub-vector and sends one part to a helper
       process, keeping the other part and recursing on it.
     - When a node has no helper left (a leaf of the process-tree) it
       CONQUERS its sub-vector with bubble sort.
     - On the way back up, each node INTERCALATES (merge) the two sorted
       parts and returns the result to its parent, until the root rebuilds
       the whole sorted vector.

   WHY A BALANCED VERSION?
     The original (mpi_bubblesort.c) always splits the sub-vector in HALF.
     With P = 2^(d+1)-1 processes that produces an UNBALANCED tree: exactly
     one process (rank (P-1)/2) ends up with a leaf TWICE as large as all the
     others. Because bubble sort is O(n^2), that single process does ~4x the
     work of the others and dominates the total time.

     Here we split each sub-vector PROPORTIONALLY to the number of leaf
     processes on each side of the tree. As a result every leaf receives the
     same amount of work (~N/P elements) and no single process is a
     bottleneck.

   IMPORTANT: compile with -lm is not required here (we use bit shifts, not
   pow()), but the LAD build line is kept uniform:
     ladcomp -env mpicc mpi_bubblesort_balanced.c get_time.c -o bubble_bal -lm
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

extern double get_time(void);

void bubble_sort(int a[], int size);
void merge_at(int a[], int size, int mid, int temp[]);
int subtree_leaves(int rank, int level, int max_rank);
void sort_parallel_mpi(int a[], int size, int temp[],
                       int level, int my_rank, int max_rank,
                       int tag, MPI_Comm comm);
int topmost_level(int my_rank);
void run_root(int a[], int size, int temp[], int max_rank, int tag,
              MPI_Comm comm);
void run_helper(int my_rank, int max_rank, int tag, MPI_Comm comm);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int comm_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int max_rank = comm_size - 1;
    int tag = 1; /* single message type exchanged between parent and helper */

    if (my_rank == 0)
    { /* Root process: owns the input vector and times the sort. */
        puts("-MPI Bubble Sort (divide and conquer, BALANCED)-\t");
        if (argc != 2)
        {
            printf("Usage: %s array-size\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int size = atoi(argv[1]);
        printf("Array size = %d\nProcesses = %d\n", size, comm_size);

        int *a = malloc(sizeof(int) * size);
        int *temp = malloc(sizeof(int) * size);
        if (a == NULL || temp == NULL)
        {
            printf("Error: Could not allocate array of size %d\n", size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* Same seed as the unbalanced version so both sort the SAME vector. */
        srand(314159);
        int i;
        for (i = 0; i < size; i++)
            a[i] = rand() % size;

        double start = get_time();
        run_root(a, size, temp, max_rank, tag, MPI_COMM_WORLD);
        double end = get_time();
        printf("Start = %.2f\nEnd = %.2f\nElapsed = %.2f\n",
               start, end, end - start);

        /* Correctness check. */
        for (i = 1; i < size; i++)
        {
            if (!(a[i - 1] <= a[i]))
            {
                printf("Implementation error: a[%d]=%d > a[%d]=%d\n",
                       i - 1, a[i - 1], i, a[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        free(a);
        free(temp);
    }
    else
    { /* Every other process is a helper. */
        run_helper(my_rank, max_rank, tag, MPI_COMM_WORLD);
    }

    fflush(stdout);
    MPI_Finalize();
    return 0;
}

/* Root drives the recursion starting at level 0. */
void run_root(int a[], int size, int temp[], int max_rank, int tag,
              MPI_Comm comm)
{
    int my_rank;
    MPI_Comm_rank(comm, &my_rank);
    if (my_rank != 0)
    {
        printf("Error: run_root called from process %d; must be process 0\n",
               my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    sort_parallel_mpi(a, size, temp, 0, my_rank, max_rank, tag, comm);
}

/* A helper waits for its parent's chunk, sorts it (possibly recursing with
   its own helpers) and sends the sorted chunk back. */
void run_helper(int my_rank, int max_rank, int tag, MPI_Comm comm)
{
    int level = topmost_level(my_rank);

    MPI_Status status;
    int size;
    /* Discover how many elements the parent is about to send. */
    MPI_Probe(MPI_ANY_SOURCE, tag, comm, &status);
    MPI_Get_count(&status, MPI_INT, &size);
    int parent_rank = status.MPI_SOURCE;

    int *a = malloc(sizeof(int) * size);
    int *temp = malloc(sizeof(int) * size);
    MPI_Recv(a, size, MPI_INT, parent_rank, tag, comm, &status);

    sort_parallel_mpi(a, size, temp, level, my_rank, max_rank, tag, comm);

    MPI_Send(a, size, MPI_INT, parent_rank, tag, comm);
    free(a);
    free(temp);
}

/* Topmost level of the process-tree in which `my_rank` participates.
   Root (rank 0) participates from level 0. */
int topmost_level(int my_rank)
{
    int level = 0;
    while ((1 << level) <= my_rank)
        level++;
    return level;
}

/* Number of LEAF processes in the sub-tree rooted at (rank, level), i.e. how
   many processes will end up running bubble sort below (and including) this
   node. This is the "weight" used to split the work proportionally.

   At each level a node has a helper at rank + 2^level. If that helper does
   not exist, the node itself is the only leaf of its sub-tree (weight 1).
   Otherwise the weight is the sum of the weights of the node's own
   continuation and of the helper's sub-tree. */
int subtree_leaves(int rank, int level, int max_rank)
{
    int helper = rank + (1 << level);
    if (helper > max_rank)
        return 1; /* leaf: only this process sorts here */
    return subtree_leaves(rank, level + 1, max_rank)
         + subtree_leaves(helper, level + 1, max_rank);
}

/* Recursive parallel sort.
   - If no helper is available, conquer with bubble sort (leaf).
   - Otherwise split the vector PROPORTIONALLY to the number of leaves on each
     side, send the helper's share, recurse on our own share, then merge. */
void sort_parallel_mpi(int a[], int size, int temp[],
                       int level, int my_rank, int max_rank,
                       int tag, MPI_Comm comm)
{
    int helper_rank = my_rank + (1 << level);

    if (helper_rank > max_rank)
    { /* Leaf of the process-tree: conquer. */
#ifdef BALANCE_LOG
        /* Per-process load report (compile with -DBALANCE_LOG). Lets us see
           the work each leaf does at each tree level -> balancing analysis. */
        double t0 = get_time();
        bubble_sort(a, size);
        fprintf(stderr, "BALANCE rank=%d level=%d leaf=%d sort_time=%.3f\n",
                my_rank, level, size, get_time() - t0);
#else
        bubble_sort(a, size);
#endif
        return;
    }

    /* Split point: keep a share proportional to the leaves we will feed on
       our side, hand the rest to the helper's sub-tree. This makes every
       leaf receive ~N/P elements (balanced load). */
    int my_weight = subtree_leaves(my_rank, level + 1, max_rank);
    int helper_weight = subtree_leaves(helper_rank, level + 1, max_rank);
    int total_weight = my_weight + helper_weight;

    /* Use long long to avoid overflow for large vectors. */
    int keep = (int)(((long long)size * my_weight) / total_weight);
    int send_count = size - keep;

    MPI_Request request;
    MPI_Status status;

    /* Send the helper's share (the upper part), asynchronously. */
    MPI_Isend(a + keep, send_count, MPI_INT, helper_rank, tag, comm, &request);
    /* Sort our own share (the lower part) while the helper works. */
    sort_parallel_mpi(a, keep, temp, level + 1, my_rank, max_rank, tag, comm);
    /* The matching receive on the helper side completes the send. */
    MPI_Request_free(&request);
    /* Receive the helper's share back, now sorted. */
    MPI_Recv(a + keep, send_count, MPI_INT, helper_rank, tag, comm, &status);
    /* Intercalate the two sorted parts (boundary at `keep`). */
    merge_at(a, size, keep, temp);
}

/* Merge the two sorted runs a[0..mid) and a[mid..size) into a, using temp.
   Unlike a fixed-midpoint merge, the boundary can be anywhere, which is what
   the proportional split requires. */
void merge_at(int a[], int size, int mid, int temp[])
{
    int i1 = 0, i2 = mid, t = 0;
    while (i1 < mid && i2 < size)
    {
        if (a[i1] <= a[i2])
            temp[t++] = a[i1++];
        else
            temp[t++] = a[i2++];
    }
    while (i1 < mid)
        temp[t++] = a[i1++];
    while (i2 < size)
        temp[t++] = a[i2++];
    memcpy(a, temp, size * sizeof(int));
}

/* Simple exchange sort (ordenacao por troca simples / bubble sort). */
void bubble_sort(int a[], int size)
{
    int i, j;
    for (i = 0; i < size - 1; i++)
        for (j = 0; j < size - i - 1; j++)
            if (a[j] > a[j + 1])
            {
                int aux = a[j];
                a[j] = a[j + 1];
                a[j + 1] = aux;
            }
}
