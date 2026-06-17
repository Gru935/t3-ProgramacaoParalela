#!/bin/bash
# =============================================================================
#  run_atlantica.sh  -  Benchmark driver for Trabalho 3 (grupo cp12)
# =============================================================================
#  Runs the divide-and-conquer bubble sort (unbalanced AND balanced) for every
#  required process count and prints the elapsed time of each run so you can
#  fill in the speed-up / efficiency tables.
#
#  USAGE (on atlantica, inside the repo directory):
#     chmod +x run_atlantica.sh
#     ./run_atlantica.sh                       # tudo, N=1000000, 3 repeticoes
#     ./run_atlantica.sh 1000000 5             # tamanho / repeticoes custom
#     ./run_atlantica.sh 1000000 3 missing     # SO o que falta coletar (~25 min)
#
#  It (1) compiles both programs with ladcomp and (2) launches every case.
# -----------------------------------------------------------------------------
#  WHY YOUR BIG RUNS (255/511/1023) FAILED AND HOW THIS FIXES IT
#  -------------------------------------------------------------
#  `srun -N 16 -n 255` REQUESTS 16 whole nodes. You only have 2 exclusive
#  nodes, so SLURM can never satisfy that request -> the job pends/fails.
#
#  You do NOT need that many nodes: in this divide-and-conquer model almost
#  every process is an INTERNAL tree node that spends its time BLOCKED in
#  MPI_Recv (idle), waiting for its child. Only the leaves actually compute.
#  So hundreds of MPI ranks happily share 2 nodes. We pack them onto the 2
#  nodes with `--overcommit`, which lets the number of tasks exceed the number
#  of CPUs. That is the whole trick.
# =============================================================================

set -u
SIZE=${1:-1000000}     # vector size (keep identical across ALL runs!)
REPS=${2:-3}           # repetitions per case (report the median/best)
MODE=${3:-full}        # full = todos os casos | missing = so o que falta coletar
NODES=2                # 2 exclusive nodes, as required by the assignment

UNBAL=mpi_bubblesort        # without load balancing
BAL=bubble_bal              # with load balancing

echo "### Compiling with ladcomp ..."
ladcomp -env mpicc mpi_bubblesort.c          get_time.c -o $UNBAL -lm || exit 1
ladcomp -env mpicc mpi_bubblesort_balanced.c get_time.c -o $BAL   -lm || exit 1
echo "### Done. SIZE=$SIZE  REPS=$REPS"
echo

# run <program> <num_procs>
run() {
  local prog=$1 P=$2
  # 1 process  -> 1 node (sequential baseline)
  # P <= 16    -> 2 nodes, one task per physical core (no HT)
  # P  = 31    -> 2 nodes, uses Hyper-Threading (32 logical CPUs)
  # P  > 32    -> 2 nodes, oversubscribed with --overcommit
  local flags
  if   [ "$P" -eq 1 ];  then flags="-N 1 -n 1"
  elif [ "$P" -le 16 ]; then flags="-N $NODES -n $P"
  elif [ "$P" -le 32 ]; then flags="-N $NODES -n $P --overcommit"   # HT region
  else                       flags="-N $NODES -n $P --overcommit"
  fi

  printf "%-16s P=%-5d:" "$prog" "$P"
  local r t
  for r in $(seq 1 "$REPS"); do
    t=$(srun $flags ./"$prog" "$SIZE" 2>/dev/null | awk '/^Elapsed = /{print $3}')
    [ -n "$t" ] || t="FAIL"
    printf " %8s" "$t"
  done
  printf "   (seconds)\n"
}

if [ "$MODE" = "missing" ]; then
  # Coleta SO o que ainda falta, reaproveitando o que ja existe:
  #  - P=1: reutilize o T(1)=3278s do desbalanceado (o balanceado em P=1 e identico);
  #  - desbalanceado 3..127: voces ja tem no graficos.ods;
  #  => roda balanceado 3..1023 + desbalanceado 255/511/1023. (~25 min, nao ~7h)
  echo "=== MODE=missing: balanceado 3..1023 + desbalanceado 255/511/1023 ==="
  echo "    (P=1 reutiliza 3278s; desbalanceado 1..127 voces ja tem)"
  for P in 3 7 15 31 63 127; do
    run $BAL $P
  done
  for P in 255 511 1023; do
    run $UNBAL $P
    run $BAL   $P
  done
else
  echo "=== Required cases: 1, 3, 7, 15, 31 (2 nodes; 31 uses HT) ==="
  for P in 1 3 7 15 31; do
    run $UNBAL $P
    run $BAL   $P
  done

  echo
  echo "=== Scaling cases: 63, 127, 255, 511, 1023 (2 nodes, --overcommit) ==="
  for P in 63 127 255 511 1023; do
    run $UNBAL $P
    run $BAL   $P
  done
fi

echo
echo "All runs finished. Plug the elapsed times into graficos.ods."
echo "Speed-up(P) = T(1) / T(P)        Efficiency(P) = Speed-up(P) / P"
