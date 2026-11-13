# MOMENT Grid Search — Slurm Architecture

Submission command:

    sbatch --array=0-2%2 slurm/search.sh


                    ┌─────────────────────────────┐
                    │        SLURM / nsfq         │
                    │                             │
                    │  Array tasks: 0, 1, 2       │
                    │  Max concurrent tasks: 2    │
                    │  Max runtime/task: 24 hrs   │
                    └──────────────┬──────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
                  ▼                                 ▼

        ┌───────────────────┐             ┌───────────────────┐
        │   ARRAY TASK 0    │             │   ARRAY TASK 1    │
        │      Kettle       │             │  Washing Machine  │
        └─────────┬─────────┘             └─────────┬─────────┘
                  │                                 │
          Resource allocation               Resource allocation
          ───────────────────               ───────────────────
          GPU:       1                      GPU:       1
          CPU cores: 12                     CPU cores: 12
          RAM:       64 GB                  RAM:       64 GB
          Tmax:      24 hrs                 Tmax:      24 hrs
                  │                                 │
                  ▼                                 ▼
        ┌───────────────────┐             ┌───────────────────┐
        │     GPU #1        │             │     GPU #2        │
        │                   │             │                   │
        │ ┌───────────────┐ │             │ ┌───────────────┐ │
        │ │   finetune    │ │             │ │   finetune    │ │
        │ └───────────────┘ │             │ └───────────────┘ │
        │ ┌───────────────┐ │             │ ┌───────────────┐ │
        │ │    frozen     │ │             │ │    frozen     │ │
        │ └───────────────┘ │             │ └───────────────┘ │
        │ ┌───────────────┐ │             │ ┌───────────────┐ │
        │ │    random     │ │             │ │    random     │ │
        │ └───────────────┘ │             │ └───────────────┘ │
        │                   │             │                   │
        │ 3 Python processes│             │ 3 Python processes│
        │ sharing ONE GPU   │             │ sharing ONE GPU   │
        └───────────────────┘             └───────────────────┘


                    ┌─────────────────────────┐
                    │      ARRAY TASK 2       │
                    │      Fridge-Freezer     │
                    │                         │
                    │        PENDING          │
                    │                         │
                    │ waits until Task 0 or 1 │
                    │ finishes and a GPU      │
                    │ becomes available       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        1 GPU            │
                    │        12 CPUs          │
                    │        64 GB RAM        │
                    │        Tmax = 24 hrs    │
                    │                         │
                    │  ┌───────────────────┐  │
                    │  │ finetune          │  │
                    │  │ frozen            │  │
                    │  │ random            │  │
                    │  └───────────────────┘  │
                    │                         │
                    │  3 processes share GPU │
                    └─────────────────────────┘


# Maximum simultaneous resource usage

Because:

    --array=0-2%2

only TWO array tasks can run simultaneously.

Therefore maximum allocation is:

    GPUs:        2 × 1 GPU      = 2 GPUs
    CPU cores:   2 × 12 cores   = 24 CPU cores
    System RAM:  2 × 64 GB      = 128 GB
    Python jobs: 2 × 3 processes = 6 Python processes


# Per-array-task Slurm configuration

    #SBATCH --partition=nsfq
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=12
    #SBATCH --mem=64G
    #SBATCH --time=24:00:00

Meaning:

    Partition       = nsfq
    GPU             = 1 per array task
    CPU             = 12 cores per array task
    System RAM      = 64 GB per array task
    Tmax            = 24 hours per array task


# Total workload

    3 appliances
        ×
    3 MOMENT configurations
        =
    9 searches total

But only:

    2 appliances × 3 searches
        =
    6 searches

can be running simultaneously because of the %2 limit.