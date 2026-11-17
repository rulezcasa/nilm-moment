# MOMENT Final Training — Slurm Architecture

Submission command:

    sbatch --array=0-8%2 slurm/train.sh


# Hyperparameters (from search-v1 + search-v2)

One config per model, shared across appliances. RevIN off.

    Model             learning_rate   head_dropout   patience   epochs   max_windows_per_file
    moment_finetune   3e-5            0.0            12         100      100000
    moment_frozen     1e-4            0.0            12         100      100000
    moment_random     1e-5            0.0            12         100      100000

Common: batch 128, adam, weight_decay 0, fp32, mse, grad_clip_norm 1.0.

Search history:

    search-v1   lr x head_dropout grid (6 trials), patience 8
                dropout had no consistent effect; best lr at grid edge for frozen/random
    search-v2   lr only, head_dropout 0.0, patience 12 (3 trials)
                finetune 3e-5 best on all 3; frozen 1e-4 best on all 3;
                random 1e-5 best on Kettle/WM and on summed loss (FF preferred 3e-5)


                    ┌─────────────────────────────┐
                    │        SLURM / nsfq         │
                    │                             │
                    │  Array tasks: 0 – 8         │
                    │  Max concurrent tasks: 2    │
                    │  Max runtime/task: 24 hrs   │
                    └──────────────┬──────────────┘
                                   │
         task i → appliance = i / 3, seed = 42 + i % 3
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼

  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
  │    Kettle    │          │ Washing Mach │          │ Fridge-Freez │
  │              │          │              │          │              │
  │ task 0  s42  │          │ task 3  s42  │          │ task 6  s42  │
  │ task 1  s43  │          │ task 4  s43  │          │ task 7  s43  │
  │ task 2  s44  │          │ task 5  s44  │          │ task 8  s44  │
  └──────────────┘          └──────────────┘          └──────────────┘


        Each task = one appliance, one seed, one GPU
        ────────────────────────────────────────────

                    ┌─────────────────────────┐
                    │        1 H100           │
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
                    │  3 processes share GPU  │
                    └─────────────────────────┘


# Why one seed per task

Final runs use 100k windows/file (search: 30k), so epochs are ~3.3x longer.
Finetune ≈ 650 s/epoch → ~8–9 h with early stopping, ~18 h at 100 epochs.
Three seeds in one task would exceed 24 h, and training runs don't resume.


# Total workload

    3 appliances × 3 seeds = 9 array tasks
    9 tasks × 3 models     = 27 training runs

    2 tasks at a time → ~5 waves × 8–18 h ≈ 2–4 days (plus queue)


# Maximum simultaneous resource usage

    GPUs:        2 × 1 GPU       = 2 GPUs
    CPU cores:   2 × 12 cores    = 24 CPU cores
    System RAM:  2 × 64 GB       = 128 GB
    Python jobs: 2 × 3 processes = 6 Python processes


# Outputs

    runs/moment_<variant>/<appliance>/<seed>/
        best.pt        best val checkpoint (~150 MB)
        config.yaml    resolved config
        metrics.csv    per-epoch train/val loss
        result.json    best epoch, best val loss
        wandb/         offline run (sync later)

Test evaluation is a separate step after training.


# Monitoring

Run on coe-hpc3 (g17) from ~/nilm-moment. Replace <jobid> with the number
printed by sbatch (current run: 116106).

## Job status

    squeue -u 019111521
    squeue --start -u 019111521     # estimated start time
    sacct -j <jobid>
    sacct -j <jobid>_<i> --format=JobID,State,Elapsed,NodeList,ExitCode

Pending reasons: (JobArrayTaskLimit) = held by %2, (Resources) = no free NSF GPU.
Final states: COMPLETED, FAILED, TIMEOUT (hit 24 h, resubmit), CANCELLED.

## Logs

    slurm/logs/moment-train-<jobid>_<i>.out                   # job start/end, node, status
    slurm/logs/<jobid>_<i>/<model>-<appliance>-<seed>.log     # one per training process
    slurm/logs/<jobid>_<i>/gpu.csv                            # GPU use + memory, once a minute

    cat slurm/logs/moment-train-<jobid>_0.out
    tail -n 3 slurm/logs/<jobid>_<i>/*.log                    # latest epochs of all 3 models
    grep -il "traceback\|error" slurm/logs/<jobid>_*/*.log    # find crashed runs

Check the finetune epoch time on the first task: well above ~650 s/epoch
means a 100-epoch run could hit the 24 h limit.

## GPU

    tail -n 5 slurm/logs/<jobid>_<i>/gpu.csv

## Results

    cat runs/moment_*/*/4[2-4]/result.json
    ls runs/moment_*/*/

## Control

    scancel <jobid>                    # cancel all tasks
    scancel <jobid>_<i>                # cancel one task
    sbatch --array=<i> slurm/train.sh  # rerun one task (restarts from scratch)


# Copy back to the Mac

    rsync -avP --exclude='best.pt' \
      --include='moment_*/' --include='moment_*/*/' --include='moment_*/*/4[2-4]/***' --exclude='*' \
      019111521@coe-hpc1.sjsu.edu:'~/nilm-moment/runs/' runs/

    wandb sync runs/moment_*/*/4[2-4]/wandb/offline-run-*
