# SJSU COE HPC GUIDE

User guide: https://hpc.sjsu.edu/hpc/coe/slurm_user_guide.html

Replace `<sjsu_id>` below with your SJSU ID (the HPC username).

## 1. Machines

| Machine | OS | Internet | GPU | Use for |
|---|---|---|---|---|
| `coe-hpc1` | CentOS 7 (glibc 2.17) | Yes | No | Entry point from outside, `git clone/pull`, `wandb sync`, receiving `rsync` from a laptop |
| `coe-hpc3` (also called `g17`) | Rocky Linux 9 (glibc 2.34) | No | 1x A40 48GB | GPU login node: Slurm (`sbatch`, `squeue`, `sinfo`), conda env, quick GPU checks |
| `g20`–`g32` | Rocky Linux 9 | No | 1x H100 95GB each | Compute nodes; jobs run here via Slurm. Never log in to them directly. |

```
laptop ──ssh──▶ coe-hpc1 (online) ──ssh──▶ coe-hpc3 / g17 (offline, Slurm) ──sbatch──▶ g20–g32 (H100 NSF Nodes)
                     └──────────── all share /home/<sjsu_id> ────────────┘
```

- `coe-hpc3` can't be reached directly from outside; always go through `coe-hpc1`.
- Login nodes (`coe-hpc1`, `g17`) are shared: no training on them. Short checks only.
- Don't run the conda env on `coe-hpc1`: it's built for Rocky 9 and won't work on CentOS 7.

## 2. Logging in

```bash
ssh <sjsu_id>@coe-hpc1.sjsu.edu      # from the laptop (password)
ssh coe-hpc3                         # from coe-hpc1 (key already set up)
exit                                 # back to coe-hpc1
```

SSH keys on the cluster were created automatically at first login (`~/.ssh/id_ed25519`, `~/.ssh/config` with `Host *`). 

## 3. Storage

- Home: `/home/<sjsu_id>` (`~`), on a shared NFS server (`172.16.3.13:/zpool/raid001/home`). It's the same files on every node.
- No per-user quota is shown; the filesystem is shared (120 TB, about 95% full). About 35 GB left for this project.
- `/scratch` and `/data` also exist but aren't used.

| Path | Contents |
|---|---|
| `~/nilm-moment` | Repo (git clone) |
| `~/nilm-moment/data/splits` | Data (copied from the laptop) |
| `~/nilm-moment/runs` | Run outputs |
| `~/miniconda3` | Miniconda; env `nilm` |
| `~/.cache/huggingface` | MOMENT-1-small checkpoint |

## 4. Slurm

| Item | Value |
|---|---|
| Partition | `nsfq` (the old `nsfqs/nsfqm/nsfql` stopped accepting jobs on 2026-11-04) |
| Nodes | `g20`–`g32` (13 nodes), 1x H100 95GB each |
| Max job time | 24 h (`1-00:01:00`); jobs are killed at the limit (state `TIMEOUT`) |
| Our limit | **2 GPUs at a time** = at most 2 running 1-GPU jobs (prof. instruction; the queue itself allows 6 nodes per user) |
| Other GPU queues | `gpuqs/gpuqm/gpuql` (2/7/14 days) not to be used without permission |

Job script header:

```bash
#SBATCH --partition=nsfq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
```

Commands (on `g17`):

| Command | Purpose |
|---|---|
| `sbatch job.sh` | Submit a job |
| `squeue -u $USER` | My jobs (PD = pending, R = running) |
| `scancel <jobid>` | Cancel a job |
| `sinfo -p nsfq -o "%P %l %D %N %G"` | Partition nodes and GPUs |
| `sacct -j <jobid>` | Finished job state, run time |

The username is all digits, so `-u $USER` can be read as a numeric UID. If `squeue -u $USER` shows nothing unexpectedly, use `squeue --me`.

On a time limit: searches resume (finished trials are skipped on resubmit). Training runs don't resume and restart from scratch.

## 5. Python environment

On `g17`:

```bash
conda activate nilm          # Python 3.11, torch (cu124), requirements.txt, uv
```

Conda is loaded from `~/.bashrc` only on Rocky nodes:

```bash
if grep -q Rocky /etc/os-release 2>/dev/null; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi
```

Don't run `conda init`. It makes every `coe-hpc1` login fail, because the shared `.bashrc` would try to run conda on CentOS 7.

Install extra packages with `uv pip install <pkg>` (needs the tunnel, section 6). Plain `pip` breaks with a SOCKS proxy (`PoolKey ... key_proxy_ssl_context`). Conda installs work through the tunnel too.

## 6. Internet on g17 (SSH tunnel)

`g17` is offline. To install packages or download models, open a SOCKS tunnel through `coe-hpc1`:

```bash
# open
ssh -f -N -D 127.0.0.1:18765 coe-hpc1
export ALL_PROXY=socks5h://127.0.0.1:18765 HTTPS_PROXY=socks5h://127.0.0.1:18765 HTTP_PROXY=socks5h://127.0.0.1:18765
curl -sI --max-time 10 https://pypi.org | head -1      # expect HTTP 200

# close
pkill -u "$(id -u)" -f "ssh -f -N -D"
unset ALL_PROXY HTTPS_PROXY HTTP_PROXY
pgrep -u "$(id -u)" -af "ssh -f -N -D"                  # expect no output
```

- Use `"$(id -u)"`, not `$USER`, with `pgrep` and `pkill`: the all-digit username is misread as a UID.
- `Address already in use` means a tunnel is already open (`pgrep` above), or another user holds the port, so pick another port.
- The `bind [::1]` warning is harmless (no IPv6). Binding to `127.0.0.1` avoids it.
- The tunnel is for setup only. Jobs on `g20`–`g32` run offline.

## 7. Offline jobs

Set this in every job script:

```bash
export HF_HUB_OFFLINE=1      # MOMENT loads from ~/.cache/huggingface
```

W&B: set `wandb.mode: offline` in `configs/base.yaml` on the cluster. The code passes the mode to `wandb.init` explicitly ([tracking.py](src/training/tracking.py)), so a `WANDB_MODE` env var may not override it.

Afterwards, upload W&B logs from `coe-hpc1`, which is online. Use a Python that runs on CentOS 7, e.g. `python3 -m pip install --user wandb` there:

```bash
wandb sync ~/nilm-moment/runs/<model>/<appliance>/<run>/wandb/offline-run-*
```

## 8. Code and data

```bash
# code: on coe-hpc1
cd ~/nilm-moment && git pull

# data: from the laptop, into the shared home through coe-hpc1
rsync -avP data/splits/ <sjsu_id>@coe-hpc1.sjsu.edu:~/nilm-moment/data/splits/
```

## 9. MOMENT search jobs

[slurm/search.sh](slurm/search.sh) runs the finetune, frozen and random searches for one appliance as 3 processes on one H100 (about 5 GB of GPU memory each). A job array covers the 3 appliances; `%2` keeps at most 2 GPUs in use.

```bash
# on coe-hpc3, from ~/nilm-moment
mkdir -p slurm/logs
sbatch --array=0-2%2 slurm/search.sh        # 0 = Kettle, 1 = Washing Machine, 2 = Fridge-Freezer
squeue --me
```

| Log | Contents |
|---|---|
| `slurm/logs/moment-search-<job>_<i>.out` | Job start/end, node, exit status |
| `slurm/logs/<job>_<i>/<model>-<appliance>.log` | Per-search output (trials, epochs, val loss) |
| `slurm/logs/<job>_<i>/gpu.csv` | GPU usage and memory, once a minute |

If a task hits the 24 h limit, resubmit just that index (e.g. `sbatch --array=0 slurm/search.sh`). Finished trials are skipped.
