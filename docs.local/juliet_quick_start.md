# Juliet GPU Training — Quick Cheat Sheet

## 0) One-time setup on your Windows laptop

**SSH config** (so `ssh juliet` “just works”)

```
# C:\Users\yuan\.ssh\config
Host juliet
    HostName juliet.mesonet.fr
    User zhenyuan
    IdentityFile C:\Users\yuan\.ssh\id_ed25519
    IdentitiesOnly yes
```

**(Optional) VS Code Remote-SSH**

* Install *Remote – SSH* extension → “Connect to Host…” → `juliet`, then open your project folder on the remote. ([Visual Studio Code][1])

---

## 1) Log in & basic FS tips

```bash
ssh juliet                      # lands you on login node (e.g., juliet1)
pwd; hostname                   # sanity check where you are
ls -la                          # list files
```

Persistent storage: use `/project/<ACCOUNT>/...` for code/data; use `/scratch_l/$USER/$SLURM_JOB_ID` inside jobs for fast, temporary I/O (stage in/out in your scripts). (Per Juliet docs/workflow.) ([mesonet.fr][2])

---

## 2) Move between login and compute nodes (Slurm)

**Start an interactive GPU shell (login → compute node):**

```bash
srun -p mesonet --gres=gpu:1 -c 8 --mem=64G --time=02:00:00 \
     --account=m25150 --pty bash -i
hostname && nvidia-smi
```

This both allocates and opens a shell on a GPU node (e.g., `juliet2`). Use `exit` (or `Ctrl+D`) to return to the login node. ([slurm.schedmd.com][3])

**Submit batch jobs:**

```bash
sbatch train_slurm.sh
squeue -u $USER
```

Juliet runs Slurm; the partition is **mesonet**. ([mesonet.fr][2])

---

## 3) Clone your private repo (GitHub)

### Option A — Create a key *on Juliet* and add to GitHub

```bash
ssh-keygen -t ed25519 -C "juliet-key"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub   # paste into GitHub → Settings → SSH keys
ssh -T git@github.com       # should greet you
cd /project/m25150 && git clone git@github.com:forfrt/SteerMoE.git
```

GitHub docs: generating a key, adding it to your account. ([GitHub Docs][4])

### Option B — Use your Windows key via agent forwarding

```powershell
# on Windows
Start-Service ssh-agent
ssh-add C:\Users\yuan\.ssh\id_ed25519
ssh -A juliet
ssh -T git@github.com
cd /project/m25150 && git clone git@github.com:forfrt/SteerMoE.git
```

(Forwarded keys let Juliet authenticate to GitHub using your local agent.) ([GitHub Docs][4])

---

## 4) Create the conda env & install GPU PyTorch

```bash
# once per env
conda create -n steermoe python=3.10 -y
conda activate steermoe

# install the correct CUDA-enabled PyTorch wheels (CUDA 12.4 build)
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

(Use the official PyTorch wheel index for CUDA builds.) ([PyTorch][5])

**Install the project deps (cleaned file):**

```bash
cd /project/m25150/SteerMoE
pip install -r requirements_juliet.txt
```

**GPU sanity check (run on a compute node):**

```bash
python - <<'PY'
import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY
```

**Freeze for reproducibility:**

```bash
pip freeze > requirements.lock.txt
conda env export --no-builds > environment.yml
```

> If pip reports `ResolutionImpossible`, read the message and adjust pins (e.g., align `fsspec/gcsfs`, `google-auth` ranges). Pip’s resolver is strict and won’t install conflicting sets. ([pip.pypa.io][6])

---

## 5) Minimal `sbatch` template (single node, 1× A100)

```bash
#!/bin/bash
#SBATCH -J steermoe-1gpu
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 28
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --account=m25150
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail

source ~/.bashrc
conda activate steermoe

SCRATCH=/scratch_l/$USER/$SLURM_JOB_ID
mkdir -p "$SCRATCH" logs
# rsync -a /project/m25150/datasets/ "$SCRATCH/datasets/"

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO

python -u train.py \
  --data "$SCRATCH/datasets" \
  --output /project/m25150/outputs/$SLURM_JOB_ID \
  --epochs 5 --batch-size 64
```

Juliet uses Slurm (`sbatch`/`srun`) and the **mesonet** partition; adapt CPUs/mem/time as needed. ([mesonet.fr][2])

---

## 6) Quick commands you used (memory jogger)

```bash
# list files (Windows CMD on your PC)
dir C:\Users\yuan\.ssh /b

# make SSH config on Windows
type nul > C:\Users\yuan\.ssh\config
notepad C:\Users\yuan\.ssh\config

# check project account(s) on Juliet (which to charge)
sacctmgr --noheader --parsable2 show association where user=$USER format=account

# interactive GPU session (compute node), then return to login node
srun -p mesonet --gres=gpu:1 -c 8 --mem=64G --time=02:00:00 --account=m25150 --pty bash -i
exit  # back to juliet1
```

---

## 7) Using VS Code with Slurm (pattern)

* Edit code in VS Code Remote-SSH (login node).
* For quick tests: open a VS Code terminal → `srun ... --pty bash -i` → run Python on the compute node.
* For long runs: `sbatch your_script.sh` and tail logs in VS Code. ([Visual Studio Code][1])

---

If you want, I can add an **8× A100 DDP** `sbatch` and a tiny **data-staging helper** you can reuse across jobs.

[1]: https://code.visualstudio.com/docs/remote/ssh?utm_source=chatgpt.com "Remote Development using SSH"
[2]: https://www.mesonet.fr/documentation/user-documentation/code_form/juliet/jobs/?utm_source=chatgpt.com "Lancer un calcul | Documentation de MesoNET"
[3]: https://slurm.schedmd.com/srun.html?utm_source=chatgpt.com "srun - Slurm Workload Manager - SchedMD"
[4]: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?utm_source=chatgpt.com "Generating a new SSH key and adding it to the ssh-agent"
[5]: https://pytorch.org/get-started/previous-versions/?utm_source=chatgpt.com "Previous PyTorch Versions"
[6]: https://pip.pypa.io/en/stable/topics/dependency-resolution/?utm_source=chatgpt.com "Dependency Resolution - pip documentation v25.2"
