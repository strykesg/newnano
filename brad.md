# Project Chimera Deployment & Distillation Playbook

This guide walks through the full lifecycle for the Chimera nanochat stack: provisioning a Vast.ai GPU environment, running the hypertraining pipeline, distilling to a deployable artifact, and standing up lightweight inference on a VPS. The instructions assume familiarity with Linux, Docker, and basic MLOps tooling.

---

## 1. Provision Vast.ai GPU Capacity

### 1.1 Prerequisites
- Vast.ai account (sign up at https://vast.ai).
- SSH keypair (`ssh-keygen -t ed25519 -C "chimera-train"`).
- Sufficient account balance (~$40-80 for 4-hour training run, depending on GPU selection).
- Basic familiarity with Docker (Vast.ai instances run in containers).

### 1.2 Select GPU Instance
Log into Vast.ai and search for instances:
1. **Filter criteria:**
   - GPU: 8×H100 (80GB) preferred, or 8×A100 (80GB) as fallback.
   - CUDA Version: ≥12.0.
   - Disk Space: ≥500 GB (preferably 1 TB for tokenizer shards + checkpoints).
   - Bandwidth: ≥100 Mbps upload (for syncing artifacts).
   - Reliability Score: ≥95% to minimize interruptions.
   - Docker Image: `pytorch/pytorch:latest` or Ubuntu-based image with CUDA pre-installed.

2. **Sort by:**
   - $/hr (ascending) to find best value.
   - DLPerf score for performance benchmarks.

3. **Recommended filters via CLI (optional):**
```bash
# Install vastai CLI
pip install vastai
vastai set api-key YOUR_API_KEY

# Search for 8×H100 instances
vastai search offers 'num_gpus=8 gpu_name=H100 disk_space>=500 reliability>=0.95 cuda_max_good>=12.0'
```

### 1.3 Add Your SSH Public Key
Before renting, add your SSH key to your Vast.ai account:
```bash
cat ~/.ssh/chimera-train.pub
```
Copy the output and paste it in **Account → SSH Keys** on the Vast.ai web console.

### 1.4 Rent & Launch the Instance
Via Web UI:
1. Click **RENT** on your selected instance.
2. Choose **On-demand** (interruptible is cheaper but risky for 4-hour runs).
3. Set **Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` or similar.
4. Set **Disk Space**: 500-1000 GB.
5. Launch instance.

Via CLI:
```bash
# Rent instance (replace INSTANCE_ID with the ID from search results)
vastai create instance INSTANCE_ID \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
  --disk 1000 \
  --onstart-cmd "apt update && apt install -y git tmux htop"

# Check instance status
vastai show instances
```

### 1.5 Connect via SSH
Once the instance status shows **running**, note the SSH command:
```bash
# Vast.ai provides a custom SSH command, typically:
ssh -p PORT_NUMBER root@INSTANCE_IP -L 8080:localhost:8080

# Or retrieve connection details via CLI
vastai ssh-url INSTANCE_ID
```

Example:
```bash
ssh -p 12345 root@123.45.67.89 -i ~/.ssh/chimera-train
```

### 1.6 Verify GPU Setup
Once connected:
```bash
nvidia-smi  # Should show 8×H100 or 8×A100
nvcc --version  # Confirm CUDA toolkit version
df -h  # Verify disk space
```

### 1.7 Cost Management & Snapshots
- **Monitor spending:** Check Vast.ai dashboard regularly; instances bill by the minute.
- **Auto-shutdown:** Set a `tmux` session with a shutdown timer if needed:
  ```bash
  echo "sudo poweroff" | at now + 5 hours
  ```
- **Save progress:** Vast.ai doesn't offer native snapshots; use rsync or cloud storage (S3, Dropbox, rclone) to back up checkpoints periodically.

---

## 2. Prepare the Training Environment

### 2.1 SSH Into the Instance
```bash
# Use the connection details from Vast.ai (usually root user in Docker containers)
ssh -p <PORT> -i ~/.ssh/chimera-train root@<INSTANCE_IP>
```

### 2.2 System Prep
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git htop nvtop tmux
# Ensure CUDA drivers are healthy
nvidia-smi
```

### 2.3 Clone & Configure nanochat
```bash
# Vast.ai instances typically mount large storage at /workspace
cd /workspace
mkdir -p chimera
cd chimera
git clone https://github.com/strykesg/newnano.git
cd nanochat
```
Set critical environment variables in `~/.bashrc` (or `.zshrc`):
```bash
cat <<'ENV' >> ~/.bashrc
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_API_KEY="<optional>"
export WANDB_RUN="chimera-speedrun"
export OPENROUTER_API_KEY="<required>"
# Optionally pin model preferences for synthetic generation
export OPENROUTER_MODELS="openai/gpt-4o-mini,anthropic/claude-3.5-sonnet"
ENV
source ~/.bashrc
```

### 2.4 Install Tooling via uv
The `speedrun.sh` script manages most bootstrap steps. Verify `uv` is present:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv --version
```
Optionally, warm the cache:
```bash
~/.local/bin/uv sync
```

### 2.5 Review Key Artifacts
- `speedrun.sh` now runs tokenizer → synthetic data generation → base/mid/SFT → DPO.
- `scripts/synthetic_data_gen.py` uses OpenRouter; confirm quotas and rate limits.
- Ensure disk (~1TB) is sufficient for 240 tokenizer shards + checkpoints.

---

## 3. Execute the Chimera Hypertraining Pipeline

### 3.1 Launch the Speedrun
```bash
cd /workspace/chimera/nanochat
bash speedrun.sh | tee speedrun.log
```
Key checkpoints:
1. Tokenizer training (~30 min).
2. Synthetic dataset generation (`~/.cache/nanochat/datasets/*`).
3. Base + mid + SFT training.
4. New DPO finetune and evaluation stage.
5. `report/report.md` summarises metrics; `chatdpo_checkpoints/` contains the preference-aligned model.

### 3.2 Monitoring & Recovery
- Use `tmux` to avoid SSH drops.
- Monitor GPU utilisation (`watch -n 30 nvidia-smi`).
- If any stage fails, re-run the individual module with the same flags (e.g. `torchrun ... -m scripts.dpo_train -- --run=$WANDB_RUN`).

### 3.3 Artifacts to Preserve
- `~/.cache/nanochat/chatsft_checkpoints/d*/` (SFT model).
- `~/.cache/nanochat/chatdpo_checkpoints/d*/` (preference model).
- `report/report.md`, `report/chat-dpo.md`, `datasets/*.jsonl` (for reproducibility).
- Speedrun log.
Consider syncing to cloud storage before terminating the instance:
```bash
# Option 1: rsync to remote server
rsync -avz ~/.cache/nanochat/chatdpo_checkpoints user@your-server:/backup/chimera-dpo/

# Option 2: rclone to S3/GCS/Dropbox
rclone copy ~/.cache/nanochat/chatdpo_checkpoints remote:chimera-dpo/ --progress

# Option 3: tar and download via scp
tar -czf chimera-checkpoints.tar.gz ~/.cache/nanochat/*_checkpoints report/
# Then from local machine: scp -P <PORT> root@<IP>:/workspace/chimera/nanochat/chimera-checkpoints.tar.gz .
```

### 3.4 Archive & Download (Final Steps)

When the GPU run is near completion, archive only the GPU-generated artifacts or the full working directory, then download to your local machine.

- Archive only GPU outputs (recommended):
  - Tokenizer
  ```bash
  tar -czf ~/tokenizer_$(date +%Y%m%d_%H%M%S).tar.gz -C ~/.cache/nanochat tokenizer
  ```
  - Datasets (SFT/DPO)
  ```bash
  tar -czf ~/datasets_$(date +%Y%m%d_%H%M%S).tar.gz -C ~/.cache/nanochat datasets
  ```
  - Checkpoints (create each if present)
  ```bash
  for D in base_checkpoints mid_checkpoints chatsft_checkpoints chatdpo_checkpoints; do
    [ -d "$HOME/.cache/nanochat/$D" ] && \
      tar -czf ~/${D}_$(date +%Y%m%d_%H%M%S).tar.gz -C ~/.cache/nanochat "$D" || true;
  done
  ```

- Download archives to local (replace <PORT> and <IP>):
  ```bash
  # From your local machine
  scp -P <PORT> root@<IP>:~/tokenizer_*.tar.gz .
  scp -P <PORT> root@<IP>:~/datasets_*.tar.gz .
  scp -P <PORT> root@<IP>:~/{base,mid,chat*sft,chat*dpo}_checkpoints_*.tar.gz . 2>/dev/null || true
  ```

- Optionally, archive the full repo+cache working set:
  ```bash
  # On the GPU server
  TS=$(date +%Y%m%d_%H%M%S)
  BDIR=/root/nanochat_bundle_$TS
  BUNDLE=/root/nanochat_bundle_$TS.tar.gz
  rm -rf "$BDIR" && mkdir -p "$BDIR"
  rsync -a /workspace/chimera/nanochat/ "$BDIR"/nanochat/
  rsync -a ~/.cache/nanochat/        "$BDIR"/cache/
  tar -czf "$BUNDLE" -C /root "nanochat_bundle_$TS"
  
  # From your local machine
  scp -P <PORT> root@<IP>:/root/nanochat_bundle_*.tar.gz .
  ```

- After download, extract locally and place under your cache if needed:
  ```bash
  tar -xzf tokenizer_*.tar.gz
  tar -xzf datasets_*.tar.gz
  mkdir -p ~/.cache/nanochat
  rsync -a tokenizer ~/.cache/nanochat/
  rsync -a datasets ~/.cache/nanochat/
  # (and extracted *_checkpoints as desired)
  ```

---

## 4. Distillation & Compression Strategy

The goal is to deploy on a modest VPS (e.g. 4–8 vCPU, 16–32 GB RAM, single consumer GPU or CPU-only). Recommended pipeline:

### 4.1 Choose a Student Architecture
- Target: 3B or smaller (e.g. `d12` or `d8` nanochat configs) to balance latency and quality.
- Clone repo on GPU VM and configure a smaller model training run; reuse tokenizer & datasets.

#### Training Command
```bash
# Example: train a d12 (≈180M params) student with SFT + DPO distillation
WANDB_RUN=chimera-student bash <<'RUN'
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12 --run=$WANDB_RUN --student_from=chatdpo
RUN
```
If full re-pretraining is too costly, run SFT-only using the existing tokenizer:
```bash
python -m scripts.chat_sft -- source=base --run=chimera-student \
  --sft_dataset=trader_mix --sft_mix_trader=3 --sft_mix_smoltalk=1 \
  --device_batch_size=8 --target_examples_per_step=64
```

### 4.2 Knowledge Distillation for DPO
1. **Dataset:** Use `trader_dpo_data.jsonl` plus additional synthetic preferences generated with the teacher model (`scripts.synthetic_data_gen --dpo-examples N`).
2. **Teacher Forcing:** In `scripts/dpo_train.py`, set `load_model("dpo")` for teacher reference instead of SFT.
3. **KL Regularisation:** Optionally add a `torch.nn.KLDivLoss` term to keep logits close for overlapping outputs (custom modification).
4. **Hyperparameters:** Reduce `beta` (e.g. 0.05) to prevent overfitting when the student capacity is smaller.

### 4.3 Quantisation & Format Conversion
- Export `state_dict` and convert to 4-bit GPTQ or AWQ using `AutoGPTQ` or `bitsandbytes`:
```python
from transformers import AutoModelForCausalLM
from auto_gptq import state_dict_to_gptq
# Convert the student checkpoint -> GPTQ for inference efficiency
```
- Alternatively, convert to GGUF via `llama.cpp` tooling for CPU-friendly inference:
```bash
python convert-to-gguf.py --input student_model --output chimera-student.gguf --quant q4_k_m
```

### 4.4 Validation
- Run `scripts.chat_eval -i dpo -a "ARC-Easy|ARC-Challenge|MMLU"` on the student.
- Evaluate latency on the GPU VM with `scripts.chat_cli` to simulate expected VPS load.

---

## 5. Deploy on a VPS

### 5.1 VPS Requirements
- CPU-only deployment (GGUF + llama.cpp): ≥16 GB RAM, AVX2 support.
- GPU-assisted (e.g. RTX 4090): ensure driver + CUDA compatibility.
- Ubuntu 22.04 recommended.

### 5.2 Prepare the Runtime
```bash
ssh user@vps
sudo apt update && sudo apt install -y git build-essential python3 python3-venv tmux
```
Transfer artifacts:
```bash
rsync -avz ~/.cache/nanochat/chatdpo_checkpoints/d12 user@vps:/opt/chimera/checkpoints/
rsync -avz chimera-student.gguf user@vps:/opt/chimera/gguf/
```

### 5.3 Lightweight Serving Options

#### Option A: llama.cpp (CPU/GPU)
```bash
cd /opt/chimera
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make LLAMA_CUBLAS=1  # set if GPU available
./main -m ../gguf/chimera-student.gguf -p "How should I size a trade?" -n 256
```
Wrap with a simple REST server (FastAPI/uvicorn) to expose endpoints.

#### Option B: Text-Generation-Inference (GPU)
- Containerise student model in Hugging Face TGI with quantisation.
- Deploy via Docker Compose with a reverse proxy (Caddy/Nginx) + TLS.

### 5.4 Observability & Autoscaling
- Collect metrics (latency, token/s) with Prometheus + Grafana or simple logs.
- Use systemd to keep the service alive:
```bash
sudo tee /etc/systemd/system/chimera.service <<'UNIT'
[Unit]
Description=Chimera nanochat inference service
After=network.target

[Service]
User=chimera
WorkingDirectory=/opt/chimera/server
ExecStart=/usr/bin/python3 serve.py
Restart=always

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl enable --now chimera.service
```

---

## 6. Operational Checklist
- [ ] Vast.ai instance rented with sufficient disk space & 8×GPU configuration verified.
- [ ] SSH access confirmed and tmux session started for stability.
- [ ] `OPENROUTER_API_KEY` functional (test via `python -m scripts.synthetic_data_gen --sft-examples 2 --dpo-examples 2`).
- [ ] Speedrun completed; `report/report.md` archived and checkpoints backed up.
- [ ] Instance terminated after successful backup (to avoid ongoing charges).
- [ ] Student/distilled model evaluated vs. teacher metrics.
- [ ] Quantised artifact validated for latency on target hardware.
- [ ] VPS hardened (fail2ban, firewall rules, automatic updates).

---

## 7. Troubleshooting Tips
- **Synthetic generation throttled:** adjust `OPENROUTER_MODELS` list, lower concurrency (script is serialised by default), or cache intermediate outputs.
- **CUDA OOM during DPO:** reduce `device_batch_size`, lower `target_examples_per_step`, or shorten `max_length`.
- **Vast.ai instance interruption:** Choose "On-demand" over "Interruptible" for critical runs. If interrupted, rent a new instance and resume from the last checkpoint saved in `~/.cache/nanochat/`.
- **Disk space exhausted:** Monitor with `df -h`; delete intermediate tokenizer shards after training or request larger disk when renting.
- **SSH connection drops:** Always use `tmux` or `screen` to keep processes running. Reconnect with `tmux attach`.
- **VPS latency spikes:** ensure swap is disabled or sized appropriately; use 4-bit quantisation or streaming KV-cache for long responses.

---

## 8. Appendices

### 8.1 Useful Commands
```bash
# Resume only the DPO stage after addressing issues
torchrun --standalone --nproc_per_node=8 -m scripts.dpo_train -- --run=$WANDB_RUN --beta=0.05

# Evaluate DPO checkpoint directly
python -m scripts.chat_eval -i dpo -g d20 -a "ARC-Easy|GSM8K" --dtype float32

# Regenerate synthetic data with a different teacher persona
OPENROUTER_MODELS="openai/gpt-4o" \
python -m scripts.synthetic_data_gen --sft-examples 1000 --dpo-examples 2000 --seed 2025
```

### 8.2 Cost Snapshot (reference)
- Vast.ai 8×H100 (80GB): ~$8–15/hr depending on availability → speedrun (~4h) ≈ $32–60.
- Vast.ai 8×A100 (80GB): ~$5–10/hr → speedrun (~4h) ≈ $20–40 (if H100 unavailable).
- OpenRouter usage depends on model choice; budget for ~15k requests (SFT + DPO). Consider caching outputs if rerunning.
- VPS (16 GB RAM, dedicated vCPU) ≈ $20–40/mo.

---

Maintain version control for all configuration changes, and tag successful runs (e.g. `git tag chimera-hypertrain-2025-02-10`). For long-term reproducibility, capture the `report/` directory, environment exports, and dataset seeds.
