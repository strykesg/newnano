# Project Chimera Deployment & Distillation Playbook

This guide walks through the full lifecycle for the Chimera nanochat stack: provisioning an Azure GPU environment, running the hypertraining pipeline, distilling to a deployable artifact, and standing up lightweight inference on a VPS. The instructions assume familiarity with Linux, Azure, and basic MLOps tooling.

---

## 1. Provision Azure GPU Capacity

### 1.1 Prerequisites
- Azure subscription with quota for ND H100 v5 or equivalent 8×H100 SKU.
- Azure CLI ≥ 2.50 installed locally (`curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash`).
- SSH keypair (`ssh-keygen -t ed25519 -C "chimera-train"`).
- Resource group (e.g. `chimera-rg`) in the target region that supports H-series GPUs (EastUS, WestEurope, etc.).

### 1.2 Check Quota & Request Increases
```bash
az login
az account set --subscription "YOUR-SUBSCRIPTION"
az vm list-usage --location eastus | grep -i h100
```
If quota is insufficient, submit an Azure support request for the chosen VM SKU (e.g. `Standard_ND96asr_v5`).

### 1.3 Create Networking & Storage
```bash
LOCATION="eastus"
RG="chimera-rg"
VNET="chimera-vnet"
SUBNET="gpu-subnet"
NSG="chimera-nsg"

az group create --name $RG --location $LOCATION
az network vnet create \
  --resource-group $RG \
  --name $VNET \
  --address-prefixes 10.42.0.0/16 \
  --subnet-name $SUBNET \
  --subnet-prefix 10.42.1.0/24
az network nsg create --resource-group $RG --name $NSG
az network nsg rule create \
  --resource-group $RG --nsg-name $NSG \
  --name AllowSSH \
  --priority 1000 --access Allow --direction Inbound \
  --protocol Tcp --source-address-prefixes "*" \
  --source-port-ranges "*" --destination-port-ranges 22
```

### 1.4 Spin Up the GPU VM
```bash
VM="chimera-trainer"
IMAGE="microsoft-dsvm:nvidia-gpu-optimized:ubuntu-2204:latest"  # includes drivers + CUDA
SIZE="Standard_ND96asr_v5"  # 8×H100, 95GB GPU memory each

az vm create \
  --resource-group $RG --name $VM \
  --image $IMAGE \
  --size $SIZE \
  --ssh-key-values ~/.ssh/chimera-train.pub \
  --vnet-name $VNET --subnet $SUBNET \
  --nsg $NSG --public-ip-sku Standard

az vm extension set \
  --publisher Microsoft.HpcCompute --name nvidia-gpu-driver-linux \
  --vm-name $VM --resource-group $RG
```
Capture the public IP:
```bash
az vm show -d -g $RG -n $VM --query publicIps -o tsv
```

### 1.5 Harden & Snapshot
- Restrict NSG to trusted IP ranges once tested.
- Enable boot diagnostics and capture an OS disk snapshot for rollback (`az snapshot create ...`).

---

## 2. Prepare the Training Environment

### 2.1 SSH Into the VM
```bash
ssh -i ~/.ssh/chimera-train ubuntu@<GPU_PUBLIC_IP>
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
cd /mnt
sudo mkdir -p /mnt/chimera && sudo chown $USER:$USER /mnt/chimera
cd /mnt/chimera
git clone https://github.com/YOUR-FORK/nanochat.git
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
cd /mnt/chimera/nanochat
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
Consider rsync to Azure Blob Storage:
```bash
az storage azcopy blob upload -c checkpoints --account-name <acct> \
  --source ~/.cache/nanochat/chatdpo_checkpoints --destination-path chimera-dpo/
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
- [ ] Azure VM quota confirmed & NSG locked down.
- [ ] `OPENROUTER_API_KEY` functional (test via `python -m scripts.synthetic_data_gen --sft-examples 2 --dpo-examples 2`).
- [ ] Speedrun completed; `report/report.md` archived.
- [ ] Student/distilled model evaluated vs. teacher metrics.
- [ ] Quantised artifact validated for latency on target hardware.
- [ ] VPS hardened (fail2ban, firewall rules, automatic updates).

---

## 7. Troubleshooting Tips
- **Synthetic generation throttled:** adjust `OPENROUTER_MODELS` list, lower concurrency (script is serialised by default), or cache intermediate outputs.
- **CUDA OOM during DPO:** reduce `device_batch_size`, lower `target_examples_per_step`, or shorten `max_length`.
- **Azure VM preemption:** NDv5 SKUs are pay-as-you-go; for reserved savings use 1-year reservation or scale set with Spot (not recommended for the 4-hour run).
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
- ND96asr_v5 @ ~$44/hr → speedrun (~4h) ≈ $176.
- OpenRouter usage depends on model choice; budget for ~15k requests (SFT + DPO). Consider caching outputs if rerunning.
- VPS (16 GB RAM, dedicated vCPU) ≈ $20–40/mo.

---

Maintain version control for all configuration changes, and tag successful runs (e.g. `git tag chimera-hypertrain-2025-02-10`). For long-term reproducibility, capture the `report/` directory, environment exports, and dataset seeds.
