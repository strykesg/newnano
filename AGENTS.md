# Nanochat Agent Guide

## Build/Lint/Test Commands

### Setup and Dependencies
- Install dependencies: `uv sync`
- Build Rust tokenizer extension: `uv run maturin develop --release --manifest-path rustbpe/Cargo.toml`
- Full end-to-end pipeline: `bash speedrun.sh`

### Testing
- Run all tests: `python -m pytest tests/ -v`
- Run specific test file: `python -m pytest tests/test_rustbpe.py -v -s`
- Run single test: `python -m pytest tests/test_rustbpe.py::test_correctness -v`
- Run slow tests: `python -m pytest tests/ -m slow -v`
- Run tests without slow ones: `python -m pytest tests/ -m "not slow" -v`

### Development Commands
- Chat CLI: `python -m scripts.chat_cli -p "your prompt"`
- Web UI: `python -m scripts.chat_web`
- Generate synthetic data: `python -m scripts.synthetic_data_gen`
- Data download: `python -m nanochat.dataset -n <num_shards>`

## Architecture and Codebase Structure

### Core Components
- **nanochat/**: Main package containing core LLM functionality
  - `gpt.py`: GPT model architecture
  - `engine.py`: Inference engine with generation logic
  - `dataloader.py`: Data loading utilities
  - `tokenizer.py`: Python wrapper for Rust BPE tokenizer
  - `checkpoint_manager.py`: Model checkpoint handling
  - `common.py`: Shared utilities and distributed training setup

- **scripts/**: Command-line entry points for training and inference
  - `base_train.py`: Pretraining script
  - `mid_train.py`: Mid-training for special tokens
  - `chat_sft.py`: Supervised finetuning
  - `chat_rl.py`: Reinforcement learning
  - `chat_web.py`: FastAPI web server with multi-GPU support

- **rustbpe/**: High-performance Rust BPE tokenizer implementation
- **datagen/**: Docker-based synthetic data generation service
- **tests/**: Test suite focusing on tokenizer correctness

### Key Technologies
- **PyTorch**: Core deep learning framework with DDP support
- **Rust**: High-performance tokenizer via PyO3 bindings
- **FastAPI**: Web serving with streaming responses
- **UV**: Modern Python dependency management
- **Maturin**: Rust/Python interop build tool
- **Celery**: Distributed task queue for data generation

### Data Flow
1. **Tokenization**: Rust BPE tokenizer processes text → tokens
2. **Pretraining**: Base model trained on large text corpus
3. **Mid-training**: Teach conversation special tokens
4. **Supervised Finetuning**: Domain adaptation with synthetic data
5. **DPO**: Direct preference optimization
6. **RL**: Optional reinforcement learning on GSM8K
7. **Inference**: Multi-GPU FastAPI server with worker pools

### External Dependencies
- **Datasets**: FineWeb, SmolTalk for training data
- **Compute**: Optimized for 8xH100 GPUs
- **Storage**: Data cached in `~/.cache/nanochat/`
- **Monitoring**: Wandb integration for logging

## Code Style Guidelines

### Python Style
- **Imports**: Standard library first, then third-party, then local
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Types**: Use type hints extensively (mypy compatible)
- **Docstrings**: Descriptive docstrings following Google style
- **Error Handling**: Use specific exceptions, log errors appropriately

### Code Patterns
```python
# Class definitions with type hints
@dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    autocast_ctx: torch.amp.autocast

# Function signatures with comprehensive type hints
def validate_chat_request(request: ChatRequest) -> None:
    """Validate chat request to prevent abuse."""

# Logging setup
logger = logging.getLogger(__name__)

# Error handling
try:
    # risky operation
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise
```

### Architecture Patterns
- **Distributed Training**: DDP with rank-based coordination
- **Worker Pools**: Multi-GPU inference with async worker management
- **Streaming**: Server-sent events for real-time text generation
- **Configuration**: Command-line args with sensible defaults
- **Checkpointing**: Automatic model saving/loading with metadata

### Testing Patterns
- **Pytest fixtures**: Module-scoped fixtures for expensive setup (dataset downloads)
- **Parametrized tests**: Multiple tokenizer implementations tested identically
- **Performance benchmarks**: Time-based assertions for training speed
- **Correctness verification**: Multiple implementations must produce identical results

### Performance Considerations
- **Memory**: Careful VRAM management with gradient accumulation
- **Precision**: BF16 autocast for inference, configurable precision
- **Parallelism**: Multi-GPU training/inference with proper synchronization
- **Batching**: Efficient batching for both training and inference

### File Organization
- **One class per file**: Core classes in separate files
- **Entry points in scripts/**: Command-line tools isolated from library code
- **Configuration**: Environment variables and CLI args, no config files
- **Caching**: Centralized cache directory for datasets and models
