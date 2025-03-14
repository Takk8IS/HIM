#!/bin/bash
set -e

# Configure Hugging Face credentials
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "Configuring Hugging Face credentials..."
    mkdir -p ~/.huggingface
    echo "{\"token\": \"$HUGGINGFACE_TOKEN\"}" > ~/.huggingface/token
fi

# Download DeepSeek model if not present
if [ ! -d "/app/models/deepseek-base" ]; then
    echo "Downloading DeepSeek model..."
    python -c "from transformers import AutoModel, AutoTokenizer; model = AutoModel.from_pretrained('deepseek-ai/deepseek-llm-7b-base', cache_dir='/app/models/deepseek-base'); tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-llm-7b-base', cache_dir='/app/models/deepseek-base')"
fi

# Run initialization for teleological system
python -c "from src.core.teleology.teleology_system import initialize_teleology; initialize_teleology()"

# Run initialization for semiotic system
python -c "from src.core.semiotics.semiotics_system import initialize_semiotics; initialize_semiotics()"

# Run initialization for pantheistic system
python -c "from src.core.pantheism.pantheism_system import initialize_pantheism; initialize_pantheism()"

# Check environment and run appropriate commands
if [ "$HIM_ENV" = "training" ]; then
    echo "Initializing training environment..."
    
    # Check if WANDB_API_KEY is set for tracking
    if [ -n "$WANDB_API_KEY" ]; then
        wandb login "$WANDB_API_KEY"
    fi
fi

# Execute the command
exec "$@"

