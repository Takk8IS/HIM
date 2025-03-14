import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv

import torch
from huggingface_hub import HfApi, Repository
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class HIMHuggingFaceIntegration:
    """Integration class for Hugging Face Hub operations."""
    
    def __init__(self, model_name: str, token: Optional[str] = None):
        """Initialize the integration with model name and token."""
        self.model_name = model_name
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            raise ValueError("Hugging Face token not provided")
        
        self.api = HfApi(token=self.token)
        self.repo = None
        
    def login_to_hub(self) -> bool:
        """Login to Hugging Face Hub."""
        try:
            self.api.whoami()
            return True
        except Exception as e:
            logger.error(f"Failed to login: {e}")
            return False
            
    def create_model_repo(self) -> bool:
        """Create or clone the model repository."""
        try:
            self.repo = Repository(
                local_dir="./model_repo",
                clone_from=self.model_name,
                token=self.token,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create/clone repository: {e}")
            return False
            
    def upload_model_to_hub(self, commit_message: str = "Update model") -> bool:
        """Upload model files to Hugging Face Hub."""
        try:
            if not self.repo:
                return False
                
            self.repo.git_add()
            self.repo.git_commit(commit_message)
            self.repo.git_push()
            return True
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return False
            
    def setup_collaborative_training(self) -> Dict[str, Any]:
        """Configure collaborative training settings."""
        config = {
            "model_name": self.model_name,
            "training_type": "distributed",
            "framework": "pytorch",
            "max_train_samples": 10000,
            "eval_steps": 500,
            "save_steps": 1000,
            "push_to_hub": True
        }
        return config

def main():
    """Main function to demonstrate Hugging Face Hub integration."""
    # Get token from environment
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        logger.error("HUGGINGFACE_TOKEN environment variable not set")
        return
        
    # Initialize integration
    hub_integration = HIMHuggingFaceIntegration(
        model_name="TeleologyHI/HIM-self",
        token=token
    )
    
    if not hub_integration.login_to_hub():
        logger.error("Failed to login to Hugging Face Hub")
        return
        
    if not hub_integration.create_model_repo():
        logger.error("Failed to create repository")
        return
        
    if not hub_integration.upload_model_to_hub():
        logger.error("Failed to upload model")
        return
        
    training_config = hub_integration.setup_collaborative_training()
    logger.info(f"Training configuration: {training_config}")
    logger.info(f"Model available at: https://huggingface.co/{hub_integration.model_name}")

if __name__ == "__main__":
    main()

