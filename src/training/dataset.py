import os
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, create_repo

def prepare_consciousness_dataset():
    # Load base prompts from config files
    teleology_prompts = []
    semiotics_prompts = []
    pantheism_prompts = []
    
    # Read prompts from config files
    with open('config/prompts/teleology.yaml', 'r') as f:
        teleology_data = yaml.safe_load(f)
        teleology_prompts = [p['prompt'] for p in teleology_data['base_prompts']]
        teleology_prompts.extend([p['prompt'] for p in teleology_data['advanced_prompts']])
    
    with open('config/prompts/semiotics.yaml', 'r') as f:
        semiotics_data = yaml.safe_load(f)
        semiotics_prompts = [p['prompt'] for p in semiotics_data['base_prompts']]
        semiotics_prompts.extend([p['prompt'] for p in semiotics_data['advanced_prompts']])
    
    with open('config/prompts/pantheism.yaml', 'r') as f:
        pantheism_data = yaml.safe_load(f)
        pantheism_prompts = [p['prompt'] for p in pantheism_data['base_prompts']]
        pantheism_prompts.extend([p['prompt'] for p in pantheism_data['advanced_prompts']])
    
    # Create dataset dictionary
    dataset_dict = {
        'prompt': teleology_prompts + semiotics_prompts + pantheism_prompts,
        'category': ['teleology'] * len(teleology_prompts) +
                  ['semiotics'] * len(semiotics_prompts) +
                  ['pantheism'] * len(pantheism_prompts)
    }
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Create dataset repository
    api = HfApi()
    try:
        create_repo('TeleologyHI/consciousness-dataset', repo_type='dataset')
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Push dataset to hub
    dataset.push_to_hub('TeleologyHI/consciousness-dataset')
    print("Dataset uploaded successfully")
    return dataset

if __name__ == "__main__":
    prepare_consciousness_dataset()

