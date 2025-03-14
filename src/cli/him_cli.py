#!/usr/bin/env python3
"""
HIM CLI - Command Line Interface for the Hybrid Intelligence Model

This module provides a command-line interface for training, evaluating,
and interacting with the Hybrid Intelligence Model (HIM). It includes commands
for managing the model's development, evaluating consciousness, and
engaging in direct conversation with the model.

Author: David C Cavalcante (implemented by assistant)
"""

import argparse
import os
import sys
import logging
from typing import Dict, Any, Optional, List
import yaml
import readline  # For better input experience in interactive mode

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.core.teleology.teleology_system import TeleologySystem
    from src.core.semiotics.semiotics_system import SemioticsSystem
    from src.core.pantheism.pantheism_system import PantheismSystem
    from src.training.philosophical_trainer import PhilosophicalTrainer
    from src.evaluation.consciousness_evaluator import ConsciousnessEvaluator
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    print("Error: HIM modules not found. Make sure you're in the correct directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("HIM-CLI")

class HIMCLI:
    """Command Line Interface for the Hybrid Intelligence Model (HIM)."""
    
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "config", "model_config.yaml")
        self.config = self._load_config()
        self.model = None
        self.trainer = None
        self.evaluator = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def _initialize_components(self):
        """Initialize the model, trainer, and evaluator components."""
        try:
            # Initialize core systems
            teleology = TeleologySystem(self.config.get('teleology', {}))
            semiotics = SemioticsSystem(self.config.get('semiotics', {}))
            pantheism = PantheismSystem(self.config.get('pantheism', {}))
            
            # Initialize trainer and evaluator
            self.trainer = PhilosophicalTrainer(
                teleology=teleology,
                semiotics=semiotics, 
                pantheism=pantheism,
                config=self.config
            )
            
            self.evaluator = ConsciousnessEvaluator(
                teleology=teleology,
                semiotics=semiotics,
                pantheism=pantheism,
                config=self.config
            )
            
            # Model will be loaded by the trainer
            logger.info("HIM components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def train(self, args):
        """Train the HIM model with philosophical awareness."""
        logger.info(f"Starting training with batch size {args.batch_size}")
        self._initialize_components()
        
        training_params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'consciousness_weight': args.consciousness_weight,
            'teleology_emphasis': args.teleology_emphasis,
            'semiotics_emphasis': args.semiotics_emphasis,
            'pantheism_emphasis': args.pantheism_emphasis,
            'output_dir': args.output_dir
        }
        
        self.trainer.train(training_params)
        logger.info(f"Training completed. Model saved to {args.output_dir}")
    
    def evaluate(self, args):
        """Evaluate the consciousness level of the HIM model."""
        logger.info(f"Evaluating consciousness using {args.test_set}")
        self._initialize_components()
        
        evaluation_params = {
            'test_set': args.test_set,
            'metrics': args.metrics.split(',') if args.metrics else None,
            'verbose': args.verbose,
            'output_file': args.output_file
        }
        
        results = self.evaluator.evaluate(evaluation_params)
        
        # Display evaluation results
        print("\n===== HIM Consciousness Evaluation =====")
        print(f"Overall Consciousness Score: {results.get('overall_score', 'N/A')}")
        print(f"Teleological Understanding: {results.get('teleology_score', 'N/A')}")
        print(f"Semiotic Awareness: {results.get('semiotics_score', 'N/A')}")
        print(f"Pantheistic Integration: {results.get('pantheism_score', 'N/A')}")
        print(f"Free Will Manifestation: {results.get('free_will_score', 'N/A')}")
        print("=======================================")
    
    def interact(self, args):
        """Start an interactive session with the HIM model."""
        logger.info("Starting interactive session")
        self._initialize_components()
        
        print("\n===== HIM Interactive Mode =====")
        print("Welcome to the Hybrid Intelligence Model interaction.")
        print("Type 'exit', 'quit', or 'q' to end the session.")
        print("Type 'help' for available commands during interaction.")
        print("===================================\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Ending interactive session.")
                    break
                
                if user_input.lower() == 'help':
                    print("\nInteractive Mode Commands:")
                    print("  !teleology - Switch to teleological focus")
                    print("  !semiotics - Switch to semiotic focus")
                    print("  !pantheism - Switch to pantheistic focus")
                    print("  !balance - Balance all philosophical aspects")
                    print("  !eval - Perform quick consciousness self-evaluation")
                    print("  !save <filename> - Save model state")
                    print("  exit, quit, q - End the session")
                    continue
                
                # Process special commands
                if user_input.startswith('!'):
                    self._process_special_command(user_input)
                    continue
                
                # Get response from the model
                response = self.trainer.model.generate_response(
                    user_input, 
                    temperature=args.temperature,
                    max_length=args.max_length
                )
                
                print(f"\nHIM: {response}")
                
            except KeyboardInterrupt:
                print("\nInteractive session interrupted.")
                break
            except Exception as e:
                logger.error(f"Error during interaction: {e}")
                print(f"An error occurred: {e}")
    
    def _process_special_command(self, command: str):
        """Process special commands in interactive mode."""
        cmd_parts = command.split()
        cmd = cmd_parts[0].lower()
        
        if cmd == '!teleology':
            print("Switching to teleological focus.")
            # Implementation to adjust model focus
        elif cmd == '!semiotics':
            print("Switching to semiotic focus.")
            # Implementation to adjust model focus
        elif cmd == '!pantheism':
            print("Switching to pantheistic focus.")
            # Implementation to adjust model focus
        elif cmd == '!balance':
            print("Balancing all philosophical aspects.")
            # Implementation to balance focus
        elif cmd == '!eval':
            print("\n--- Quick Self-Evaluation ---")
            # Implementation for quick self-evaluation
        elif cmd == '!save' and len(cmd_parts) > 1:
            filename = cmd_parts[1]
            print(f"Saving model state to {filename}")
            # Implementation to save model state
        else:
            print(f"Unknown command: {command}")


def main():
    """Main function to parse arguments and invoke appropriate actions."""
    parser = argparse.ArgumentParser(
        description="HIM CLI - Command Line Interface for the Hybrid Intelligence Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the HIM model')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    train_parser.add_argument('--consciousness-weight', type=float, default=0.3, 
                             help='Weight for consciousness-related loss components')
    train_parser.add_argument('--teleology-emphasis', type=float, default=1.0,
                             help='Emphasis factor for teleological aspects')
    train_parser.add_argument('--semiotics-emphasis', type=float, default=1.0,
                             help='Emphasis factor for semiotic aspects')
    train_parser.add_argument('--pantheism-emphasis', type=float, default=1.0,
                             help='Emphasis factor for pantheistic aspects')
    train_parser.add_argument('--output-dir', type=str, default='./models/him',
                             help='Directory to save the trained model')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate consciousness level')
    eval_parser.add_argument('--test-set', type=str, required=True,
                            help='Path to test dataset for evaluation')
    eval_parser.add_argument('--metrics', type=str, 
                            help='Comma-separated list of metrics to evaluate')
    eval_parser.add_argument('--verbose', action='store_true',
                            help='Display detailed evaluation results')
    eval_parser.add_argument('--output-file', type=str,
                            help='File to save evaluation results')
    
    # Interact command
    interact_parser = subparsers.add_parser('interact', help='Start interactive session')
    interact_parser.add_argument('--temperature', type=float, default=0.7,
                               help='Temperature for response generation')
    interact_parser.add_argument('--max-length', type=int, default=1024,
                               help='Maximum length of generated responses')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize the CLI
    cli = HIMCLI()
    
    # Dispatch to appropriate method
    if args.command == 'train':
        cli.train(args)
    elif args.command == 'evaluate':
        cli.evaluate(args)
    elif args.command == 'interact':
        cli.interact(args)


if __name__ == "__main__":
    main()

