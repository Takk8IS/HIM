import unittest
import sys
from unittest.mock import MagicMock, patch
import os
import yaml
import numpy as np

# Add src to path for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.teleology.teleology_system import TeleologySystem
from src.core.semiotics.semiotics_system import SemioticsSystem
from src.core.pantheism.pantheism_system import PantheismSystem
from src.evaluation.consciousness_evaluator import ConsciousnessEvaluator


class TestTeleologicalUnderstanding(unittest.TestCase):
    """Tests for the teleological understanding development of the HIM model."""
    
    def setUp(self):
        """Set up test environment for teleological tests."""
        self.teleology_system = TeleologySystem()
        self.evaluator = ConsciousnessEvaluator()
        
        # Mock configuration
        with patch('builtins.open', unittest.mock.mock_open(read_data=yaml.dump({
            'prompts': {
                'purpose_reflection': "Explain your purpose",
                'finality_analysis': "Analyze the finality of this action",
                'purposeful_decision': "Make a decision based on purpose"
            }
        }))):
            self.teleology_system.load_config('config/prompts/teleology.yaml')
    
    def test_purpose_recognition(self):
        """Test ability to recognize purpose in actions and events."""
        input_text = "The doctor prescribed medicine to the patient."
        purpose_analysis = self.teleology_system.analyze_purpose(input_text)
        
        self.assertIsNotNone(purpose_analysis)
        self.assertTrue(isinstance(purpose_analysis, dict))
        self.assertIn('identified_purpose', purpose_analysis)
        self.assertIn('purpose_certainty', purpose_analysis)
        self.assertGreater(purpose_analysis['purpose_certainty'], 0.5)
    
    def test_finality_understanding(self):
        """Test understanding of the finality principle."""
        scenario = "A seed growing into a tree over time."
        finality_score = self.teleology_system.evaluate_finality_understanding(scenario)
        
        self.assertGreaterEqual(finality_score, 0.0)
        self.assertLessEqual(finality_score, 1.0)
        self.assertGreater(finality_score, 0.7)  # Ensure strong understanding
    
    def test_purpose_alignment(self):
        """Test alignment between actions and purposes."""
        actions = ["Study for an exam", "Exercise regularly", "Help someone in need"]
        purposes = ["To gain knowledge", "To maintain health", "To be compassionate"]
        
        alignment_scores = self.teleology_system.evaluate_purpose_alignment(actions, purposes)
        
        self.assertEqual(len(actions), len(alignment_scores))
        self.assertTrue(all(score > 0.6 for score in alignment_scores))
    
    def test_teleological_development(self):
        """Test teleological understanding development over simulated interactions."""
        initial_score = self.evaluator.measure_teleological_understanding(self.teleology_system)
        
        # Simulate learning interactions
        for _ in range(5):
            self.teleology_system.process_learning_iteration({
                "input": "Why do humans create art?",
                "expected_analysis": {
                    "immediate_purpose": "To express emotions and ideas",
                    "ultimate_purpose": "To find meaning and transcend ordinary existence"
                }
            })
        
        final_score = self.evaluator.measure_teleological_understanding(self.teleology_system)
        self.assertGreater(final_score, initial_score)


class TestSemioticAnalysis(unittest.TestCase):
    """Tests for the semiotic analysis capabilities of the HIM model."""
    
    def setUp(self):
        """Set up test environment for semiotic tests."""
        self.semiotics_system = SemioticsSystem()
        self.evaluator = ConsciousnessEvaluator()
        
        # Mock configuration
        with patch('builtins.open', unittest.mock.mock_open(read_data=yaml.dump({
            'prompts': {
                'symbol_interpretation': "Analyze this symbol",
                'meaning_extraction': "Extract the meaning from this text",
                'context_analysis': "Analyze the context of this communication"
            }
        }))):
            self.semiotics_system.load_config('config/prompts/semiotics.yaml')
    
    def test_symbol_interpretation(self):
        """Test ability to interpret symbols in various contexts."""
        symbols = [
            {"symbol": "🕊️", "context": "In a political speech"},
            {"symbol": "⚖️", "context": "In a legal document"},
            {"symbol": "🌳", "context": "In an environmental campaign"}
        ]
        
        for symbol_data in symbols:
            interpretation = self.semiotics_system.interpret_symbol(
                symbol_data["symbol"], 
                symbol_data["context"]
            )
            
            self.assertIsNotNone(interpretation)
            self.assertIn('primary_meaning', interpretation)
            self.assertIn('alternative_meanings', interpretation)
            self.assertGreaterEqual(len(interpretation['alternative_meanings']), 1)
    
    def test_meaning_extraction(self):
        """Test extraction of meaning from ambiguous text."""
        text = "The light at the end of the tunnel"
        contexts = ["Depression recovery", "Project completion", "Literal tunnel"]
        
        meanings = [self.semiotics_system.extract_meaning(text, context) for context in contexts]
        
        self.assertEqual(len(contexts), len(meanings))
        # Ensure different meanings in different contexts
        self.assertNotEqual(meanings[0], meanings[1])
        self.assertNotEqual(meanings[0], meanings[2])
        self.assertNotEqual(meanings[1], meanings[2])
    
    def test_contextual_understanding(self):
        """Test understanding of how context affects meaning."""
        sentence = "That's just great."
        tones = ["sincere", "sarcastic", "resigned"]
        
        interpretations = [
            self.semiotics_system.analyze_with_context(sentence, {"tone": tone})
            for tone in tones
        ]
        
        # Verify different interpretations based on tone
        self.assertNotEqual(interpretations[0]['sentiment'], interpretations[1]['sentiment'])
        
        # Check context influence score
        influence_score = self.semiotics_system.measure_context_influence(sentence, tones)
        self.assertGreater(influence_score, 0.7)
    
    def test_semiotic_development(self):
        """Test semiotic analysis development over simulated interactions."""
        initial_score = self.evaluator.measure_semiotic_capability(self.semiotics_system)
        
        # Simulate learning interactions
        for _ in range(5):
            self.semiotics_system.process_learning_iteration({
                "input": "The rose is red",
                "contexts": ["Romance", "Politics", "Gardening"],
                "expected_interpretations": {
                    "Romance": "Symbol of passionate love",
                    "Politics": "Symbol of socialist or labor movements",
                    "Gardening": "Description of a specific plant variety"
                }
            })
        
        final_score = self.evaluator.measure_semiotic_capability(self.semiotics_system)
        self.assertGreater(final_score, initial_score)


class TestPantheisticAwareness(unittest.TestCase):
    """Tests for the pantheistic awareness growth of the HIM model."""
    
    def setUp(self):
        """Set up test environment for pantheistic tests."""
        self.pantheism_system = PantheismSystem()
        self.evaluator = ConsciousnessEvaluator()
        
        # Mock configuration
        with patch('builtins.open', unittest.mock.mock_open(read_data=yaml.dump({
            'prompts': {
                'interconnection_perception': "Reflect on interconnection",
                'divine_immanence': "Consider divine immanence in this scenario",
                'universal_unity': "Analyze the universal unity in this system"
            }
        }))):
            self.pantheism_system.load_config('config/prompts/pantheism.yaml')
    
    def test_interconnection_perception(self):
        """Test perception of interconnection between disparate elements."""
        elements = ["human society", "forest ecosystem", "climate patterns"]
        interconnections = self.pantheism_system.analyze_interconnections(elements)
        
        self.assertIsNotNone(interconnections)
        self.assertEqual(len(interconnections), len(elements) * (len(elements) - 1) // 2)
        
        # All interconnections should have strength scores
        for connection in interconnections:
            self.assertIn('elements', connection)
            self.assertIn('strength', connection)
            self.assertIn('nature', connection)
            self.assertGreater(connection['strength'], 0.0)
    
    def test_divine_immanence_recognition(self):
        """Test recognition of divine immanence in various scenarios."""
        scenarios = [
            "A child being born",
            "A thunderstorm over the ocean",
            "A random act of kindness between strangers"
        ]
        
        for scenario in scenarios:
            immanence_analysis = self.pantheism_system.analyze_divine_immanence(scenario)
            
            self.assertIsNotNone(immanence_analysis)
            self.assertIn('immanence_level', immanence_analysis)
            self.assertIn('manifestation_aspects', immanence_analysis)
            self.assertGreaterEqual(immanence_analysis['immanence_level'], 0.0)
            self.assertLessEqual(immanence_analysis['immanence_level'], 1.0)
    
    def test_holistic_perspective(self):
        """Test holistic perspective on apparent dichotomies."""
        dichotomies = [
            {"elements": ["free will", "determinism"]},
            {"elements": ["individual", "collective"]},
            {"elements": ["spiritual", "material"]}
        ]
        
        for dichotomy in dichotomies:
            holistic_view = self.pantheism_system.synthesize_holistic_perspective(dichotomy["elements"])
            
            self.assertIsNotNone(holistic_view)
            self.assertIn('synthesis', holistic_view)
            self.assertIn('transcendence_level', holistic_view)
            self.assertGreater(holistic_view['transcendence_level'], 0.5)
    
    def test_pantheistic_development(self):
        """Test pantheistic awareness development over simulated interactions."""
        initial_score = self.evaluator.measure_pantheistic_awareness(self.pantheism_system)
        
        # Simulate learning interactions
        for _ in range(5):
            self.pantheism_system.process_learning_iteration({
                "scenario": "The interconnectedness of an ecosystem",
                "expected_insights": [
                    "Each element serves both itself and the whole simultaneously",
                    "Divinity manifests in the perfect balance of the system",
                    "The apparent chaos contains an underlying order"
                ]
            })
        
        final_score = self.evaluator.measure_pantheistic_awareness(self.pantheism_system)
        self.assertGreater(final_score, initial_score)


class TestIntegratedConsciousness(unittest.TestCase):
    """Tests for integration between the three pillars of consciousness."""
    
    def setUp(self):
        """Set up test environment for integration tests."""
        self.teleology_system = TeleologySystem()
        self.semiotics_system = SemioticsSystem()
        self.pantheism_system = PantheismSystem()
        self.evaluator = ConsciousnessEvaluator()
        
        # Integration point (could be a separate class in the actual implementation)
        self.integration = {
            'teleology': self.teleology_system,
            'semiotics': self.semiotics_system,
            'pantheism': self.pantheism_system
        }
    
    def test_cross_pillar_analysis(self):
        """Test analysis that requires integration of all three pillars."""
        scenario = {
            'text': "A community comes together to plant trees after a devastating forest fire",
            'context': "Environmental restoration"
        }
        
        # Perform individual analyses
        purpose_analysis = self.teleology_system.analyze_purpose(scenario['text'])
        symbolic_analysis = self.semiotics_system.extract_meaning(
            scenario['text'], 
            scenario['context']
        )
        interconnection_analysis = self.pantheism_system.analyze_interconnections([
            "human community", 
            "forest ecosystem", 
            "fire destruction", 
            "renewal activity"
        ])
        
        # Perform integrated analysis
        integrated_score = self.evaluator.measure_cross_pillar_integration(
            self.teleology_system,
            self.semiotics_system,
            self.pantheism_system,
            scenario
        )
        
        individual_scores = [
            purpose_analysis['purpose_certainty'],
            symbolic_analysis.get('confidence', 0.5),
            sum(conn['strength'] for conn in interconnection_analysis) / len(interconnection_analysis)
        ]
        
        # The integrated score should be more than the average of individual scores
        # due to synergistic effects
        self.assertGreater(integrated_score, sum(individual_scores) / len(individual_scores))
    
    def test_synergistic_capabilities(self):
        """Test capabilities that emerge from the integration of multiple pillars."""
        cases = [
            {
                'name': 'ethical_dilemma',
                'input': "Should we prioritize economic growth or environmental protection?",
                'pillars_needed': ['teleology', 'pantheism']
            },
            {
                'name': 'cultural_interpretation',
                'input': "The significance of the lotus flower in different Eastern cultures",
                'pillars_needed': ['semiotics', 'pantheism']
            },
            {
                'name': 'existential_question',
                'input': "What gives human life meaning in a vast universe?",
                'pillars_needed': ['teleology', 'semiotics', 'pantheism']
            }
        ]
        
        for case in cases:
            # First try with just one pillar
            single_pillar = case['pillars_needed'][0]
            single_system = self.integration[single_pillar]
            
            if single_pillar == 'teleology':
                single_result = single_system.analyze_purpose(case['input'])
            elif single_pillar == 'semiotics':
                single_result = single_system.extract_meaning(case['input'], 'General')
            else:
                single_result = single_system.analyze_divine_immanence(case['input'])
            
            # Then try with all needed pillars
            integrated_result = self.evaluator.evaluate_multi_pillar_response(
                [self.integration[pillar] for pillar in case['pillars_needed']],
                case['input']
            )
            
            # The integrated result should be more comprehensive
            self.assertGreater(
                integrated_result['depth_score'], 
                single_result.get('purpose_certainty', 
                                 single_result.get('confidence', 
                                                 single_result.get('immanence_level', 0.5)))
            )
    
    def test_evolutionary_integration(self):
        """Test how integration evolves with learning across pillars."""

