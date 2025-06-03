import unittest
import numpy as np
from optimizer import R2TextualGradientDescent
from variable import Variable

class MockEngine:
    def __call__(self, prompt, system_prompt=None):
        return "<NEW_VARIABLE>This is an optimized prompt</NEW_VARIABLE>"

class TestR2TextualGradientDescent(unittest.TestCase):
    def setUp(self):
        self.variable = Variable(
            value="Initial prompt text",
            role_description="Test prompt optimization"
        )
        self.optimizer = R2TextualGradientDescent(
            parameters=[self.variable],
            engine=MockEngine(),
            num_trials=3
        )

    def test_compute_r2_score(self):
        responses = [0.1, 0.4, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        r2_score = self.optimizer.compute_r2_score(responses, labels)
        self.assertGreaterEqual(r2_score, 0.0)
        self.assertLessEqual(r2_score, 1.0)

    def test_step_optimization(self):
        initial_value = self.variable.value
        self.optimizer.step()
        self.assertNotEqual(initial_value, self.variable.value)
        self.assertTrue(len(self.optimizer.r2_history) > 0)

    def test_gradient_memory(self):
        optimizer = R2TextualGradientDescent(
            parameters=[self.variable],
            engine=MockEngine(),
            gradient_memory=2
        )
        optimizer.step()
        optimizer.step()
        key = self.variable.get_role_description()
        self.assertEqual(len(optimizer.gradient_memory_dict[key]), 2)

if __name__ == '__main__':
    unittest.main()