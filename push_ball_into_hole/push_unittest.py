import unittest
import numpy as np
import os
from base import MujocoSimulation

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "pushxml", "push.xml")

class TestMujocoSimulation(unittest.TestCase):

    def setUp(self):
        # Set up the environment with all necessary arguments including model_path, initial_qpos, and n_substeps
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self.env = MujocoSimulation(
            model_path=MODEL_XML_PATH,          # The model path for the environment XML file
            initial_qpos=initial_qpos,          # Initial positions of joints
            n_substeps=20,                      # Number of substeps per simulation step
            block_end_effector=True,
            has_object=True,
            obj_range=0.2,
            target_range=0.2,
            distance_threshold=0.05,
            reward_type="dense"
        )

    def test_action_space(self):
        # Test the action space size and range
        action_space = self.env.action_space
        self.assertEqual(action_space.shape[0], 4)  # action space is 4-dimensional
        self.assertTrue((action_space.low == -1).all())  # lower bound is -1
        self.assertTrue((action_space.high == 1).all())  # upper bound is 1

    def test_observation_space(self):
        # Test the observation space
        obs = self.env._get_obs()
        self.assertIn('observation', obs)
        self.assertIn('achieved_goal', obs)
        self.assertIn('desired_goal', obs)
        self.assertEqual(len(obs['observation']), 25)  # length of observation vector

    def test_compute_reward(self):
        # Test the reward function
        achieved_goal = np.array([1.0, 1.0, 1.0])
        desired_goal = np.array([1.0, 1.0, 1.0])
        reward = self.env.compute_reward(achieved_goal, desired_goal, {})
        self.assertEqual(reward, 0)  # reward should be 0 if goals are the same

        # Test dense reward
        self.env.reward_type = 'dense'
        achieved_goal = np.array([1.0, 1.0, 1.0])
        desired_goal = np.array([2.0, 2.0, 2.0])
        reward = self.env.compute_reward(achieved_goal, desired_goal, {})
        self.assertLess(reward, 0)  # reward should be negative since goals are different

    def test_reset_sim(self):
        # Test if environment resets correctly
        result = self.env._reset_sim()
        self.assertTrue(result)

    def test_sample_goal(self):
        # Test goal sampling function
        goal = self.env._sample_goal()
        self.assertEqual(len(goal), 3)  # Goal should have 3 dimensions (x, y, z)

    def test_step_callback(self):
        # Ensure the step callback doesn't raise an exception
        try:
            self.env._step_callback()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result)

    def test_set_action(self):
        # Test setting an action
        action = np.array([0.1, 0.1, 0.1, 0.1])
        self.env._set_action(action)
        # No assertion needed; just make sure it doesn't raise an error


if __name__ == "__main__":
    unittest.main()


