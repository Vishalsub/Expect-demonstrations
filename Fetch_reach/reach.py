import os
import numpy as np
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.fetch import MujocoFetchEnv, MujocoPyFetchEnv

# Define the XML model path for the Mujoco simulation
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "reachxml", "reach.xml")


class MujocoFetchReachEnv(MujocoFetchEnv, EzPickle):
    """
    Custom Fetch Reach environment based on Mujoco.
    """

    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    def step(self, action):
        """
        Execute one step in the environment and ensure the reward is always a float.
        """
        # Unpack five outputs from Gymnasium's step()
        observation, reward, terminated, truncated, info = super().step(action)

        # Debugging: Print reward before conversion
        print(f"DEBUG: Reward type before conversion: {type(reward)}, value: {reward}")

        # Convert reward to float
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())  # Convert NumPy array to Python float
        else:
            reward = float(reward)  # Convert other types to float

        # Debugging: Print reward after conversion
        print(f"DEBUG: Reward type after conversion: {type(reward)}, value: {reward}")

        # Return all five values to be compatible with Gymnasium v0.26+
        return observation, reward, terminated, truncated, info


    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the reward based on goal distance and ensure it's always a float.
        """
        reward = -np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        # Convert to a single float if reward is an array
        if isinstance(reward, np.ndarray) and reward.shape == ():
            reward = float(reward.item())  # Handles single-value NumPy arrays
        elif isinstance(reward, np.ndarray):
            reward = reward.astype(float)  # Convert array to float if batch processing

        # Debugging: Print reward type
        print(f"DEBUG: Compute_reward output type: {type(reward)}, value: {reward}")

        return reward



class MujocoPyFetchReachEnv(MujocoPyFetchEnv, EzPickle):
    """
    Alternative implementation using MujocoPy.
    """

    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        MujocoPyFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    def step(self, action):
        """
        Execute one step in the environment and ensure the reward is always a float.
        """
        observation, reward, done, info = super().step(action)

        # Debugging: Print reward before conversion
        print(f"DEBUG: Reward type before conversion: {type(reward)}, value: {reward}")

        # Convert reward to float
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())  # Convert NumPy array to Python float
        else:
            reward = float(reward)  # Convert other types to float

        # Debugging: Print reward after conversion
        print(f"DEBUG: Reward type after conversion: {type(reward)}, value: {reward}")

        return observation, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute reward based on goal distance and ensure it's always a float.
        """
        reward = -np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        # Convert to float
        reward = float(reward)

        # Debugging: Print reward type
        print(f"DEBUG: Compute_reward output type: {type(reward)}, value: {reward}")

        return reward
