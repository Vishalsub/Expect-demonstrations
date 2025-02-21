import os
import numpy as np
from gymnasium.utils.ezpickle import EzPickle  # Enables pickling/unpickling for easy environment saving/loading
from base import MujocoSimulation

# Define the XML model path for the Mujoco simulation
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "pushxml", "push.xml")


class PushingBallEnv(MujocoSimulation, EzPickle):
    def __init__(self, reward_type="dense", **kwargs):
        """
        Initializes the PushingBall environment.
        """
        initial_qpos = {           
            "robot0:slide0": 0.405,  # X-axis position
            "robot0:slide1": 0.48,   # Y-axis position
            "robot0:slide2": 0.0,    # Z-axis position
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],  # Object position & orientation
        }

        # Initialize Mujoco Simulation with parameters
        MujocoSimulation.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,  
            block_end_effector=True,  
            n_substeps=20,  
            obj_range=0.2,  
            target_range=0.2,  
            distance_threshold=0.05,  
            initial_qpos=initial_qpos,  
            reward_type=reward_type,  
            ball_randomize_positions=False,
            hole_randomize_positions=False,
            **kwargs
        )

        EzPickle.__init__(self, reward_type=reward_type, **kwargs)


def compute_reward(self, achieved_goal, desired_goal, info):
    """
    Computes the reward based on the distance between achieved and desired goals.
    Strengthens rewards for success and movement toward goal.
    """
    distance = np.linalg.norm(achieved_goal - desired_goal)

    # Stronger negative reward for distance
    reward = -distance * 30  

    # Bigger reward for reaching the goal
    if distance < 0.05:
        reward += 100  # Increased from 20
    elif distance < 0.1:
        reward += 50  # Increased from 10

    # Encourage movement toward the goal
    reward += 25 * (1 - distance)

    # Clip rewards to prevent extreme values
    reward = np.clip(reward, -50, 50)

    return np.array([reward], dtype=np.float32)
