import os
import numpy as np
from gymnasium.utils.ezpickle import EzPickle  
from base import MujocoSimulation

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "pushxml", "push.xml")

class PushingBallEnv(MujocoSimulation, EzPickle):
    def __init__(self, reward_type="dense", **kwargs):
        """
        Initializes the PushingBall environment.
        Ensures Mujoco simulation is properly set up.
        """
        initial_qpos = {           
            "robot0:slide0": 0.405,  
            "robot0:slide1": 0.48,   
            "robot0:slide2": 0.0,    
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],  
        }

        # ✅ Call MujocoSimulation __init__ to initialize `self.sim`
        super().__init__(
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

        # ✅ Verify Mujoco Initialization
        if not hasattr(self, 'sim'):
            raise RuntimeError("Mujoco simulation (`self.sim`) was not initialized correctly!")

        EzPickle.__init__(self, reward_type=reward_type, **kwargs)


    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        Ensures Mujoco is initialized before applying domain randomization.
        """
        super().reset(seed=seed)
    
        # 🔥 Ensure Mujoco simulation is initialized before using it
        if not hasattr(self, 'sim') or self.sim is None:
            raise RuntimeError("Mujoco simulation (`self.sim`) is missing or not initialized!")
    
        # Reset Mujoco state
        self.sim.set_state(self.initial_state)
        self.sim.forward()
    
        # ✅ Apply domain randomization (if enabled)
        if self.ball_randomize_positions:
            noise = np.random.uniform(-0.03, 0.03, size=2)  
            self.sim.data.qpos[self.sim.model.joint_name2id("object0:joint")][:2] += noise
    
        if self.hole_randomize_positions:
            noise = np.random.uniform(-0.02, 0.02, size=2)
            self.sim.data.qpos[self.sim.model.joint_name2id("goal:joint")][:2] += noise
    
        # Step simulation forward
        self.sim.forward()
    
        # Get initial observation
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """
        Apply an action, step the simulation, and return the observation, reward, done, and info.
        """
        self._set_action(action)  # Apply action
        self.sim.step()  # Step simulation
    
        # Get new observation
        obs = self._get_obs()
        
        # Compute reward and check success condition
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        success = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]) < self.distance_threshold
    
        info = {"is_success": success}
        
        return obs, reward, False, False, info  # done is always False for continuous tasks
    
