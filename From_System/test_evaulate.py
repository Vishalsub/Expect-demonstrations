import numpy as np
from gymnasium.utils.ezpickle import EzPickle  
from base import MujocoSimulation

class PushingBallEnv(MujocoSimulation, EzPickle):
    def __init__(self, reward_type="dense", **kwargs):
        initial_qpos = {           
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],  
        }

        MujocoSimulation.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,  
            block_end_effector=True,  
            n_substeps=20,  
            obj_range=0.3,  
            target_range=0.3,  
            distance_threshold=0.05,  
            initial_qpos=initial_qpos,  
            reward_type=reward_type,  
            ball_randomize_positions=True,
            hole_randomize_positions=True,
            **kwargs
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        Apply domain randomization if enabled.
        """
        super().reset(seed=seed)
    
        # Ensure Mujoco simulation is initialized
        if not hasattr(self, 'sim'):
            raise AttributeError("Mujoco simulation (`self.sim`) is not initialized!")
    
        # Reset state
        self.sim.set_state(self.initial_state)
        self.sim.forward()
    
        # Apply domain randomization (if enabled)
        if self.ball_randomize_positions:
            noise = np.random.uniform(-0.03, 0.03, size=2)  # Small perturbations
            self.sim.data.qpos[self.sim.model.joint_name2id("object0:joint")][:2] += noise
    
        if self.hole_randomize_positions:
            noise = np.random.uniform(-0.02, 0.02, size=2)
            self.sim.data.qpos[self.sim.model.joint_name2id("goal:joint")][:2] += noise
    
        # Step forward
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
    
