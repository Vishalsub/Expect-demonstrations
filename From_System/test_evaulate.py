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
        super().reset(seed=seed, options=options)
        noise = np.random.uniform(-0.02, 0.02, size=2)  
        self.sim.data.qpos[self.sim.model.joint_name2id("object0:joint")][:2] += noise
        self.sim.forward()
        return self._get_obs(), {}

    def step(self, action):
        if np.random.rand() < 0.1:
            force = np.random.uniform(-0.02, 0.02, size=2)
            self.sim.data.qfrc_applied[self.sim.model.joint_name2id("object0:joint")][:2] += force

        self.sim.model.geom_friction[:] *= np.random.uniform(0.8, 1.2)
        if np.random.rand() < 0.05:
            self.sim.model.opt.gravity[:2] += np.random.uniform(-0.1, 0.1, size=2)

        obs, reward, done, truncated, info = super().step(action)
        return obs, reward, done, truncated, info
