import numpy as np

from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations


#CHANGE CAMERA ANGLE FROM HERE

DEFAULT_CAMERA_CONFIG = {
    "distance": 3,
    "azimuth": 120,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}


def get_base_env(RobotEnvClass: MujocoRobotEnv):
   #This function returns a class that takes its structure from the MujocoRobotEnv class which allows us to make robotic environments using the Mujoco Simulator.
   #Below is the class that will set up our framework for the 7-DOF robot we are using which will have a hollow half-cylinder end effector that will be able to push a ball into a hole.

    class PushEnv(RobotEnvClass):
        
        def __init__(
            self,
            block_end_effector,
            has_object: bool,
            obj_range,
            target_range,
            distance_threshold,
            reward_type,
            ball_randomize_positions = False,
            hole_randomize_positions = False,
            **kwargs,
        ):
            # Initializes our environment with the following parameters:

            # Args:
            #     model_path - : path to the environments XML file (from the RobotEnv class)

            #     n_substeps - : number of substeps the simulation runs on every call to step (from the RobotEnv class). This will affect the control frequency of the robot, as the same action will be run in one step for n_substeps times.

            #     initial_qpos - a dictionary of joint names and values that define the initial configuration (from the RobotEnv class)

            #     block_end_effector (boolean): our end-effector does not move at all and will always be the same orientation. As of now it is a half-hollow-cylinder that will push the ball into the hole.
            #     This is left as a boolean if we decide to change our end-effector

            #     has_object : whether or not the environment has an object

            #     obj_range : The range within which the objects positions are randomly chosen
            #     target_range : The range within which the target position is randomly chosen

            #     distance_threshold: Used in the reward function to determine how how close to the goal is considered successful
            
            #     reward_type ('sparse' or 'dense'): Our initial idea is to use a dense reward function because we want to reduce the number of steps it takes to push the ball into the hole,
            #     but sparse reward is also being considered
            self.ball_randomize_positions = ball_randomize_positions  
            self.hole_randomize_positions = hole_randomize_positions
            self.block_end_effector = block_end_effector
            self.has_object = has_object
            self.obj_range = obj_range
            self.target_range = target_range
            self.distance_threshold = distance_threshold
            self.reward_type = reward_type

            super().__init__(n_actions=4, **kwargs) #The action space is defined in the MujocoRobotEnv class, and we have chosen a 4-dimensional vector where the first 3 dimensions will be the position of the end effector and the 4th dimension is the end_effector control.

            ###--> self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")


        #REWARD FUNCTION    


        def compute_reward(self, achieved_goal, goal, info):
            # The Euclidean Distance between the achieved goal and the target goal
            d = np.linalg.norm(achieved_goal - goal, axis=-1)
            
            if self.reward_type == "dense":
                return -d                
            else:
               return -(d > self.distance_threshold).astype(np.float32) # Returns 0 if the distance is less than the threshold, -1 otherwise . This is for making it a sparse reward structure, however for ease of training time we will use dense as default.
        
        #ACTION FUNCTION

        def _set_action(self, action):
            #As mentioned before, the action space is a Box(-1.0, 1.0, shape=(4)), where an action represents the cartesion movement dx dy dz of the end effector, and last action that controls opening and closing of our end effector.
            # Min -1 Max 1
            assert action.shape == (4,)
            action = (
                action.copy()
            )  # ensure that we don't change the action outside of this scope
            pos_ctrl, end_effector_ctrl = action[:3], action[3]

            pos_ctrl *= 0.05  # limit maximum change in position
            rot_ctrl = [
                1.0,
                0.0,
                1.0,
                0.0,
            ]  # fixed rotation of the end effector, expressed as a quaternion
            end_effector_ctrl = np.array([end_effector_ctrl, end_effector_ctrl])
            assert end_effector_ctrl.shape == (2,)
            if self.block_end_effector:
                end_effector_ctrl = np.zeros_like(end_effector_ctrl) #We are not controlling the end effector at all for our task, just a pushing action thus this
            action = np.concatenate([pos_ctrl, rot_ctrl, end_effector_ctrl])

            return action

        def _get_obs(self):
            (
                end_effector_pos, # x y z position of the end effector in world coordinates
                object_pos, # ball x y z position in world coordinates
                object_rel_pos, # x y z Relative position of ball to end effector
                end_effector_state, # displacement of left and right cylinder end effector
                object_rot, # x y z rotation of ball in Euler frame rotation
                object_velp, # x y z ball linear velocity with respect to end effector
                object_velr, #x y z axis ball angular velocity
                grip_velp, # velocity of left and right cylinder end effector
                end_effector_vel, # x y z end effector velocity 

                #All the observations that we want to recieve from the environment

            ) = self.generate_mujoco_observations()

            if not self.has_object:
                achieved_goal = end_effector_pos.copy() # Represents where the ball is pushed till, x y z position in world coordinates.
            else:
                achieved_goal = np.squeeze(object_pos.copy())

            obs = np.concatenate(
                [
                    end_effector_pos,
                    object_pos.ravel(),
                    object_rel_pos.ravel(),
                    end_effector_state,
                    object_rot.ravel(),
                    object_velp.ravel(),
                    object_velr.ravel(),
                    grip_velp,
                    end_effector_vel,
                ]
            )

            return {
                "observation": obs.copy(), 
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.copy(), #Represents the final goal to be achived, x y z position in world coordinates. 
            }

        def generate_mujoco_observations(self):

            raise NotImplementedError

        def _get_gripper_xpos(self):

            raise NotImplementedError

        def _sample_goal(self):
            if self.hole_randomize_positions:
                # Randomized goal position
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
                goal[2] = self.height_offset if self.has_object else goal[2]
            else:
                # Fixed goal position
                goal = self.initial_gripper_xpos[:3]+np.array([0.3, 0.2, 0])  # Replace with your desired fixed position

            return goal.copy()

        def _is_success(self, achieved_goal, desired_goal):
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return (d < self.distance_threshold).astype(np.float32)
        
             #This function determines whether the task goal has been achieved by comparing the achieved_goal with the desired_goal:

    return PushEnv

class MujocoSimulation(get_base_env(MujocoRobotEnv)):


    #This is the simulation class which has all the Mujoco bindings to show the actual robot arm and the ball moving in the mujoco simulator


    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)

    def _step_callback(self):
        if self.block_end_effector:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_end_effector_joint", 0.0
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_end_effector_joint", 0.0
            )
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self):
        # positions
        end_effector_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            )
            object_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - end_effector_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        end_effector_state = robot_qpos[-2:]

        end_effector_vel = (
            robot_qvel[-2:] * dt
        ) 

        return (
            end_effector_pos,
            object_pos,
            object_rel_pos,
            end_effector_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            end_effector_vel,
        )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Set object (ball) position
        if self.has_object:
            if self.ball_randomize_positions:
                # Randomized position
                object_xpos = self.initial_gripper_xpos[:2]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                        -self.obj_range, self.obj_range, size=2
                    )
            else:
                # Fixed position
                object_xpos = self.initial_gripper_xpos[:2] + np.array([0.1, 0.1] ) # Replace with your desired fixed position

            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431]
        ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", gripper_rotation
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0"
            )[2]