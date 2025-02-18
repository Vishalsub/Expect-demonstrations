import os

from gymnasium.utils.ezpickle import EzPickle #Enables pickling and unpickling of the environment, facilitating serialization, which is helpful for saving and loading simulations.

from base import MujocoSimulation

# Use os.path.expanduser to handle the tilde (~) for the home directory
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "pushxml", "push.xml")



class PushingBallEnv(MujocoSimulation, EzPickle):
    def __init__(self, reward_type="dense", **kwargs):
        
        initial_qpos = {           
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],


            #From our xml files, robot0:slide0, robot0:slide1, robot0:slide2 refer to joints in the simulation that allow the robot's base or end_effector to move along xyz axes respectively.
            #Object0:joint refers to the joint of the object that the robot arm will interact with, and its value is [0, 0 ,0 ,0, 0, 0, 0], where the first 3 values are x y z position, and the last four values are orientation in quaternion format.

            #We have set the initial values above.
        }
        MujocoSimulation.__init__(
            self,
            model_path=MODEL_XML_PATH,  # Path to the XML model file that defines the environment setup and robot.

            has_object=True,  # Specifies whether the environment contains an object to interact with.

            block_end_effector=True,  # Restricts end-effector movement, ensuring it remains in the same orientation.

            n_substeps=20,  # Number of substeps per action, affecting the control frequency of the robot.

            obj_range=0.2,  # Range within which the object's initial position is randomly selected. Position is chosen randomly within a square region centered around the robot's end effector

            target_range=0.2,  # Range within which the target position is randomly selected. Position is chosen randomly within a square region centered around the robot's end effector

            distance_threshold=0.05,  # Distance from the goal at which the task is considered successfully completed.

            initial_qpos=initial_qpos,  # Dictionary specifying the initial joint positions and orientations of the robot and object.

            reward_type=reward_type,  # Defines the type of reward function to be used ('sparse' or 'dense').
            
            ball_randomize_positions = False, # to randomize the ball position

            hole_randomize_positions = False, #to randomize the hole position

            **kwargs  # Any additional arguments passed to the environment.

        )

        EzPickle.__init__(self, reward_type=reward_type, **kwargs)