import numpy as np

from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class PositionControlAviary(BaseRLAviary):
    """Single agent RL problem: Position control with 12D obs: rpy, vel, ang_vel, delta_xyz."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=np.array([[0.0, 0.0, 6.0]]).reshape(1, 3),
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 250,
                 ctrl_freq: int = 50,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.ATT_THR
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 10

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################

    def _observationSpace(self):
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo]*12], dtype=np.float32)
        obs_upper_bound = np.array([[hi]*12], dtype=np.float32)
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
    
    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]
        delta = self.TARGET_POS - pos

        obs = np.hstack([rpy, vel, ang_vel, delta]).astype(np.float32)
        return obs.reshape(1, 12)
    
    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        delta = self.TARGET_POS - pos

        e = float(np.linalg.norm(delta)) # ||e_k||_2

        a = 7.0
        sigma = 0.5

        eps = 1e-3
        r1 = 1.0 / (a * max(e, eps)) # avoid divide by zero
        r2 = a / (np.sqrt(2 * np.pi * (sigma**2))) * np.exp(-0.5 * ((e / sigma)**2))

        return float(r1 + r2)

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < 0.05:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """        
        max_ctrl_steps = int(self.EPISODE_LEN_SEC * self.CTRL_FREQ)
        cur_ctrl_steps = int(self.step_counter / self.PYB_STEPS_PER_CTRL)
        if cur_ctrl_steps >= max_ctrl_steps:
            return True


        state = self._getDroneStateVector(0)
        x, y, z = state[0:3]
        roll, pitch = state[7], state[8]

        init_x, init_y, init_z = self.INIT_POS
        
        # Define a bounding sphere around the initial position
        margin = 2.0
        bound = self.R + margin

        if (abs(x - init_x) > bound or abs(y - init_y) > bound or abs(z - init_z) > bound # Truncate when the drone is too far away
             or z < 0.05 # Truncate when the drone is too close to the ground
             or abs(roll) > 1.3 or abs(pitch) > 1.3 # Truncate when the drone is too tilted
        ):
            return True
        else:
            return False

    ################################################################################
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Cache initial pose from the sim, per episode
        self.INIT_POS = self._getDroneStateVector(0)[0:3].copy()

        ### SAMPLING INITIAL CONDITIONS ###
        v = self.np_random.normal(size=3) # random vector
        v = v / (np.linalg.norm(v) + 1e-12) # normalize to get a random direction

        self.R = 3.0
        r = self.R * (self.np_random.random() ** (1.0/3.0)) # Sample uniformly in a sphere of radius R

        delta = r * v # random vector with magnitude r

        self.TARGET_POS = self.INIT_POS + delta

        # avoid impossible target positions
        self.TARGET_POS[2] = max(self.TARGET_POS[2], 0.1) 

        # recompute obs 
        obs = self._computeObs()
        return obs, info

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
