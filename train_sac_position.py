"""Script to train a SAC agent in the PositionControlAviary environment.

Class PositionControlAviary is used as learning envs for the SAC algorithm.

Example
-------
In a terminal, run as:

    $ python train_sac_position.py
    $ python train_sac_position.py
"""
import os
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from gym_pybullet_drones.envs.PositionControlAviary import PositionControlAviary
from gym_pybullet_drones.utils.utils import  str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType.ATT_THR 

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    train_env = make_vec_env(PositionControlAviary,
                             env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                             n_envs=2,
                             seed=0
                             )

    eval_env = Monitor(PositionControlAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=False))
    eval_env.reset(seed=0)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    model = SAC("MlpPolicy",
                train_env,
                learning_rate=7e-4,
                buffer_size=1_000_000,
                learning_starts=10_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                verbose=1,
                device="cpu",
            )

    eval_callback = EvalCallback(eval_env,
                                #  callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(10000), # 10_000
                                 deterministic=True,
                                 render=False)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,              # every 50k env steps
        save_path=filename + "/ckpt/",
        name_prefix="sac_pos",
        save_replay_buffer=True,       # important for off-policy SAC
        save_vecnormalize=True         # only matters if you use VecNormalize
    )
    # timesteps: 2_750_000
    model.learn(total_timesteps=int(2_750_000), 
                callback=[eval_callback, checkpoint_callback],
                log_interval=100)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        timesteps = data['timesteps']
        results = data['results'][:, 0] 
        print("Data from evaluations.npz")
        for j in range(timesteps.shape[0]):
            print(f"{timesteps[j]},{results[j]}")
        if local:
            plt.plot(timesteps, results, marker='o', linestyle='-', markersize=4)
            plt.xlabel('Training Steps')
            plt.ylabel('Episode Reward')
            plt.grid(True, alpha=0.6)
            plt.show()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Train a SAC agent in the PositionControlAviary environment.')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
