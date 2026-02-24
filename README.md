# SAC_drone_control

A project that trains and tests a **Soft Actor-Critic (SAC)** policy for **quadcopter control** in **PyBullet**.

This repo intentionally contains only 3 files:

- `PositionControlAviary.py` — custom Gymnasium environment (obs + reward + termination/truncation + target sampling)
- `train_sac_position.py` — SAC training script (SB3)
- `play.py` — playback script to visualize a trained policy

✅ **Important:** This project depends on a **modified fork** (by me) of [`gym-pybullet-drones`](https://github.com/esatiyev/gym-pybullet-drones) (controller + enums + action type changes).  
You must install that fork first, then run this project.

---
Demo: 
<!-- ![demo](https://github.com/user-attachments/assets/8ef3dd0c-ed58-4fc0-98e7-b9ee6303443d) -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/8ef3dd0c-ed58-4fc0-98e7-b9ee6303443d" width="600"/>
</p>

---

## What this project does

### Observation (12D)
`[ roll, pitch, yaw, vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z, Δx, Δy, Δz ]`

where `Δ = target_position - drone_position`.

### Action (ATT_THR)
Policy outputs **3 values** in `[-1, 1]`:

`[ U, φ_des, θ_des ]`

- yaw is held at the current yaw  
- `U` maps to a **PWM-like thrust command** expected by the underlying PID controller  
- desired attitude + thrust are converted to motor RPMs via the PID inner loop  

---

## Installation

### 1) Install the modified `gym-pybullet-drones` fork (required)

```bash
git clone https://github.com/esatiyev/gym-pybullet-drones.git
cd gym-pybullet-drones

conda create -n drones python=3.10 -y
conda activate drones

pip install --upgrade pip
pip install -e .
```

### 2) Clone this repo

```bash
cd ..
git clone https://github.com/esatiyev/SAC_drone_control.git
cd SAC_drone_control
conda activate drones
```

---

## Train

```bash
python train_sac_position.py
```

Artifacts are saved under:

```
results/save-<timestamp>/
  best_model.zip
  final_model.zip
  evaluations.npz
  ckpt/
    sac_pos_<N>_steps.zip
    sac_pos_replay_buffer_<N>_steps.pkl
```

> Off-policy SAC resumes best when you load both the checkpoint `.zip` **and** the replay buffer `.pkl`.

---

## Play / visualize

```bash
python play.py --model_path results/save-<timestamp>/best_model.zip --gui True
```

---

## Notes

- The modified simulator fork contains changes to:
  - `BaseAviary.py`
  - `BaseRLAviary.py`
  - `DSLPIDControl.py`
  - `enums.py`
