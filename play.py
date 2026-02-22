import os
import time
import argparse
import numpy as np
import pybullet as p

from stable_baselines3 import SAC

from gym_pybullet_drones.envs.PositionControlAviary import PositionControlAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool


def spawn_target_marker(env, radius=0.06):
    vid = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[1, 0, 0, 1],
        physicsClientId=env.CLIENT
    )
    bid = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vid,
        basePosition=env.TARGET_POS.tolist(),
        physicsClientId=env.CLIENT
    )
    return bid


def update_target_marker(env, marker_id):
    p.resetBasePositionAndOrientation(
        marker_id,
        env.TARGET_POS.tolist(),
        [0, 0, 0, 1],
        physicsClientId=env.CLIENT
    )


def run_episode(model, env, seed, max_ctrl_steps=502, show=True, marker=True, success_radius=0.10):
    obs, info = env.reset(seed=seed, options={})

    marker_id = None
    if show and marker:
        marker_id = spawn_target_marker(env)

    start = time.time()
    strict_success = False
    min_dist = 1e9

    for i in range(max_ctrl_steps):
        if show:
            if marker_id is not None:
                update_target_marker(env, marker_id)
            env.render()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        pos = env._getDroneStateVector(0)[0:3]
        dist = float(np.linalg.norm(env.TARGET_POS - pos))
        if dist < min_dist:
            min_dist = dist

        if terminated:
            strict_success = True
            break
        if truncated:
            break

        if show:
            sync(i, start, env.CTRL_TIMESTEP)

    practical_success = (min_dist <= success_radius)
    steps = i + 1
    return strict_success, practical_success, steps, min_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gui", type=str2bool, default=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--scenario", type=str, default="random",
                        choices=["random", "fixed_target", "stress"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success_radius", type=float, default=0.10,
                        help="Practical success if min distance to target <= this (meters).")
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"[ERROR] model not found: {args.model_path}")
        return

    model = SAC.load(args.model_path)
    print(f"[INFO] Loaded: {args.model_path}")

    env = PositionControlAviary(
        gui=args.gui,
        obs=ObservationType.KIN,
        act=ActionType.ATT_THR,
        pyb_freq=250,
        ctrl_freq=50
    )

    strict_ok = 0
    practical_ok = 0
    lengths = []
    min_dists = []

    for ep in range(args.episodes):
        seed = args.seed + ep

        if args.scenario == "random":
            strict_s, practical_s, steps, min_dist = run_episode(
                model, env, seed=seed, show=args.gui, marker=True,
                success_radius=args.success_radius
            )

        elif args.scenario == "fixed_target":
            obs, info = env.reset(seed=seed, options={})
            env.TARGET_POS = np.array([2.0, 1.0, 4.5], dtype=float)
            obs = env._computeObs()

            marker_id = spawn_target_marker(env) if args.gui else None
            start = time.time()
            strict_s = False
            min_dist = 1e9

            for i in range(502):
                if args.gui:
                    if marker_id is not None:
                        update_target_marker(env, marker_id)
                    env.render()

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                pos = env._getDroneStateVector(0)[0:3]
                d = float(np.linalg.norm(env.TARGET_POS - pos))
                min_dist = min(min_dist, d)

                if terminated:
                    strict_s = True
                    break
                if truncated:
                    break

                if args.gui:
                    sync(i, start, env.CTRL_TIMESTEP)

            steps = i + 1
            practical_s = (min_dist <= args.success_radius)

        else:  # stress
            obs, info = env.reset(seed=seed, options={})
            init = env.INIT_POS.copy()
            env.TARGET_POS = init + np.array([2.8, 0.0, 0.2], dtype=float)
            env.TARGET_POS[2] = max(env.TARGET_POS[2], 0.1)
            obs = env._computeObs()

            marker_id = spawn_target_marker(env) if args.gui else None
            start = time.time()
            strict_s = False
            min_dist = 1e9

            for i in range(502):
                if args.gui:
                    if marker_id is not None:
                        update_target_marker(env, marker_id)
                    env.render()

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                pos = env._getDroneStateVector(0)[0:3]
                d = float(np.linalg.norm(env.TARGET_POS - pos))
                min_dist = min(min_dist, d)

                if terminated:
                    strict_s = True
                    break
                if truncated:
                    break

                if args.gui:
                    sync(i, start, env.CTRL_TIMESTEP)

            steps = i + 1
            practical_s = (min_dist <= args.success_radius)

        strict_ok += int(strict_s)
        practical_ok += int(practical_s)
        lengths.append(steps)
        min_dists.append(min_dist)

        print(f"ep {ep:03d} seed={seed} strict={strict_s} practical={practical_s} "
              f"min_dist={min_dist:.3f} steps={steps}")

        if args.gui:
            time.sleep(0.15)

    env.close()

    print("\n=== RESULTS ===")
    print(f"scenario: {args.scenario}")
    print(f"episodes: {args.episodes}")
    print(f"strict success (terminated): {strict_ok/args.episodes:.2%}")
    print(f"practical success (min_dist <= {args.success_radius}m): {practical_ok/args.episodes:.2%}")
    print(f"avg min_dist: {np.mean(min_dists):.3f}  median: {np.median(min_dists):.3f}")
    print(f"avg steps: {np.mean(lengths):.1f}  median: {np.median(lengths):.1f}  max: {np.max(lengths)}")


if __name__ == "__main__":
    main()
