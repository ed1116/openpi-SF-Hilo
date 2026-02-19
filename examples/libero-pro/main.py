import collections
import dataclasses
import importlib
import logging
import math
import pathlib
import sys
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import yaml

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
LIBERO_PRO_ROOT = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "LIBERO-PRO"  #[COPILOT] Libero-Pro root path.
MAX_STEPS_BY_PREFIX = {  #[COPILOT] Base max-step policy for LIBERO and LIBERO-PRO suite families.
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8001
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_object"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    # Path to save videos. If None, derived from task_suite_name in __post_init__.
    video_out_path: Optional[str] = None
    # Path to save per-episode log. If None, derived from task_suite_name in __post_init__.
    log_out_path: Optional[str] = None
    evaluation_config_path: str = str(LIBERO_PRO_ROOT / "evaluation_config.yaml")  #[COPILOT] Libero-Pro evaluation config.

    seed: int = 7  # Random Seed (for reproducibility)

    def __post_init__(self) -> None:
        if self.video_out_path is None:
            self.video_out_path = f"data/libero/videos/{self.task_suite_name}"
        if self.log_out_path is None:
            self.log_out_path = f"data/libero/logs/{self.task_suite_name}.txt"


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    #[COPILOT] Initialize benchmark registry first so suite-name resolution can validate against available suites.
    benchmark_dict = benchmark.get_benchmark_dict()
    args.task_suite_name = _resolve_suite_name_for_libero_pro(
        args.task_suite_name, args.evaluation_config_path, args.seed, benchmark_dict
    )  #[COPILOT] Expand to *_temp/*_lan/*_object/... as needed for Libero-Pro.
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Initialize LIBERO task suite
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    
    #[COPILOT] Prepare log file
    pathlib.Path(args.log_out_path).parent.mkdir(parents=True, exist_ok=True)

    log_file = pathlib.Path(args.log_out_path)
    log_file.write_text("")

    max_steps = _infer_max_steps(args.task_suite_name)  #[COPILOT] Support LIBERO-PRO variants by prefix.

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False  #[COPILOT] Ensure episode result is defined even if an exception occurs.
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            #[COPILOT] Log per-episode results
            with log_file.open("a") as f:
                f.write(
                    f"task_id={task_id} episode={episode_idx + 1} result={suffix}\n"
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    #[COPILOT] Log final results to file
    with log_file.open("a") as f:
        f.write(
            f"total_episodes={total_episodes} total_successes={total_successes} success_rate={float(total_successes) / float(total_episodes):.4f}\n"
        )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _infer_max_steps(task_suite_name: str) -> int:
    #[COPILOT] Reuse the same max-steps for all suite variants that share the same base family.
    for prefix, max_steps in MAX_STEPS_BY_PREFIX.items():
        if task_suite_name == prefix or task_suite_name.startswith(f"{prefix}_"):
            return max_steps
    raise ValueError(f"Unknown task suite for max_steps: {task_suite_name}")


def _resolve_suite_name_for_libero_pro(
    task_suite_name: str,
    evaluation_config_path: str,
    seed: int,
    benchmark_dict: dict,
) -> str:
    #[COPILOT] Resolve Libero-Pro suite name from evaluation flags; no-op when config is missing.
    config_path = pathlib.Path(evaluation_config_path)
    if not config_path.exists():
        logging.info("Libero-Pro evaluation config not found at %s; using task suite as-is.", config_path)
        return task_suite_name

    with config_path.open("r", encoding="utf-8") as f:
        evaluation_cfg = yaml.safe_load(f) or {}

    #[COPILOT] Normalize all config paths to absolute paths relative to LIBERO-PRO repo root.
    repo_root = config_path.parent
    evaluation_cfg["bddl_files_path"] = str(_resolve_cfg_path(evaluation_cfg.get("bddl_files_path", ""), repo_root))
    evaluation_cfg["init_file_dir"] = str(_resolve_cfg_path(evaluation_cfg.get("init_file_dir", ""), repo_root))
    evaluation_cfg["script_path"] = str(_resolve_cfg_path(evaluation_cfg.get("script_path", ""), repo_root))

    ood_cfg = evaluation_cfg.get("ood_task_configs", {})
    if isinstance(ood_cfg, dict):
        evaluation_cfg["ood_task_configs"] = {
            key: str(_resolve_cfg_path(value, repo_root)) for key, value in ood_cfg.items()
        }

    #[COPILOT] Set base suite and deterministic seed for perturbation generation.
    evaluation_cfg["task_suite_name"] = task_suite_name
    evaluation_cfg["seed"] = seed
    evaluation_cfg["bddl_files_path"] = str(pathlib.Path(evaluation_cfg["bddl_files_path"]) / task_suite_name)

    #[COPILOT] Inspect perturbation flags to select resolved suite.
    enabled_flags = [
        flag_name
        for flag_name in ("use_swap", "use_object", "use_language", "use_task", "use_environment")
        if bool(evaluation_cfg.get(flag_name, False))
    ]
    if not enabled_flags:
        return task_suite_name

    perturb_map = evaluation_cfg.get("perturbation_mapping", {}) or {}
    if len(enabled_flags) == 1:
        suffix = perturb_map.get(enabled_flags[0], "")
        if suffix:
            mapped_suite = f"{task_suite_name}_{suffix}"
            if mapped_suite in benchmark_dict:
                return mapped_suite
            logging.warning(
                "Mapped Libero-Pro suite %s is not registered. Falling back to temporary generated suite.",
                mapped_suite,
            )

    #[COPILOT] Multi-flag (or missing mapped suite) falls back to generated *_temp suite.
    temp_suite = f"{task_suite_name}_temp"
    if temp_suite not in benchmark_dict:
        raise ValueError(f"Expected Libero-Pro temp suite is not registered: {temp_suite}")

    _create_libero_pro_temp_env(evaluation_cfg)
    return temp_suite


def _resolve_cfg_path(path_str: str, base_dir: pathlib.Path) -> pathlib.Path:
    #[COPILOT] Resolve config-relative paths in evaluation_config.yaml.
    if not path_str:
        return base_dir
    path = pathlib.Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _create_libero_pro_temp_env(evaluation_cfg: dict) -> None:
    #[COPILOT] Import and execute LIBERO-PRO perturbation pipeline to materialize *_temp benchmark files.
    if str(LIBERO_PRO_ROOT) not in sys.path:
        sys.path.insert(0, str(LIBERO_PRO_ROOT))
    perturbation = importlib.import_module("perturbation")
    perturbation.create_env(configs=evaluation_cfg)


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
