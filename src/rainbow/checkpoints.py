import torch
import time

def save_checkpoint(
    online_model,
    seed,
    episodes,
    frame_count,
    average_reward,
    evaluation_mean_reward=0,
    start_time=0,
    config=None,
):
    checkpoint = {
    "model_state_dict": online_model.state_dict(),
    "seed": seed,
    "episodes": episodes + 1,
    "emulator_frames": frame_count,
    "last_interval_training_episode_average_reward": average_reward,
    "evaluation_mean_reward": 0,
    "training_duration_seconds": time.perf_counter() - start_time,
    "config": config,
}

    checkpoint_path = config["evaluate"]["checkpoint_path"]
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
