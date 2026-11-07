import torch
import time

def save_checkpoint(
    online_model,
    seed,
    episodes,
    frame_count,
    average_reward,
    evaluation_mean_reward=0,
    training_duration_seconds=0,
    config=None,
):
    checkpoint = {
    "model_state_dict": online_model.state_dict(),
    "seed": seed,
    "episodes": episodes,
    "emulator_frames": frame_count,
    "best_interval_training_episode_average_reward": average_reward,
    "evaluation_mean_reward": evaluation_mean_reward,
    "training_duration_seconds": training_duration_seconds,
    "config": config,
}

    checkpoint_path = config["evaluate"]["checkpoint_path"]
    torch.save(checkpoint, checkpoint_path)
