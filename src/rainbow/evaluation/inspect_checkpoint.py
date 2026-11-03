import argparse
import torch
from pathlib import Path


def print_checkpoint_infos(checkpoint):

    print("========== CHECKPOINT INFORMATION ==========")

    for key, value in checkpoint.items():
        if key == "model_state_dict" or key == "config":
            pass
        elif key =="training_duration_seconds" :
            print(f"{key} : {value//3600}h {(value% 3600) // 60}m {value%60}s")
        else:
            print(f"{key} : {value}")

    print("============================================")


def main():#To avoid trying parsing when importing the file in another module
    #------------------Parsing config------------------#
    parser = argparse.ArgumentParser(description="Inspect a Rainbow DQN checkpoint")
    parser.add_argument("checkpoint_path", help="Path to the checkpoint file")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    #--------------------------------------------------#

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint_name = Path(checkpoint_path).stem
    print(f"checkpoint name : {checkpoint_name}")
    print_checkpoint_infos(checkpoint)

if __name__=="__main__":
    main()





