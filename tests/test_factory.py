from rainbow.factory import Factory
from config import config
import torch

# Create the environment and models
factory = Factory(config)

env = factory.create_environment()
online_model = factory.create_online_model()
target_model = factory.create_target_model()

# Initialize LazyLinear layers
dummy_tensor = torch.randn(1, 4, 84, 84)

online_model(dummy_tensor)
target_model(dummy_tensor)

# Copy online weights to target
target_model.load_state_dict(online_model.state_dict())

# Test parameter equality
all_equal = True

for (online_name, online_param), (target_name, target_param) in zip(
    online_model.named_parameters(),
    target_model.named_parameters()
):
    equal = torch.equal(online_param, target_param)

    print(f"{online_name}: {'PASS' if equal else 'FAIL'}")

    if not equal:
        all_equal = False

print("\nAll parameters identical:", all_equal)
