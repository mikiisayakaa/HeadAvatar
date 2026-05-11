import torch
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Light Check Script")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model data file')
    
    args = parser.parse_args()
    path = os.path.join(args.model_path, "final", "model_data.pt")
    model_data = torch.load(path)
    
    activation = torch.nn.Softplus()
    light = activation(model_data["envLight"])
    print(light)
    
    light_min = light.min().item()
    light_max = light.max().item()
    light_mean = light.mean().item()
    light_std = light.std().item()
    
    print(f"Light Min: {light_min}")
    print(f"Light Max: {light_max}")
    print(f"Light Mean: {light_mean}")
    print(f"Light Std Dev: {light_std}")