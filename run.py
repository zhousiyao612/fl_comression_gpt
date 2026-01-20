import argparse
import yaml
from fl.utils import set_seed
from fl.federated import run_federated

def set_nested_value(cfg, key_path, value):
    """Set a nested dictionary value using dot notation (e.g., 'noniid.alpha')"""
    keys = key_path.split('.')
    current = cfg
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value, converting to appropriate type
    final_key = keys[-1]
    try:
        # Try to convert to float first, then int, otherwise keep as string
        if '.' in value:
            current[final_key] = float(value)
        else:
            try:
                current[final_key] = int(value)
            except ValueError:
                current[final_key] = value
    except ValueError:
        current[final_key] = value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/example.yaml")
    
    # Parse known args to handle config overrides
    args, unknown = parser.parse_known_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Process config overrides from remaining arguments
    # Expected format: key.subkey value key2.subkey2 value2
    i = 0
    while i < len(unknown):
        if i + 1 < len(unknown):
            key_path = unknown[i]
            value = unknown[i + 1]
            set_nested_value(cfg, key_path, value)
            i += 2
        else:
            print(f"Warning: Ignoring unpaired argument: {unknown[i]}")
            i += 1

    set_seed(cfg.get("seed", 42))
    run_federated(cfg)

if __name__ == "__main__":
    main()