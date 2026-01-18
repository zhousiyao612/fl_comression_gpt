import argparse, yaml
from fl.utils import set_seed
from fl.federated import run_federated

def parse_config_overrides(args_list):
    """Parse config overrides from command line arguments like 'federated.rounds 50'"""
    overrides = {}
    i = 0
    while i < len(args_list):
        if '.' in args_list[i] and i + 1 < len(args_list):
            key_path = args_list[i]
            value = args_list[i + 1]
            
            # Try to convert to appropriate type
            try:
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
            except:
                pass  # Keep as string
            
            overrides[key_path] = value
            i += 2
        else:
            i += 1
    
    return overrides

def apply_config_overrides(cfg, overrides):
    """Apply config overrides to the config dictionary"""
    for key_path, value in overrides.items():
        keys = key_path.split('.')
        current = cfg
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value

def generate_filename_suffix(overrides):
    """Generate filename suffix from config overrides"""
    if not overrides:
        return ""
    
    suffix_parts = []
    for key_path, value in overrides.items():
        # Replace dots with underscores and add value
        key_clean = key_path.replace('.', '_')
        suffix_parts.append(f"{key_clean}_{value}")
    
    return "_" + "_".join(suffix_parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/example.yaml")
    args, unknown_args = parser.parse_known_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Parse config overrides from remaining arguments
    overrides = parse_config_overrides(unknown_args)
    
    # Apply overrides to config
    apply_config_overrides(cfg, overrides)
    
    # Generate filename suffix
    filename_suffix = generate_filename_suffix(overrides)

    set_seed(cfg.get("seed", 42))
    run_federated(cfg, args.cfg, filename_suffix)

if __name__ == "__main__":
    main()
