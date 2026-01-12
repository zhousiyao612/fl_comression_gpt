import argparse, yaml
from fl.utils import set_seed
from fl.federated import run_federated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/example.yaml")
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    run_federated(cfg)

if __name__ == "__main__":
    main()
