# Federated Learning Framework

A modular PyTorch-based federated learning framework supporting:
- Model compression with error-feedback
- Non-IID data partitioning
- CNN / ViT / LLM model families
- Adjustable number of clients
- Communication bit accounting

## Supported Models

### Vision Transformers (ViT) and DeiT
The framework now supports various Vision Transformer models through the timm library:

- **DeiT-Small/16**: `deit_small_patch16_224` - Efficient for federated learning
- **DeiT-Base/16**: `deit_base_patch16_224` - Larger DeiT model
- **ViT-Base/16**: `vit_base_patch16_224` - Original Vision Transformer
- **ViT-Small/16**: `vit_small_patch16_224` - Smaller ViT variant
- **ViT-Tiny/16**: `vit_tiny_patch16_224` - Minimal ViT model

### CNN Models
- ResNet18
- MobileNet V3 Small

## Usage

Run with default CNN configuration:
```bash
python run.py --cfg configs/example.yaml
```

Run with DeiT-Small model:
```bash
python run.py --cfg configs/deit_example.yaml
```

Test model creation:
```bash
python test_deit_models.py
```

## Configuration

To use ViT/DeiT models, set the model family to "vit" and specify the model name:

```yaml
model:
  family: "vit"
  vit_name: "deit_small_patch16_224"  # or other supported models
```

## Requirements

- PyTorch
- torchvision  
- timm (for ViT/DeiT models)
- PyYAML

Install dependencies:
```bash
pip install torch torchvision timm pyyaml
```
