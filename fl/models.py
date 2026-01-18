import torch.nn as nn

def build_model(cfg: dict) -> nn.Module:
    fam = cfg["model"]["family"].lower()

    if fam == "cnn":
        return build_cnn(cfg)
    if fam == "vit":
        return build_vit(cfg)
    if fam == "llm":
        return build_llm_stub(cfg)  # 先给占位接口
    raise ValueError(f"Unknown model family: {fam}")

def build_model_for_client(cfg: dict) -> nn.Module:
    # client/server model must match
    return build_model(cfg)

def build_cnn(cfg):
    from torchvision import models
    name = cfg["model"]["name"].lower()
    num_classes = cfg["task"]["num_classes"]

    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        
        # Proper weight initialization
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
        
        m.apply(init_weights)
        return m
        
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        
        # Proper weight initialization
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
        
        m.apply(init_weights)
        return m
        
    raise ValueError(f"Unknown CNN model: {name}")

def build_vit(cfg):
    """
    Build Vision Transformer models using timm library.
    
    Supported models:
    - vit_tiny_patch16_224: ViT-Tiny/16
    - vit_small_patch16_224: ViT-Small/16  
    - vit_base_patch16_224: ViT-Base/16
    - deit_small_patch16_224: DeiT-Small/16
    - deit_base_patch16_224: DeiT-Base/16
    - And other timm ViT variants
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm library is required for ViT models. Install with: pip install timm")
    
    vit_name = cfg["model"]["vit_name"]
    num_classes = cfg["task"]["num_classes"]
    
    # Validate model name exists in timm
    available_models = timm.list_models()
    if vit_name not in available_models:
        # Try to find similar models for better error message
        similar_models = [m for m in available_models if any(keyword in m for keyword in ['vit', 'deit'])][:5]
        raise ValueError(f"Model '{vit_name}' not found in timm. Available ViT/DeiT models include: {similar_models}")
    
    # Create model without pretrained weights for federated learning
    m = timm.create_model(vit_name, pretrained=False, num_classes=num_classes)
    
    # Apply proper weight initialization for federated learning
    def init_vit_weights(module):
        """Initialize ViT weights following standard practices"""
        if isinstance(module, nn.Linear):
            # Xavier uniform for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            # For patch embedding
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    m.apply(init_vit_weights)
    return m

def build_llm_stub(cfg):
    """
    LLM联邦一般不会“全参FedAvg”，更常见是：
    - LoRA/Adapter/Prefix-tuning 的少量可训练参数联邦聚合
    - 或者只同步梯度/低秩更新
    这里先提供占位结构，确保框架能扩展。
    """
    vocab = 1000
    hidden = 256
    num_classes = cfg["task"].get("num_classes", 2)

    class TinyTextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, hidden)
            self.encoder = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Linear(hidden, num_classes)

        def forward(self, x):
            # x: [B, T]
            h = self.emb(x)
            out, _ = self.encoder(h)
            last = out[:, -1, :]
            return self.head(last)

    return TinyTextModel()
