"""
Model information utilities
"""

import torch
import torch.nn as nn


def count_parameters(model):
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def compute_flops(model, img_size=640, device='cpu'):
    """
    Compute FLOPs using thop library
    
    Args:
        model: PyTorch model
        img_size: Input image size
        device: Device to run on
    
    Returns:
        flops: Number of FLOPs
        params: Number of parameters
    """
    try:
        from thop import profile, clever_format
        
        model = model.to(device)
        input_tensor = torch.randn(1, 3, img_size, img_size).to(device)
        
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        
        return flops, params
    
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        total_params, _ = count_parameters(model)
        return "N/A", f"{total_params / 1e6:.2f}M"


def print_model_summary(model, img_size=640, device='cpu'):
    """
    Print detailed YOLO-style model summary with layer-by-layer breakdown
    
    Args:
        model: PyTorch model
        img_size: Input image size
        device: Device to run on
    """
    print("\n" + "="*100)
    print("AURA-Net Model Summary".center(100))
    print("="*100)
    
    # Header
    print(f"\n{'from':<6}{'n':<6}{'params':<12}{'module':<45}{'arguments':<30}")
    
    layer_idx = 0
    total_params = 0
    total_layers = 0
    
    # Helper function to format module name
    def get_module_name(module):
        return module.__class__.__name__
    
    # Helper function to count parameters in a module
    def count_module_params(module):
        return sum(p.numel() for p in module.parameters())
    
    # Traverse model structure
    def print_layer(module, name, from_idx=-1, depth=0):
        nonlocal layer_idx, total_params, total_layers
        
        # Skip if no parameters and not a container
        module_params = count_module_params(module)
        
        # Get module type
        module_type = get_module_name(module)
        
        # Format arguments based on module type
        args = ""
        if isinstance(module, nn.Conv2d):
            args = f"[{module.in_channels}, {module.out_channels}, {module.kernel_size[0]}, {module.stride[0]}]"
        elif isinstance(module, nn.Linear):
            args = f"[{module.in_features}, {module.out_features}]"
        elif isinstance(module, nn.BatchNorm2d):
            args = f"[{module.num_features}]"
        elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            args = f"[{module.in_channels}, {module.out_channels}]"
        
        # Print layer info if it has parameters or is a key module
        if module_params > 0 or module_type in ['Sequential', 'ModuleList']:
            print(f"{layer_idx:<6}{from_idx:<6}{module_params:<12}{name:<45}{args:<30}")
            total_params += module_params
            total_layers += 1
            layer_idx += 1
    
    # Print main components
    if hasattr(model, 'stem'):
        print_layer(model.stem, 'models.stem.MultiResolutionVisualStem', -1)
    
    if hasattr(model, 'local_encoder'):
        print_layer(model.local_encoder, 'models.evidence_encoder.LocalTextureEncoder', -1)
    
    if hasattr(model, 'global_memory'):
        print_layer(model.global_memory, 'models.global_memory.GlobalSceneMemory', -1)
    
    if hasattr(model, 'frequency_detail'):
        print_layer(model.frequency_detail, 'models.frequency_detail.FrequencyDetailEncoder', -1)
    
    if hasattr(model, 'evidence_generator'):
        print_layer(model.evidence_generator, 'models.region_evidence.RegionEvidenceGenerator', -1)
    
    if hasattr(model, 'adaptive_router'):
        print_layer(model.adaptive_router, 'models.adaptive_router.AdaptiveRegionProcessor', -1)
    
    if hasattr(model, 'detail_inspector'):
        print_layer(model.detail_inspector, 'models.detail_inspector.DetailEvidenceInspector', -1)
    
    if hasattr(model, 'decoder'):
        print_layer(model.decoder, 'models.fusion_decoder.EvidenceFusionDecoder', -1)
    
    # Compute FLOPs
    flops_str = "N/A"
    try:
        from thop import profile, clever_format
        input_tensor = torch.randn(1, 3, img_size, img_size).to(device)
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        flops_str = f"{flops / 1e9:.1f}"
    except:
        pass
    
    # Summary
    print("="*100)
    print(f"AURA-Net summary: {total_layers} layers, {total_params:,} parameters, {total_params:,} gradients, {flops_str} GFLOPs")
    print("="*100 + "\n")


def print_model_info(model, img_size=640, device='cpu'):
    """
    Print model information (backward compatibility)
    
    Args:
        model: PyTorch model
        img_size: Input image size
        device: Device to run on
    """
    print_model_summary(model, img_size, device)


def get_model_size(model):
    """
    Get model size in MB
    
    Args:
        model: PyTorch model
    
    Returns:
        size_mb: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return size_mb
