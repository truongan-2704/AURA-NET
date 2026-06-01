"""
Enhanced ONNX export with optimization for AURA-Net
"""

import torch
import onnx
import onnxruntime as ort
from pathlib import Path
import numpy as np


def export_onnx(model, img_size=640, output_path='aura_net.onnx',
                opset_version=12, simplify=True, dynamic_batch=False,
                check=True, verbose=True):
    """
    Export AURA-Net to ONNX format with optimization

    Args:
        model: AURA-Net model
        img_size: Input image size
        output_path: Output ONNX file path
        opset_version: ONNX opset version
        simplify: Whether to simplify the model
        dynamic_batch: Whether to use dynamic batch size
        check: Whether to check the exported model
        verbose: Print verbose information

    Returns:
        output_path: Path to exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set model to eval mode
    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    if verbose:
        print(f'Exporting AURA-Net to ONNX...')
        print(f'  Input shape: {dummy_input.shape}')
        print(f'  Output path: {output_path}')
        print(f'  Opset version: {opset_version}')
        print(f'  Dynamic batch: {dynamic_batch}')

    # Dynamic axes
    if dynamic_batch:
        dynamic_axes = {
            'images': {0: 'batch'},
            'boxes': {0: 'batch'},
            'classes': {0: 'batch'},
            'objectness': {0: 'batch'},
            'evidence': {0: 'batch'},
            'uncertainty': {0: 'batch'}
        }
    else:
        dynamic_axes = None

    # Export
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['boxes', 'classes', 'objectness', 'evidence', 'uncertainty'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )

    if verbose:
        print(f'ONNX export successful: {output_path}')

    # Simplify
    if simplify:
        try:
            import onnxsim
            if verbose:
                print('Simplifying ONNX model...')

            model_onnx = onnx.load(str(output_path))
            model_onnx, check_ok = onnxsim.simplify(model_onnx)

            if check_ok:
                onnx.save(model_onnx, str(output_path))
                if verbose:
                    print('ONNX simplification successful')
            else:
                if verbose:
                    print('ONNX simplification failed, using original model')

        except ImportError:
            if verbose:
                print('onnx-simplifier not installed. Install with: pip install onnx-simplifier')
        except Exception as e:
            if verbose:
                print(f'ONNX simplification failed: {e}')

    # Check model
    if check:
        try:
            if verbose:
                print('Checking ONNX model...')

            model_onnx = onnx.load(str(output_path))
            onnx.checker.check_model(model_onnx)

            if verbose:
                print('ONNX model check passed')

            # Test inference
            if verbose:
                print('Testing ONNX inference...')

            ort_session = ort.InferenceSession(str(output_path))

            # Prepare input
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}

            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)

            if verbose:
                print('ONNX inference test passed')
                print(f'  Output shapes:')
                for i, output in enumerate(ort_outputs):
                    print(f'    Output {i}: {output.shape}')

        except Exception as e:
            if verbose:
                print(f'ONNX model check failed: {e}')

    # Print model info
    if verbose:
        model_onnx = onnx.load(str(output_path))
        print(f'\nONNX Model Info:')
        print(f'  IR version: {model_onnx.ir_version}')
        print(f'  Producer: {model_onnx.producer_name}')
        print(f'  Opset version: {model_onnx.opset_import[0].version}')
        print(f'  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB')

    return str(output_path)


def optimize_onnx(onnx_path, output_path=None, verbose=True):
    """
    Optimize ONNX model for inference

    Args:
        onnx_path: Path to ONNX model
        output_path: Output path (default: same as input with _optimized suffix)
        verbose: Print verbose information

    Returns:
        output_path: Path to optimized ONNX file
    """
    onnx_path = Path(onnx_path)

    if output_path is None:
        output_path = onnx_path.parent / f'{onnx_path.stem}_optimized.onnx'
    else:
        output_path = Path(output_path)

    if verbose:
        print(f'Optimizing ONNX model: {onnx_path}')

    # Load model
    model = onnx.load(str(onnx_path))

    # Optimize
    from onnxruntime.transformers import optimizer

    optimized_model = optimizer.optimize_model(
        str(onnx_path),
        model_type='bert',  # Generic optimization
        num_heads=0,
        hidden_size=0
    )

    # Save
    optimized_model.save_model_to_file(str(output_path))

    if verbose:
        print(f'Optimized ONNX model saved: {output_path}')
        print(f'  Original size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB')
        print(f'  Optimized size: {output_path.stat().st_size / 1024 / 1024:.2f} MB')

    return str(output_path)


def benchmark_onnx(onnx_path, img_size=640, num_runs=100, warmup=10, verbose=True):
    """
    Benchmark ONNX model inference speed

    Args:
        onnx_path: Path to ONNX model
        img_size: Input image size
        num_runs: Number of inference runs
        warmup: Number of warmup runs
        verbose: Print verbose information

    Returns:
        results: Dictionary with benchmark results
    """
    import time

    if verbose:
        print(f'Benchmarking ONNX model: {onnx_path}')

    # Create session
    ort_session = ort.InferenceSession(str(onnx_path))

    # Prepare input
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}

    # Warmup
    if verbose:
        print(f'Warming up ({warmup} runs)...')
    for _ in range(warmup):
        ort_session.run(None, ort_inputs)

    # Benchmark
    if verbose:
        print(f'Running benchmark ({num_runs} runs)...')

    times = []
    for _ in range(num_runs):
        start = time.time()
        ort_session.run(None, ort_inputs)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms

    # Calculate statistics
    times = np.array(times)
    results = {
        'mean': times.mean(),
        'std': times.std(),
        'min': times.min(),
        'max': times.max(),
        'median': np.median(times),
        'fps': 1000 / times.mean()
    }

    if verbose:
        print(f'\nBenchmark Results:')
        print(f'  Mean: {results["mean"]:.2f} ms')
        print(f'  Std: {results["std"]:.2f} ms')
        print(f'  Min: {results["min"]:.2f} ms')
        print(f'  Max: {results["max"]:.2f} ms')
        print(f'  Median: {results["median"]:.2f} ms')
        print(f'  FPS: {results["fps"]:.2f}')

    return results


class ONNXInference:
    """
    ONNX inference wrapper for AURA-Net
    """

    def __init__(self, onnx_path, conf_thres=0.25, iou_thres=0.45):
        """
        Args:
            onnx_path: Path to ONNX model
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
        """
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Create session
        self.session = ort.InferenceSession(str(onnx_path))

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        print(f'ONNX model loaded: {onnx_path}')
        print(f'  Input: {self.input_name}')
        print(f'  Outputs: {self.output_names}')

    def __call__(self, x):
        """
        Run inference

        Args:
            x: Input tensor or numpy array (B, 3, H, W)

        Returns:
            predictions: Dictionary of predictions
        """
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        # Run inference
        ort_inputs = {self.input_name: x}
        ort_outputs = self.session.run(self.output_names, ort_inputs)

        # Convert to dictionary
        predictions = {
            'boxes': torch.from_numpy(ort_outputs[0]),
            'classes': torch.from_numpy(ort_outputs[1]),
            'objectness': torch.from_numpy(ort_outputs[2]),
            'evidence': torch.from_numpy(ort_outputs[3]),
            'uncertainty': torch.from_numpy(ort_outputs[4])
        }

        return predictions
