"""
Export AURA-Net to ONNX format
"""

import sys
import yaml
import torch
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models.aura_net import build_aura_net
from utils.checkpoint import load_model_weights
from utils.onnx_export import export_onnx, optimize_onnx, benchmark_onnx


def parse_args():
    parser = argparse.ArgumentParser(description='Export AURA-Net to ONNX')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--config', type=str, default='configs/aura_net_s.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='aura_net.onnx',
                       help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--opset', type=int, default=12,
                       help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true',
                       help='Simplify ONNX model')
    parser.add_argument('--dynamic', action='store_true',
                       help='Dynamic batch size')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize ONNX model')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark ONNX model')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for export')
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device(args.device)
    print(f'Using device: {device}')

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Build model
    print('Building model...')
    model = build_aura_net(config)

    # Load weights
    print(f'Loading weights from {args.weights}')
    model = load_model_weights(model, args.weights, device)
    model = model.to(device)

    # Export to ONNX
    print('\nExporting to ONNX...')
    onnx_path = export_onnx(
        model,
        img_size=args.imgsz,
        output_path=args.output,
        opset_version=args.opset,
        simplify=args.simplify,
        dynamic_batch=args.dynamic,
        check=True,
        verbose=True
    )

    print(f'\nONNX export successful: {onnx_path}')

    # Optimize
    if args.optimize:
        print('\nOptimizing ONNX model...')
        optimized_path = optimize_onnx(onnx_path, verbose=True)
        print(f'Optimized ONNX saved: {optimized_path}')

    # Benchmark
    if args.benchmark:
        print('\nBenchmarking ONNX model...')
        results = benchmark_onnx(onnx_path, img_size=args.imgsz, verbose=True)

        if args.optimize:
            print('\nBenchmarking optimized ONNX model...')
            results_opt = benchmark_onnx(optimized_path, img_size=args.imgsz, verbose=True)

            print('\nComparison:')
            print(f'  Original: {results["mean"]:.2f} ms ({results["fps"]:.2f} FPS)')
            print(f'  Optimized: {results_opt["mean"]:.2f} ms ({results_opt["fps"]:.2f} FPS)')
            speedup = results["mean"] / results_opt["mean"]
            print(f'  Speedup: {speedup:.2f}x')

    print('\nDone!')


if __name__ == '__main__':
    main()
