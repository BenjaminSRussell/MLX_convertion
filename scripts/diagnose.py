"""
Diagnostic script to identify library and environment issues
"""
import sys
import subprocess

def check_python_version():
    print("=" * 60)
    print("PYTHON VERSION")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print()

def check_packages():
    print("=" * 60)
    print("PACKAGE VERSIONS")
    print("=" * 60)

    packages = [
        'torch',
        'transformers',
        'datasets',
        'numpy',
        'psutil',
        'sklearn',
        'yaml',
        'mlx'  # This should fail, which is expected
    ]

    for pkg in packages:
        try:
            if pkg == 'yaml':
                import yaml
                mod = yaml
            elif pkg == 'sklearn':
                import sklearn
                mod = sklearn
            elif pkg == 'mlx':
                try:
                    import mlx.core as mx
                    mod = mx
                except ImportError as e:
                    print(f"❌ {pkg:20s} NOT AVAILABLE (Expected - requires Apple Silicon)")
                    print(f"   Error: {str(e)}")
                    continue
            else:
                mod = __import__(pkg)

            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {pkg:20s} {version}")
        except ImportError as e:
            print(f"❌ {pkg:20s} NOT INSTALLED")
            print(f"   Error: {str(e)}")
    print()

def check_cuda():
    print("=" * 60)
    print("CUDA / GPU STATUS")
    print("=" * 60)
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
        else:
            print("Running on CPU (this is fine for quantization)")
    except Exception as e:
        print(f"Error checking CUDA: {e}")
    print()

def test_import_converter():
    print("=" * 60)
    print("CONVERTER IMPORT TEST")
    print("=" * 60)
    try:
        sys.path.insert(0, '/home/user/MLX_convertion')
        from scripts.convert_pytorch import PyTorchInt8Converter
        print("✅ Converter imports successfully")

        # Try to create converter instance
        converter = PyTorchInt8Converter(config_path='config/models.yaml')
        print(f"✅ Converter instance created")
        print(f"   Models configured: {len(converter.config['models'])}")
    except Exception as e:
        print(f"❌ Failed to import/create converter")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    print()

def test_simple_quantization():
    print("=" * 60)
    print("SIMPLE QUANTIZATION TEST")
    print("=" * 60)
    try:
        import torch
        import torch.nn as nn

        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

        print("✅ Created test model")

        # Try quantization
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )

        print("✅ Quantization works!")

        # Test inference
        test_input = torch.randn(1, 10)
        output = quantized(test_input)
        print(f"✅ Inference works! Output shape: {output.shape}")

    except Exception as e:
        print(f"❌ Quantization test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    print()

def main():
    print("\n" + "=" * 60)
    print("MLX CONVERSION PIPELINE - DIAGNOSTIC REPORT")
    print("=" * 60)
    print()

    check_python_version()
    check_packages()
    check_cuda()
    test_import_converter()
    test_simple_quantization()

    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print()
    print("If you see errors above, please share this output so I can help fix the issue.")
    print()

if __name__ == '__main__':
    main()
