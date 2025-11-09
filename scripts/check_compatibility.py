"""
Version compatibility checker and fixer
"""
import sys
import subprocess
import pkg_resources
from packaging import version

def get_package_version(package_name):
    """Get installed version of a package"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def check_versions():
    """Check if installed versions are compatible"""

    print("=" * 60)
    print("VERSION COMPATIBILITY CHECK")
    print("=" * 60)

    issues = []
    warnings = []

    # Check critical packages
    checks = {
        'torch': {
            'min': '2.0.0',
            'max': '2.9.999',
            'recommended': '2.1.0',
            'reason': 'Quantization API compatibility'
        },
        'transformers': {
            'min': '4.30.0',
            'max': '4.60.0',
            'recommended': '4.35.0',
            'reason': 'Model loading and saving'
        },
        'numpy': {
            'min': '1.20.0',
            'max': '1.26.999',
            'recommended': '1.24.3',
            'reason': 'NumPy 2.x has breaking changes'
        }
    }

    for package, constraints in checks.items():
        installed = get_package_version(package)

        if installed is None:
            issues.append(f"❌ {package} is NOT INSTALLED")
            continue

        v = version.parse(installed)
        min_v = version.parse(constraints['min'])
        max_v = version.parse(constraints['max'])
        rec_v = version.parse(constraints['recommended'])

        if v < min_v:
            issues.append(
                f"❌ {package} {installed} is TOO OLD (minimum: {constraints['min']})\n"
                f"   Reason: {constraints['reason']}\n"
                f"   Fix: pip install {package}>={constraints['min']}"
            )
        elif v > max_v:
            warnings.append(
                f"⚠️  {package} {installed} is VERY NEW (may have breaking changes)\n"
                f"   Recommended: {constraints['recommended']}\n"
                f"   Reason: {constraints['reason']}\n"
                f"   Fix: pip install {package}=={constraints['recommended']}"
            )
        elif v != rec_v:
            warnings.append(
                f"ℹ️  {package} {installed} (recommended: {constraints['recommended']})\n"
                f"   Current version should work, but {constraints['recommended']} is tested"
            )
        else:
            print(f"✅ {package} {installed} (optimal)")

    print()

    if warnings:
        print("WARNINGS:")
        print("-" * 60)
        for w in warnings:
            print(w)
            print()

    if issues:
        print("CRITICAL ISSUES:")
        print("-" * 60)
        for i in issues:
            print(i)
            print()
        print("=" * 60)
        print("FIX: Install compatible versions")
        print("=" * 60)
        print("pip install -r requirements-stable.txt")
        print()
        return False

    if warnings:
        print("=" * 60)
        print("RECOMMENDED ACTION")
        print("=" * 60)
        print("Use tested stable versions:")
        print("pip install -r requirements-stable.txt")
        print()
    else:
        print("=" * 60)
        print("✅ ALL VERSIONS COMPATIBLE")
        print("=" * 60)
        print()

    return True

def auto_fix():
    """Automatically install stable versions"""
    print("Installing stable, tested versions...")
    print()

    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-r', 'requirements-stable.txt'
    ])

    print()
    print("✅ Stable versions installed!")
    print()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check version compatibility')
    parser.add_argument('--fix', action='store_true', help='Automatically install stable versions')
    args = parser.parse_args()

    if args.fix:
        auto_fix()
    else:
        compatible = check_versions()
        if not compatible:
            print("Run with --fix to automatically install compatible versions:")
            print("  python scripts/check_compatibility.py --fix")
            print()
            sys.exit(1)
