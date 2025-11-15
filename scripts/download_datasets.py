#!/usr/bin/env python3
"""
Dataset Download and Preparation Script

Downloads and prepares benchmark datasets for model evaluation.
Organizes datasets by name with metadata and sample data.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from datetime import datetime
import sys


def load_datasets_config(config_path: Path) -> Dict[str, Any]:
    """Load datasets configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset_from_hf(ds_name: str, subset: Optional[str], cache_dir: Path):
    """Load dataset from HuggingFace Hub"""
    try:
        if subset:
            return load_dataset(ds_name, subset, cache_dir=str(cache_dir))
        else:
            return load_dataset(ds_name, cache_dir=str(cache_dir))
    except Exception as e:
        print(f"  ✗ Failed to download: {e}", file=sys.stderr)
        return None


def collect_dataset_metadata(dataset, dataset_name: str, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect metadata about the downloaded dataset"""
    metadata = {
        'dataset_name': dataset_name,
        'hf_name': dataset_config['name'],
        'subset': dataset_config.get('subset'),
        'downloaded': datetime.now().isoformat(),
        'splits': {},
        'metrics': dataset_config.get('metrics', []),
        'preprocessing': dataset_config.get('preprocessing', {}),
        'description': dataset_config.get('description', ''),
    }
    
    for split_name, split_data in dataset.items():
        metadata['splits'][split_name] = {
            'num_examples': len(split_data),
            'features': list(split_data.features.keys()),
        }
    
    return metadata


def save_dataset_metadata(metadata: Dict[str, Any], output_dir: Path) -> Path:
    """Save metadata to JSON file"""
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata_file


def save_dataset_samples(dataset, output_dir: Path) -> Path:
    """Save sample data to JSON file"""
    sample_file = output_dir / 'samples.json'
    samples = {}
    
    for split_name, split_data in dataset.items():
        num_samples = min(5, len(split_data))
        if num_samples > 0:
            samples[split_name] = [
                {k: str(v) for k, v in example.items()}
                for example in split_data.select(range(num_samples))
            ]
    
    with open(sample_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    return sample_file


def download_dataset(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    cache_dir: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Download and prepare a single dataset.
    
    Returns metadata about the downloaded dataset.
    """
    print(f"\n{'='*80}")
    print(f"Downloading: {dataset_name}")
    print(f"{'='*80}")

    ds_name = dataset_config['name']
    subset = dataset_config.get('subset')
    
    print(f"  HuggingFace name: {ds_name}")
    if subset:
        print(f"  Subset: {subset}")
    
    # Load dataset
    dataset = load_dataset_from_hf(ds_name, subset, cache_dir)
    if dataset is None:
        return None
    
    # Collect metadata
    metadata = collect_dataset_metadata(dataset, dataset_name, dataset_config)
    
    # Create output directory
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata_file = save_dataset_metadata(metadata, dataset_output_dir)
    print(f"  Metadata saved: {metadata_file}")
    
    # Save sample data
    sample_file = save_dataset_samples(dataset, dataset_output_dir)
    print(f"  Samples saved: {sample_file}")
    
    total_examples = sum(
        split_info['num_examples']
        for split_info in metadata['splits'].values()
    )
    
    print(f"  ✓ Downloaded successfully")
    print(f"  Splits: {list(metadata['splits'].keys())}")
    print(f"  Total examples: {total_examples:,}")
    
    return metadata


def download_all_datasets(datasets_config: Dict[str, Any], cache_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Download all datasets in the configuration"""
    print(f"\nDownloading {len(datasets_config)} datasets")
    print(f"Cache directory: {cache_dir}")
    print(f"Output directory: {output_dir}")

    results = []
    failed = []

    for i, (dataset_name, dataset_config) in enumerate(datasets_config.items(), 1):
        print(f"\n[{i}/{len(datasets_config)}] Processing {dataset_name}...")

        result = download_dataset(dataset_name, dataset_config, cache_dir, output_dir)

        if result:
            results.append(result)
        else:
            failed.append(dataset_name)

    # Save summary
    summary = {
        'total_datasets': len(datasets_config),
        'successful': len(results),
        'failed': len(failed),
        'failed_datasets': failed,
        'download_date': datetime.now().isoformat(),
        'cache_dir': str(cache_dir),
        'datasets': results
    }

    summary_file = output_dir / 'download_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets: {len(datasets_config)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed datasets:")
        for name in failed:
            print(f"  - {name}")

    print(f"\nSummary saved: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare benchmark datasets'
    )
    parser.add_argument(
        '--config',
        default='config/datasets.yaml',
        help='Path to datasets config file'
    )
    parser.add_argument(
        '--cache-dir',
        default='datasets/.cache',
        help='Directory for HuggingFace cache'
    )
    parser.add_argument(
        '--output-dir',
        default='datasets',
        help='Output directory for dataset metadata'
    )
    parser.add_argument(
        '--dataset',
        help='Specific dataset to download (downloads all if not specified)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets and exit'
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / args.config
    cache_dir = script_dir / args.cache_dir
    output_dir = script_dir / args.output_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    try:
        config = load_datasets_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    datasets_config = config.get('datasets', {})

    # List mode
    if args.list:
        print(f"\nAvailable datasets: {len(datasets_config)}\n")
        print(f"{'Dataset Name':<35} {'HF Name':<40} {'Metrics'}")
        print("-" * 100)

        for name, ds_config in sorted(datasets_config.items()):
            hf_name = ds_config['name']
            subset = ds_config.get('subset', '')
            if subset:
                hf_name += f" ({subset})"
            metrics = ', '.join(ds_config.get('metrics', []))
            print(f"{name:<35} {hf_name:<40} {metrics}")

        return 0

    # Download datasets
    if args.dataset:
        # Download single dataset
        if args.dataset not in datasets_config:
            print(f"Error: Dataset '{args.dataset}' not found in config", file=sys.stderr)
            return 1

        dataset_config = datasets_config[args.dataset]
        result = download_dataset(args.dataset, dataset_config, cache_dir, output_dir)

        if result is None:
            return 1

    else:
        # Download all datasets
        download_all_datasets(datasets_config, cache_dir, output_dir)

    print("\n✓ Dataset download complete!")
    return 0


if __name__ == "__main__":
    exit(main())
