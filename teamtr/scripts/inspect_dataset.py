#!/usr/bin/env python3
"""
Inspect HF dataset splits and columns.
"""

from __future__ import annotations

import argparse

import datasets


def main():
    parser = argparse.ArgumentParser(description="Inspect HF dataset schema.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config-name", default=None)
    args = parser.parse_args()

    ds = datasets.load_dataset(args.dataset_name, args.config_name) if args.config_name else datasets.load_dataset(
        args.dataset_name
    )
    print("Splits:", list(ds.keys()))
    for split in ds.keys():
        print(f"{split} columns:", ds[split].column_names)


if __name__ == "__main__":
    main()
