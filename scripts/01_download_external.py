#!/usr/bin/env python3
import argparse
from vehicle_fingerprint.external.registry import download_external
p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/external_datasets.yaml'); p.add_argument('--root', required=True); p.add_argument('names', nargs='*')
a=p.parse_args(); download_external(a.config, a.root, a.names or None)
