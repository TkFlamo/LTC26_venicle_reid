#!/usr/bin/env python3
import argparse
from vehicle_fingerprint.data.part_dataset import convert_carparts
p=argparse.ArgumentParser(description='Convert Carparts-Seg to the compact semantic ontology used by DINO part tokens')
p.add_argument('--carparts',required=True);p.add_argument('--out',required=True)
a=p.parse_args();print('carparts',convert_carparts(a.carparts,a.out))
