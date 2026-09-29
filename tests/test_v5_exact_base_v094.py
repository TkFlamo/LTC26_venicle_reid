from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from vehicle_fingerprint.baseline_v5_exact import exact_v5_base_argv


def _value(argv,key):
    idx=[i for i,x in enumerate(argv) if x==key][-1]
    return argv[idx+1]


def test_vendor_v5_source_is_byte_exact():
    expected={
        'vehicle_reid_v5.py':'1d8a2872f5f6e46cbabee3f67183ebd1f25437252335752033c7a6b21b83fc4c',
        'train_best_v5.py':'4a682f415962ef6c7483712e8b9f42463da577a03f1836154e6ba40a96f5e688',
    }
    root=ROOT/'third_party'/'vehicle_reid_v5_official'
    for name,h in expected.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h


def test_exact_base_recipe_comes_from_supplied_v5_base_profile():
    argv=exact_v5_base_argv(model_name='convnext_base.dinov3_lvd1689m',height=384,width=576,out='x',images_dir='raw/images',train_csv='train.csv',device='0')
    assert _value(argv,'--model')=='convnext_base.dinov3_lvd1689m'
    assert (_value(argv,'--height'),_value(argv,'--width'))==('384','576')
    assert (_value(argv,'--epochs'),_value(argv,'--warmup-epochs'),_value(argv,'--freeze-backbone-epochs'))==('48','2','1')
    assert _value(argv,'--patience')=='12'
    assert (_value(argv,'--p'),_value(argv,'--k'))==('16','4'
    )
    assert _value(argv,'--lr-backbone')=='3e-5'
    assert _value(argv,'--lr-head')=='3e-4'
    assert _value(argv,'--sampler')=='camera'
    assert _value(argv,'--lr-milestones')=='32,40'
    assert _value(argv,'--stage2-epoch')=='14'
    assert _value(argv,'--ce-weight')=='1.0'
    assert _value(argv,'--triplet-weight')=='1.0'
    assert _value(argv,'--circle-weight')=='0.5'
    assert _value(argv,'--stage2-ce-weight')=='0.25'
    assert _value(argv,'--stage2-triplet-weight')=='1.0'
    assert _value(argv,'--stage2-circle-weight')=='0.20'
    assert '--camera-aware-triplet' in argv and '--cross-camera-triplet' in argv and '--same-camera-hard-neg' in argv
    assert _value(argv,'--device')=='cuda:0'


def test_v094_pipeline_is_single_split_base_only_max_resolution():
    cfg=yaml.safe_load((ROOT/'configs'/'full_cv_pipeline.yaml').read_text())
    assert cfg['cv']['mode']=='single_fixed_development_split'
    assert cfg['cv']['inner_val_fraction']==0.20
    assert cfg['backbones']==['convnext_base','vit_base']
    assert cfg['resolutions']==[[384,576]]
    assert cfg['baseline']['profile']=='base'
    assert cfg['baseline']['p']==16 and cfg['baseline']['k']==4
    assert cfg['ensemble']['enabled'] is False
