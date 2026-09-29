from pathlib import Path
import pandas as pd
from PIL import Image
from vehicle_fingerprint.data.prepare import prepare_hackathon_dataset


def test_prepare(tmp_path: Path):
    imgs=tmp_path/'images'; imgs.mkdir()
    Image.new('RGB',(100,80),'gray').save(imgs/'abc.jpg')
    csv=tmp_path/'train.csv'
    pd.DataFrame([{'image_id':'abc','x':10,'y':10,'w':50,'h':40,'vehicle_id':1,'camera_id':2}]).to_csv(csv,index=False)
    out=prepare_hackathon_dataset(csv,imgs,tmp_path/'out',val_fraction=0,eval_fraction=0)
    assert len(out)==1
    assert Path(out.iloc[0].path).exists()
    assert out.iloc[0].split=='train'
