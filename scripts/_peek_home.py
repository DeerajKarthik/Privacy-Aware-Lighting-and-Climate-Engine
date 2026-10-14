import sys, zipfile
sys.path.insert(0,'src')
from palace.data.stream import aggregate
z=zipfile.ZipFile('data/raw/casas_home/new_labeled_data.zip')
for n in ['aruba.txt','cairo.txt','milan.txt','tulum1.txt','tulum2.txt']:
    with z.open(n) as f: d=aggregate(f)
    print(n,len(d),d.index.min(),d.index.max(),'bad',d.attrs['bad_lines'], d.motion_count.sum(), d.activity.nunique())
