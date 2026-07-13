import sys; sys.path.insert(0,'src')
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from datasets.mesh_loading import load_mesh
ROOT=Path('/home/datasets/customDatasets/cmr2/reconstructions')
laptops=sorted({d.name for d in ROOT.glob('laptop__*__v000')})[:16]
fig=plt.figure(figsize=(16,16)); rng=np.random.default_rng(0)
for i,name in enumerate(laptops):
    mp=ROOT/name/'mesh.glb'
    if not mp.is_file(): continue
    m=load_mesh(mp,process=False); v=np.asarray(m.vertices)
    sel=rng.choice(len(v),min(8000,len(v)),replace=False)
    ax=fig.add_subplot(4,4,i+1,projection='3d')
    ax.scatter(v[sel,0],v[sel,2],v[sel,1],s=1,c=v[sel,1],cmap='viridis')
    ax.set_title(name.replace('laptop__','').replace('__v000','')+f"\n{len(v)}v",fontsize=7); ax.set_axis_off()
    h=(v[sel].max(0)-v[sel].min(0)).max()/2+1e-6; mid=(v[sel].max(0)+v[sel].min(0))/2
    ax.set_xlim(mid[0]-h,mid[0]+h);ax.set_ylim(mid[2]-h,mid[2]+h);ax.set_zlim(mid[1]-h,mid[1]+h)
    ax.view_init(elev=20,azim=-60)
fig.savefig('outputs/renders/laptop_gallery.png',dpi=90,bbox_inches='tight')
print('saved outputs/renders/laptop_gallery.png')
for n in laptops: print(n)
