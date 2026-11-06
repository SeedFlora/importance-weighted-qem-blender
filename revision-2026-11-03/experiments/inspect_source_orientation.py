from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
fig,axes=plt.subplots(3,3,figsize=(11,11))
for row,name in enumerate(['nefertiti','lucy','happy-buddha']):
    xyz=np.array([[float(s) for s in line.split()[1:4]] for line in (HERE/'external'/f'{name}.obj').read_text().splitlines() if line.startswith('v ')])
    # Native Z-up sources keep their native axes after cancelling OBJ conversion.
    if name=='happy-buddha': xyz=xyz[:,[0,2,1]] * [1,-1,1]
    xyz=xyz-(xyz.min(0)+xyz.max(0))/2
    xyz=2*xyz/np.ptp(xyz,axis=0).max()
    for col,(a,b) in enumerate([(0,2),(1,2),(0,1)]):
        ax=axes[row,col]; ax.scatter(xyz[::5,a],xyz[::5,b],s=.25,color='#39475b')
        ax.set_aspect('equal'); ax.set_title(f'{name}: '+['XZ','YZ','XY'][col]); ax.grid(alpha=.15)
fig.suptitle('Source geometry only, before comparative outcomes; proposed upright orientation')
fig.tight_layout()
fig.savefig(HERE/'source_orientation.png',dpi=140)
