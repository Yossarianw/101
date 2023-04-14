import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from edt import edt
import imageio
from copy import copy
ps.visualization.set_mpl_style()
np.random.seed(5)
im = ps.generators.blobs(shape=[500, 500], porosity=0.6, blobiness=2)
bd = np.zeros_like(im, dtype=bool)
bd[:, 0] = 1
bd *= im
im = ps.filters.trim_disconnected_blobs(im=im, inlets=bd)
dt = edt(im)
out = ps.filters.ibip(im=im, inlets=bd, maxiter=15000)
inv_seq, inv_size = out.inv_sequence, out.inv_sizes
inv_satn = ps.filters.seq_to_satn(seq=inv_seq)
inv_seq_trapping = ps.filters.find_trapped_regions(seq=inv_seq, bins=None, return_mask=False)
inv_satn_trapping = ps.filters.seq_to_satn(seq=inv_seq_trapping)
sizes = np.arange(int(dt.max())+1, 0, -1)
mio = ps.filters.porosimetry(im=im, inlets=bd, sizes=sizes, mode='mio')
mio_seq = ps.filters.size_to_seq(mio)
mio_seq[im*(mio_seq == 0)] = -1  # Adjust to set uninvaded to -1
mio_satn = ps.filters.seq_to_satn(mio_seq)
mio_seq_trapping = ps.filters.find_trapped_regions(seq=mio_seq, bins=None,return_mask=False)
fig, ax = plt.subplots(1, 2, figsize=[8, 4])
ax[0].imshow(inv_satn/im, origin='lower', interpolation='none')
ax[1].imshow(mio_satn/im, origin='lower', interpolation='none');
inv_satn_t = np.around(inv_satn, decimals=4)
mio_satn_t = np.around(mio_satn, decimals=4)
satns = np.unique(mio_satn_t)[1:]
err = []
diff = np.zeros_like(im, dtype=float)
for s in satns:
    ip_mask = (inv_satn_t <= s) * (inv_satn_t > 0)
    mio_mask = (mio_satn_t <= s) * (mio_satn_t > 0)
    diff[(mio_mask == 1)*(ip_mask == 0)*(im == 1)] = 1
    diff[(mio_mask == 0)*(ip_mask == 1)*(im == 1)] = -1
    err.append((mio_mask != ip_mask).sum())
fig, ax = plt.subplots(1, 2, figsize=[8, 4])
ax[0].imshow(diff/im, origin='lower')
ax[1].plot(satns, err, 'o-');
d = ps.metrics.pc_curve_from_ibip(im=im, seq=inv_seq, sizes=inv_size, voxel_size=1e-5);
e = ps.metrics.pc_curve(im=im, sizes=mio, voxel_size=1e-5);
fig, ax = plt.subplots()
ax.plot(np.log10(d.pc), d.snwp, 'g-', linewidth=2, label='IBIP')
ax.step(np.array(np.log10(e.pc)), e.snwp, 'r--', where='post', markersize=20, linewidth=3, alpha=0.6, label='MIO')
plt.xlabel('log(Capillary Pressure [Pa])')
plt.ylabel('Non-wetting Phase Saturation')
plt.legend()
ax.xaxis.grid(True, which='both')
    