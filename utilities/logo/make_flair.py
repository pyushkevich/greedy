"""Give the T1 MRI a FLAIR-like contrast: dark CSF, grey matter brighter than white
matter, bright caps around the ventricles and a few white matter lesions."""
import sys
import numpy as np
import nibabel as nib
t1, sdt, out = sys.argv[1], sys.argv[2], sys.argv[3]
im = nib.load(t1)
v = im.get_fdata().squeeze()
d = nib.load(sdt).get_fdata().squeeze()    # signed distance to CSF (see make_logo.sh)
# T1 tissue intensities (Otsu classes): CSF ~19, grey ~43-56, white ~68, fat up to 269
g = np.interp(v, [0, 20, 40, 54, 68, 100, 269], [0, 12, 120, 175, 115, 105, 150])
g += 55 * np.exp(-np.clip(d, 0, None) / 1.5) * (d > 0) * (v > 40)  # periventricular rim
i, j, k = np.meshgrid(*[np.arange(n) for n in v.shape], indexing="ij")
for c, r in (((30, 40, 44), 2.5), ((47, 44, 46), 2.0), ((27, 34, 46), 2.2), ((52, 33, 30), 2.5)):
    blob = np.exp(-((i - c[0]) ** 2 + (j - c[1]) ** 2 + (k - c[2]) ** 2) / (2 * r ** 2))
    g += 110 * blob * (v > 60)
nib.save(nib.Nifti1Image(g.astype(np.float32), im.affine), out)
