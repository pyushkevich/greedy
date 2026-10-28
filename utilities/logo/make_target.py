"""Simulate the acquisition of the warped FLAIR: thick axial slices (4 mm by default)
and Rician noise."""
import sys
import numpy as np
import nibabel as nib
src, out = sys.argv[1], sys.argv[2]
thick_mm, sigma = (int(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (4, 6.0)
im = nib.load(src)
v = im.get_fdata().squeeze()
nx, ny, nz = v.shape
# average each slab of thick_mm slices, add noise to the acquired slices, and show
# each as a block of identical slices
n = nz // thick_mm
thick = v[:, :, :n * thick_mm].reshape(nx, ny, n, thick_mm).mean(axis=3)
rng = np.random.default_rng(1)
thick = np.hypot(thick + rng.normal(0, sigma, thick.shape), rng.normal(0, sigma, thick.shape))
v = np.repeat(thick, thick_mm, axis=2)
v = np.concatenate([v, np.repeat(v[:, :, -1:], nz - v.shape[2], axis=2)], axis=2)
nib.save(nib.Nifti1Image(v.astype(np.float32), im.affine), out)
