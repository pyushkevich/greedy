"""Write a smooth synthetic displacement field (components dx, dy, dz in mm) that
fades to zero at the edges of the volume, so the warped image has no empty borders."""
import sys
import numpy as np
import nibabel as nib
ref, outdir, amp = sys.argv[1], sys.argv[2], float(sys.argv[3])
im = nib.load(ref)
nx, ny, nz = im.shape[:3]
i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
taper = (np.sin(np.pi * i / (nx - 1)) * np.sin(np.pi * j / (ny - 1))) ** 0.25 * np.sin(np.pi * k / (nz - 1)) ** 0.8
dx = amp * np.sin(2 * np.pi * j / ny + 0.4) * np.cos(np.pi * k / nz)
dy = amp * np.sin(2 * np.pi * i / nx + 0.6) * np.cos(np.pi * k / nz)
dz = 0.7 * amp * np.sin(2 * np.pi * (i + j) / (nx + ny) + 0.3)
for name, d in (("dx", dx), ("dy", dy), ("dz", dz)):
    nib.save(nib.Nifti1Image((d * taper).astype(np.float32), im.affine), f"{outdir}/{name}.nii.gz")
