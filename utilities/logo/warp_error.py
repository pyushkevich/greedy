"""Print the mean displacement of a greedy warp and its mean error against the
synthetic warp, inside the box shown in the logo."""
import sys
import numpy as np
import nibabel as nib
c = (slice(15, 63), slice(31, 49), slice(2, 64))
s = nib.load(sys.argv[1]).get_fdata().squeeze()[c]
w = nib.load(sys.argv[2]).get_fdata().squeeze()[c]
print(sys.argv[3], "mean |u| %.2f of %.2f, error %.2f mm" % (
    np.linalg.norm(w, axis=-1).mean(), np.linalg.norm(s, axis=-1).mean(), np.linalg.norm(w - s, axis=-1).mean()))
