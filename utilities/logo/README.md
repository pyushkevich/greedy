# Greedy logo

`greedy.jpg` is the Greedy logo, and `greedy-card.jpg` is a smaller version without
captions for places where captions would be too small to read, such as website cards.
Both are made from a real greedy registration.

The moving image is a crop of a T1 MRI. The fixed image is made from it in four steps:
1. It is given a FLAIR-like contrast: dark CSF, grey matter brighter than white matter,
   bright caps around the ventricles, and a few white matter lesions.
2. It is warped by a known smooth deformation.
3. It is reduced to 4 mm axial slices.
4. Noise is added.

Greedy registers the two with NMI. The boxes in the middle show the moving image
resampled through the warp at several points of the optimization, with a grid
carried along by the warp.

To regenerate them, you need `greedy` and `c3d` on the path, Python 3 with `numpy`,
`scipy`, `nibabel` and `pillow`, the MRI
[`MRIcrop-orig.gipl.gz`](https://github.com/pyushkevich/itksnap/raw/master/Testing/TestData/MRIcrop-orig.gipl.gz)
from the ITK-SNAP test data, and the font
[Barlow SemiBold](https://github.com/google/fonts/raw/main/ofl/barlow/Barlow-SemiBold.ttf)
(SIL Open Font License). Then run:

```
./make_logo.sh MRIcrop-orig.gipl.gz work Barlow-SemiBold.ttf
```

This writes the intermediate images, warps and both logos to `work/`. It also prints
how far each warp is from the known deformation.

- `make_logo.sh` runs the steps in order.
- `make_warp.py` makes the synthetic deformation.
- `make_flair.py` gives the MRI its FLAIR-like contrast.
- `make_target.py` simulates the thick slices and noise.
- `warp_error.py` compares a warp with the known deformation.
- `compose_logo.py` draws the boxes.
