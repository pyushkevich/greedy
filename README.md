<p align="center">
  <img src="https://raw.githubusercontent.com/pyushkevich/greedy/master/utilities/logo/greedy.jpg" alt="Greedy" width="600">
</p>

# Greedy

Greedy is a tool for fast medical image registration, developed at the
[Penn Image Computing and Science Laboratory (PICSL)](https://picsl.upenn.edu) at the
University of Pennsylvania. Greedy was built to be a very fast CPU-based deformable
registration tool. It is meant for applications and pipelines where many images have to
be registered in parallel, such as multi-atlas segmentation.

Greedy shares many concepts and implementation strategies with Symmetric Normalization
(SyN) in [ANTs](https://github.com/ANTsX/ANTs). Unlike SyN, greedy is not symmetric,
which makes it faster; applications such as multi-atlas segmentation don't need
symmetry. Greedy also uses highly optimized code to compute image metrics.

Features:

- diffeomorphic deformable registration, plus affine and rigid registration, including
  initialization by matching moments of inertia;
- normalized cross-correlation, mutual information and sum of squared differences
  metrics;
- 2D and 3D images, including multiple and multi-component images;
- masks that restrict registration to a region of interest;
- multithreaded, SIMD-optimized code;
- a single executable for registration and for reslicing images and meshes;
- an API used by other software, including [ITK-SNAP](http://www.itksnap.org).

For example, this command registers two 3D images rigidly and saves the transform:

```
greedy -d 3 -a -dof 6 -m NMI -i fixed.nii.gz moving.nii.gz -o rigid.mat -n 100x50x10
```

## Getting Greedy

- **ITK-SNAP:** greedy comes with [ITK-SNAP](http://www.itksnap.org) 3.8 and later.
  Choose *Help → Install Command-Line Tools* from the ITK-SNAP menu.
- **Binaries:** nightly builds are on [SourceForge](https://sourceforge.net/projects/greedy-reg/files/).
- **Source:** see the [installation guide](https://greedy.readthedocs.io/en/latest/install.html).
- **Python:** `pip install picsl_greedy` installs the Python wrapper
  ([greedy_python](https://github.com/pyushkevich/greedy_python)). It runs greedy
  commands from Python and passes [SimpleITK](https://simpleitk.org) images to and from
  greedy.

## Documentation

See the [Greedy documentation](https://greedy.readthedocs.io/en/latest/), which has a
quick start guide and a full reference. Its source is in [docs](docs).

The logo is made from a real greedy registration; see
[utilities/logo](https://github.com/pyushkevich/greedy/tree/master/utilities/logo).

This work is funded by the NIH/NIBIB under grants R01 EB-017255 and R01 EB-014146.
