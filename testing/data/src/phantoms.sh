#!/bin/bash

# Generate phantom 01
c3d phantom01_source.nii.gz -scale 60 -smooth 1vox -info -type uchar -o phantom01_fixed.nii.gz

# Create a mask
c3d phantom01_source.nii.gz -thresh 0.5 inf 1 0 -dilate 1 2x2x2 \
  -type uchar -o phantom01_mask.nii.gz

# Generate the rigid transform
c3d_affine_tool \
  -tran -64 -64 64 \
  -rot 15 1 2 3 \
  -tran 3 4 5 \
  -tran 64 64 -64 \
  -mult -mult -mult -info \
  -o phantom01_rigid.mat

# Generate the moving phantom image 01
c3d phantom01_fixed.nii.gz phantom01_fixed.nii.gz -reslice-matrix phantom01_rigid.mat \
  -type uchar -o phantom01_moving.nii.gz

# Generate the second phantom for mutual information
c3d seg.nii.gz -replace 4 2 3 1 1 4 2 3 -scale 60 -smooth 1vox -info \
  -type uchar -o phantom02_fixed.nii.gz

# Apply rotation to the second phantom
c3d phantom02_fixed.nii.gz phantom02_fixed.nii.gz -reslice-matrix phantom01_rigid.mat \
  -type uchar -o phantom02_moving.nii.gz

# Create a phantom with a smooth gradient to examine issues with NCC
c3d phantom01_source.nii.gz -as X -pad 64x64x64 64x64x64 0 -info \
  -dup -cmv -pop -popas Y -times -push Y -add -smooth 1mm -pim r -stretch 0% 100% 0 255 \
  -popas Q -push X -push Q -reslice-identity -o phantom03_fixed.nii.gz \
  -push X -push Q -reslice-matrix phantom01_rigid.mat -o phantom03_moving.nii.gz
