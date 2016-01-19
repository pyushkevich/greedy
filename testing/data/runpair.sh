#!/bin/bash

# Parameters
# $1 - fixed phantom number
# $2 - moving phantom number
# $3 - metric (NCC in quotes)
# $4 - mask 1/0

rm -rf /tmp/test_affine.mat /tmp/src_reslice.nii.gz

if [[ ${4?} -eq 1 ]]; then
  MASK="-gm phantom01_mask.nii.gz"
fi

# Perform the registration
echo ../../../xc64rel/greedy -d 3 \
  -m $3 -a -i phantom${1}_fixed.nii.gz phantom${2}_moving.nii.gz \
  $MASK -o /tmp/test_affine.mat -n 40x40 -debug-deriv

../../../xc64rel/greedy -d 3 \
  -m $3 -a -i phantom${1}_fixed.nii.gz phantom${2}_moving.nii.gz \
  $MASK -o /tmp/test_affine.mat -n 40x40 -debug-deriv

# Apply to the phantom
../../../xc64rel/greedy -d 3 -ri LABEL 0.1vox \
    -rm phantom01_source.nii.gz /tmp/src_reslice.nii.gz \
    -rf phantom01_fixed.nii.gz -r /tmp/test_affine.mat phantom01_rigid.mat 

# Compute the overlap
c3d /tmp/src_reslice.nii.gz phantom01_source.nii.gz -label-overlap


