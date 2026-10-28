#!/bin/bash
# Make the Greedy logo (greedy.jpg) and a smaller version without captions
# (greedy-card.jpg) from a real greedy registration. See README.md.
set -e
if [[ $# -ne 3 ]]; then
  echo "usage: $0 MRIcrop-orig.gipl.gz work_dir Barlow-SemiBold.ttf"
  exit 1
fi
T=$1; G=$2; FONT=$3
D=$(cd $(dirname $0) && pwd)
mkdir -p $G

# moving image: the T1 MRI; also the signed distance to CSF, used to add FLAIR features
c3d $T -o $G/moving.nii.gz
c3d $T -smooth 1vox -otsu 3 -thresh 0 0 1 0 -sdt -o $G/sdt.nii.gz

# fixed image: a FLAIR-like version of the MRI, warped by a known smooth deformation,
# with 4 mm slices and noise
python3 $D/make_warp.py $G/moving.nii.gz $G 10
c3d $G/dx.nii.gz $G/dy.nii.gz $G/dz.nii.gz -omc $G/synth_warp.nii.gz
python3 $D/make_flair.py $G/moving.nii.gz $G/sdt.nii.gz $G/flair.nii.gz
greedy -d 3 -rf $G/moving.nii.gz -rm $G/flair.nii.gz $G/warped.nii.gz -r $G/synth_warp.nii.gz
python3 $D/make_target.py $G/warped.nii.gz $G/fixed.nii.gz

# register, stopping at several points of the optimization; NMI because the
# contrasts differ. Prints the error against the known deformation.
for n in 200x5x0 200x15x0 200x40x0 200x100x50; do
  greedy -d 3 -i $G/fixed.nii.gz $G/moving.nii.gz -m NMI -n $n -e 1.0 -s 3vox 1vox -o $G/warp_$n.nii.gz > $G/log_$n.txt
  python3 $D/warp_error.py $G/synth_warp.nii.gz $G/warp_$n.nii.gz $n
done

python3 $D/compose_logo.py $G $FONT $G/greedy.jpg
python3 $D/compose_logo.py $G $FONT $G/greedy-card.jpg thumb
