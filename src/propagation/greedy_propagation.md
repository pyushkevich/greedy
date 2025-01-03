# Greedy Segmentation Propagation Tool

Greedy Segmentation Propagation Tool applys greedy registration to warp a 3D segmentation from a reference time point to target timepoints in a 4D image.

## Example Usage

```
greedy_propagation \
-i img4d.nii.gz \
-sr3 seg05.nii.gz \
-tpr 5 \
-tpt 1,2,3,4,6,7 \
-o /your/output/directory \
-debug /your/debug/output/directory \
-n 100x100 -m SSD -s 3mm 1.5mm -threads 10 -V 1
```

The above command warps the segmentation image `seg05.nii.gz` from reference time point `5` to time points `1,2,3,4,6,7` by running greedy registrations between time points with `img4d.nii.gz` as the reference 4d image.


## Propagation Parameters
### 4D Image Input `-i <image>`
Specifies the 4D image that is the base of the segmentation. Propagation algorithm will extract 3D time point images from the 4D image and use them as fix/moving images in the registration runs.

### 3D Reference Segmentation Input `-sr3 <image>`
Specifies the 3D segmentation image of the reference time point. This option will override all previously specified reference segmentation inputs by `-sps` or `sps-4d`. 

### 4D Reference Segmentation Input `-sr4 <image>`
Specify the 4D segmentation image containing the segmentation slice for the reference time point. Only the segmentation image from the reference time point will be used for the propagation run. This option will override all previously specified reference segmentation inputs by `-sr3` or `sr4`.

### Reference Time Point `-tpr <time point>`
Specifies the reference time point. Propagation will warp the segmentation from this time point to all the target time points.

### Target Time Point List `-tpt <time point list>`
Specifies the target time points in comma separated list. Propagation will warp the segmentation from the reference time point to all the target time points in this list.

### Output Directory `-o <output directory>`
Specifies the output directory for the propagation run. All the resliced segmentations and meshes will be written into this directory.

### Output Segmentation Filename Pattern (optional) `-sr-op <pattern>`
Specifies the output filename pattern for the resliced segmentations. The output filenames can be configured using a c-style pattern string, with a timepoint number embedded. For example: "Seg_%02d_resliced.nrrd" will generate "Seg_05_resliced.nrrd"

### Output Segmentation Mesh Filename Pattern (optional) `-sr-mop <pattern>`
Specifies the output filename pattern for the resliced segmentation meshes. The output filenames can be configured using a c-style pattern string, with a timepoint number embedded. For example: "SegMesh_%02d_resliced.vtk" will generate "SegMesh_05_resliced.vtk"

### Extra Meshes to Wrap (optional) `-emr <path to reference mesh> <output filename pattern>`
Specifies the external meshes to be warped together with the main segmentation mesh. Multiple -emr command can be provided to warp multiple meshes. For each mesh input file added, provide a corresponding output file pattern. See -sr-op for more info about filename patterns.

### Turn on Propagation Debugging Mode (optional) `-debug <debugging output directory>`
Turns on the debugging mode for propagation. An output directory is needed for storing generated intermediary files. Files dumped includes time point images, affine matrices, warp images, masks etc.

### Configure Propagation Verbosity (optional) `-verbose <0 | 1 | 2>`
Configures the verbosity of the propagation run. `0 - None; 1 - Default; 2 - Verbose`

## Accepted Greedy Parameters (optional)
### Metric `-m`
Specifies the metric to use for greedy registrations. Default is `SSD`.

### Multi-Resolution Schedule `-n`
Specifies the number of iterations at each resolution level for greedy registrations. Default is `100x100`

### Smoothing Kernels `-s`
Specifies the smoothing kernel for greedy registrations. Propagation default is `3mm 1.5mm`

### Affine DOF `-dof`
Specifies the type of affine registrations for the propagation run. Default is `12`

### Floating Point Precision `-float`
Turns on floating point precision for greedy registrations.

### Verbosity `-V <0 | 1 | 2>`
Configures the verbosity of the greedy registrations. Default is `1`

### Debugging Parameters `-dump-pyramid` and `-dump-metric`
Flags control debugging files dumping in greedy.