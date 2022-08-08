#!/bin/bash
set -e

# Inputs to the script
MANIFEST=
JSON=
TEMPLATE_DIR=

# Script name and dir
SCRIPT_DIRECTORY=$(dirname "$0")
SCRIPT_BASE_NAME=$(basename "$0")
SCRIPT="$SCRIPT_DIRECTORY/$SCRIPT_BASE_NAME"

# Starting iteration
START_ITER=0

# Initial template (subject or image)
INIT_TEMPLATE_IMAGE=
INIT_TEMPLATE_MASK=
INIT_TEMPLATE_ID=
INIT_TRANSFORM_MANIFEST=

# Temp directory
if [[ ! $TMPDIR ]]; then
  TMPDIR=/tmp
fi

function usage()
{
  echo "greedy_build_template : template building script"
  echo "usage:"
  echo "  greedy_build_template [options]"
  echo "required options:"
  echo "  -p <file.json>        : Parameter file, JSON"
  echo "  -i <manifest.csv>     : Manifest file with ids,images,masks"
  echo "  -o <dir>              : Output directory"
  echo "additional options:"
  echo "  -s <n>                : Start at iteration n (default: 0)"
  echo "  -n <n>                : Terminate after n iterations"
  echo "  -t <file.nii.gz>      : Specify initial template image"
  echo "  -m <file.nii.gz>      : Specify initial template mask"
  echo "  -T <id>               : Specify subject to use as initial"
  echo "  -R <manifest.csv>     : File with id,matfile specifying initial transforms"
  echo "  -d                    : Print extra debugging information"
  echo "  -P                    : Print sample parameter file"
  echo "  -S                    : Print registration schedule"
#  echo "  -W                    : Write intermediate registration outputs to disk"
}

function json_sample()
{
  cat <<'SAMPLE'
{
  "dimensions": 3,
  "masks": {
    "fixed": true,
    "moving": false
  },
  "options": {
    "rigid": "-search 1000 20 5 -n 100x40x0 -m WNCC 2x2x2",
    "affine": "-n 100x40x0 -m WNCC 2x2x2",
    "deformable": "-n 100x40x10 -m WNCC 2x2x2",
    "averaging": ""
  },
  "iterations": {
    "total": 6,
    "rigid": [0, -1],
    "affine": [0, -1],
    "deformable": [2, -1],
    "averaging": [0, -1],
    "shape_averaging": [0, -1]
  },
  "scheduler": {
    "enabled": true,
    "options": {
      "averaging" : "-n 4 -m 16G",
      "registration" : "-n 1 -m 8G"
    }
  }
}
SAMPLE
}

# No agruments? Print usage
if [[ $# -eq 0 ]]; then usage; exit 255; fi

# Read the command-line options
while getopts "i:o:p:E:s:n:t:m:T:R:dhPSW" opt; do
  case $opt in
    i) MANIFEST=$OPTARG;;
    o) TEMPLATE_DIR=$OPTARG;;
    p) JSON=$OPTARG;;
    s) START_ITER=$OPTARG;;
    n) RUN_ITER=$OPTARG;;
    t) INIT_TEMPLATE_IMAGE=$OPTARG;;
    m) INIT_TEMPLATE_MASK=$OPTARG;;
    T) INIT_TEMPLATE_ID=$OPTARG;;
    R) INIT_TRANSFORM_MANIFEST=$OPTARG;;
    W) WRITE_INTERMEDIATE=1;;
    d) set -x;;
    P) json_sample; exit 0;;
    S) CMD_MAIN=print_schedule;;
    h) usage; exit 0;;
    \?) echo "Unknown option $OPTARG"; exit 2;;
    :) echo "Option $OPTARG requires an argument"; exit 2;;
  esac
done

# Shift the arguments
shift $((OPTIND -1))

# Check required parameters
if [[ ! $MANIFEST || ! -f "$MANIFEST" ]]; then
  echo "Error: Missing input manifest file (-i)"
  exit 255
fi

if [[ ! $JSON || ! -f "$JSON" ]]; then
  echo "Error: Missing json parameter file (-p)"
  exit 255
fi

if [[ ! $TEMPLATE_DIR ]]; then
  echo "Error: Missing output directory (-o)"
  exit 255
fi

# Check the required commands
for tool in greedy greedy_template_average jq; do
  TOOLPATH=$(command -v $tool || echo "")
  if [[ $TOOLPATH ]]; then
    echo "Using $TOOLPATH"
  else
    echo "Error: $tool not found in PATH"
    exit 255
  fi
done
# Create the template directory
mkdir -p "$TEMPLATE_DIR"

# Get the list of ids
ALL_IDS=($(awk -F, '{print $1}' $MANIFEST))

# Initial transforms cannot be used without an initial template
if [[ $INIT_TRANSFORM_MANIFEST ]]; then
  if [[ ! $INIT_TEMPLATE_IMAGE && ! $INIT_TEMPLATE_ID ]]; then
    echo "Error: -R option cannot be used without -t or -T options"
    exit 255
  fi
fi

# Read the essential parameters from the .json file
GP_DIM=$(jq -r '.dimensions // 3' "$JSON")

# Fixed and moving masks
GP_MASK_FIX=$(jq -r 'if .masks.fixed then 1 else 0 end' "$JSON")
GP_MASK_MOV=$(jq -r 'if .masks.moving then 1 else 0 end' "$JSON")
if [[ $GP_MASK_FIX -eq 1 || $GP_MASK_MOV -eq 1 ]]; then
  GP_USE_MASKS=1
fi

# Affine and deformable options
GP_OPT_RIGID=$(jq -r '.options.rigid // "-n 100x100 -m WNCC 2"' "$JSON")
GP_OPT_AFFINE=$(jq -r '.options.affine // "-n 100x100 -m WNCC 2"' "$JSON")
GP_OPT_DEFORM=$(jq -r '.options.deformable // "-n 100x100 -m WNCC 2"' "$JSON")
GP_OPT_AVG=$(jq -r '.options.averaging // "-U 1 -N 0 .99 0 255"' "$JSON")
GP_OPT_RESLICE_IMAGE=$(jq -r '.options.reslice.image // ""' "$JSON")
GP_OPT_RESLICE_MASK=$(jq -r '.options.reslice.mask // "-ri label 0.2vox"' "$JSON")

# Iterations
GP_ITER=$(jq -r '.iterations.total // 5' "$JSON")
GP_ITER_RIG=($(jq -r "(.iterations.rigid // [0,0])[]" "$JSON"))
GP_ITER_AFF=($(jq -r "(.iterations.affine // [0,-1])[]" "$JSON"))
GP_ITER_DEF=($(jq -r "(.iterations.deformable // [1,-1])[]" "$JSON"))
GP_ITER_AVG=($(jq -r "(.iterations.averaging // [1,-1])[]" "$JSON"))
GP_ITER_SHAPE_AVG=($(jq -r "(.iterations.averaging // $GP_ITER_AVG)[]" "$JSON"))

# Scheduler
GP_USE_PYBATCH=$(jq -r '.scheduler.enabled // false' "$JSON")
GP_PYBATCH_OPT_AVG=$(jq -r '.scheduler.options.averaging' "$JSON")
GP_PYBATCH_OPT_REG=$(jq -r '.scheduler.options.registration' "$JSON")

# Create dump directory
if [[ $GP_USE_PYBATCH ]]; then
  DUMPDIR="$TEMPLATE_DIR/dump"
  mkdir -p "$DUMPDIR"
fi

# Create a log directory
LOGDIR="$TEMPLATE_DIR/logs"
mkdir -p "$LOGDIR"

# Run stuff in queue
function pybatch()
{
  bash "$SCRIPT_DIRECTORY/pybatch.sh" -o "$DUMPDIR" "$@"
}

# Set the variables for an individual subject
function set_subject_vars()
{
  local id iter iterstr

  # Specify the id
  id=${1?}
  iter=${2?}

  # Read the input image
  SUBJ_IMAGE=$(awk -v id=$id -F, '$1 == id {print $2}' "$MANIFEST")

  # Read the mask
  SUBJ_MASK=
  if [[ $GP_USE_MASKS ]]; then
    SUBJ_MASK=$(awk -v id=$id -F, '$1 == id {print $3}' "$MANIFEST")
  fi

  # When using the -W flag, the intermediate results are not overwritten
  # and we include the iteration number in the filename
  if [[ $WRITE_INTERMEDIATE ]]; then
    iterstr="${id}_iter${iter}"
  else
    iterstr="${id}"
  fi

  # Affine transformation
  SUBJ_RIGID_MATRIX="$TEMPLATE_DIR/rigid_${iterstr}_to_template.mat"
  SUBJ_AFFINE_MATRIX="$TEMPLATE_DIR/affine_${iterstr}_to_template.mat"

  # Initial transform
  SUBJ_INIT_TRANSFORM=
  if [[ $INIT_TRANSFORM_MANIFEST ]]; then
    SUBJ_INIT_TRANSFORM=$(awk -v id=$id -F, '$1 == id {print $2}' "$INIT_TRANSFORM_MANIFEST")
  fi

  # Deformable transformation
  SUBJ_ROOT_WARP="$TEMPLATE_DIR/warp_root_${iterstr}_to_template.nii.gz"
  SUBJ_WARP="$TEMPLATE_DIR/warp_${iterstr}_to_template.nii.gz"

  # Resliced image
  SUBJ_RESLICE_IMAGE="$TEMPLATE_DIR/reslice_${iterstr}_image_to_template.nii.gz"

  # Resliced mask
  SUBJ_RESLICE_MASK="$TEMPLATE_DIR/reslice_${iterstr}_mask_to_template.nii.gz"

  # Registration dump file
  SUBJ_RIGID_OUTPUT="$LOGDIR/log_rigid_${id}_iter${iter}.txt"
  SUBJ_AFFINE_OUTPUT="$LOGDIR/log_affine_${id}_iter${iter}.txt"
  SUBJ_DEFORM_OUTPUT="$LOGDIR/log_deformable_${id}_iter${iter}.txt"
  SUBJ_RESLICE_OUTPUT="$LOGDIR/log_reslice_${id}_iter${iter}.txt"
}

# Set template variables for given iteration
function set_template_vars()
{
  local iter

  # Read the iteration
  iter=${1?}

  # Iteration ID
  iter_id=$(printf "iter_%02d" $iter)

  # Set the template image
  TEMPLATE_IMAGE="$TEMPLATE_DIR/greedy_template_${iter_id}_image.nii.gz"

  if [[ $GP_USE_MASKS ]]; then
    TEMPLATE_MASK="$TEMPLATE_DIR/greedy_template_${iter_id}_mask.nii.gz"
  fi
}

# Check if an iteration is within range
function check_iter()
{
  local iter r0 r1
  iter=${1?}
  r0=$2
  r1=$3

  if [[ $r0 && $r1 ]]; then
    # Take care of negative range values
    if [[ $r0 -lt 0 ]]; then r0=$((GP_ITER+r0)); fi
    if [[ $r1 -lt 0 ]]; then r1=$((GP_ITER+r1)); fi

    # Check if in range
    if [[ $iter -ge $r0 && $iter -le $r1 ]]; then echo "1"; else echo ""; fi
  else
    echo ""
  fi
}



# Set iteration specific variables
function set_iter_vars()
{
  local iter
  iter=${1?}

  DO_RIGID=$(check_iter $iter ${GP_ITER_RIG[*]})
  DO_AFFINE=$(check_iter $iter ${GP_ITER_AFF[*]})
  DO_DEFORM=$(check_iter $iter ${GP_ITER_DEF[*]})
  DO_AVERAGE=$(check_iter $iter ${GP_ITER_AVG[*]})
  DO_SHAPE_AVG=$(check_iter $iter ${GP_ITER_SHAPE_AVG[*]})
}

# Run command echoing the command to a log file
function runlog()
{
  local logfile CMD

  logfile=${1?}
  shift

  echo "$@" > "$logfile"
  CMD=${1?}
  shift

  $CMD "$@" | tee -a "$logfile"
}

# Register subject to template
function template_register_and_reslice()
{
  read -r id iter <<< "$@"

  # Set the variables
  set_iter_vars $iter
  set_template_vars $iter
  set_subject_vars $id $iter

  # On the first iteration, if manual init rigid is provided, put that in place
  if [[ $iter -eq 0 ]]; then
    if [[ $SUBJ_INIT_TRANSFORM ]]; then
      cp "$SUBJ_INIT_TRANSFORM" "$SUBJ_RIGID_MATRIX"
    fi
  fi

  # Build the mask command
  if [[ $GP_USE_MASKS ]]; then

    if [[ $GP_MASK_FIX -gt 0 ]]; then
      MASK_CMD="-gm $TEMPLATE_MASK "
    fi

    if [[ $GP_MASK_MOV -gt 0 ]]; then
      MASK_CMD="${MASK_CMD} -mm ${SUBJ_MASK}"
    fi

    # Create a reslicing command
    MASK_RESLICE_CMD="$GP_OPT_RESLICE_MASK -rm $SUBJ_MASK $SUBJ_RESLICE_MASK"
  else
    MASK_CMD=
    MASK_RESLICE_CMD=
  fi

  # Do the rigid registration
  if [[ $DO_RIGID ]]; then

    # Perform registration
    runlog "$SUBJ_RIGID_OUTPUT" greedy -d $GP_DIM -a -dof 6 \
      -i $TEMPLATE_IMAGE $SUBJ_IMAGE \
      $MASK_CMD $GP_OPT_RIGID \
      $([ -f $SUBJ_RIGID_MATRIX ] && echo "-ia $SUBJ_RIGID_MATRIX") \
      -o $SUBJ_RIGID_MATRIX

  fi

  # Here the initialization should be either the rigid or affine
  # matrix, whichever is newer
  # TODO: this does not work with the -W option, need special logic
  # to determine the last iteration rigid or affine
  LATEST_AFFINE=
  if [[ $SUBJ_RIGID_MATRIX -nt $SUBJ_AFFINE_MATRIX ]]; then
    LATEST_AFFINE=$SUBJ_RIGID_MATRIX
  elif [[ -f $SUBJ_AFFINE_MATRIX ]]; then
    LATEST_AFFINE=$SUBJ_AFFINE_MATRIX
  fi

  # Do the affine registration, start with rigid if available
  if [[ $DO_AFFINE ]]; then
    runlog "$SUBJ_AFFINE_OUTPUT" greedy -d $GP_DIM -a -dof 12 \
      -i $TEMPLATE_IMAGE $SUBJ_IMAGE \
      $MASK_CMD $GP_OPT_AFFINE \
      $([ $LATEST_AFFINE ] && echo "-ia $LATEST_AFFINE") \
      -o $SUBJ_AFFINE_MATRIX

    LATEST_AFFINE=$SUBJ_AFFINE_MATRIX
  fi

  # Do registration
  if [[ $DO_DEFORM ]]; then

    # Perform registration
    runlog "$SUBJ_DEFORM_OUTPUT" greedy -d $GP_DIM -sv \
      -i $TEMPLATE_IMAGE $SUBJ_IMAGE \
      $MASK_CMD $GP_OPT_DEFORM \
      $([ $LATEST_AFFINE ] && echo "-it $LATEST_AFFINE") \
      -oroot $SUBJ_ROOT_WARP -o $SUBJ_WARP

  fi

  # There has to be at least one transform so no if statements here
  runlog "$SUBJ_RESLICE_OUTPUT" greedy -d $GP_DIM -rf $TEMPLATE_IMAGE \
    $GP_OPT_RESLICE_IMAGE -rm $SUBJ_IMAGE $SUBJ_RESLICE_IMAGE \
    $MASK_RESLICE_CMD \
    -r \
    $([ $DO_DEFORM ] && echo "$SUBJ_WARP") \
    $([ $LATEST_AFFINE ] && echo "$LATEST_AFFINE")
}


function template_make_average()
{
  local iter args
  read -r iter args <<< "$@"

  echo "Make average iter $iter"

  # Set the template variables
  set_template_vars $iter

  # Set the iteration variables, but for a previous iteration, since at this
  # stage we are collecting what was generated at the last iteration
  set_iter_vars $((iter-1))

  # Look up the initial template
  if [[ $INIT_TEMPLATE_ID ]]; then
    set_subject_vars $INIT_TEMPLATE_ID $iter
    INIT_TEMPLATE_IMAGE=$SUBJ_IMAGE
    INIT_TEMPLATE_MASK=$SUBJ_MASK
  fi

  # Create lists of all images, masks, etc
  ALL_IMAGES=()
  ALL_MASKS=()
  ALL_TFORMS=()
  for ((i=0;i<${#ALL_IDS[*]};i++)); do
    set_subject_vars ${ALL_IDS[i]} $iter
    if [[ iter -eq 0 ]]; then
      ALL_IMAGES[i]=$SUBJ_IMAGE
      ALL_MASKS[i]=$SUBJ_MASK
    else
      ALL_IMAGES[i]=$SUBJ_RESLICE_IMAGE
      ALL_MASKS[i]=$SUBJ_RESLICE_MASK

      # Transforms are only collected after deformable iterations
      [ $DO_DEFORM ] && ALL_TFORMS[i]=$SUBJ_ROOT_WARP
    fi
  done

  # Split depending on whether we are doing intensity averaging or
  # simply registering everything to the initial template
  set_iter_vars $iter
  if [[ $DO_AVERAGE ]]; then

    # Perform averaging
    greedy_template_average \
       -d $GP_DIM -i ${ALL_IMAGES[*]} $TEMPLATE_IMAGE \
       $([ $GP_USE_MASKS ] && echo "-m ${ALL_MASKS[*]} $TEMPLATE_MASK") \
       $([ $DO_SHAPE_AVG ] && [ $ALL_TFORMS ] && echo "-w ${ALL_TFORMS[*]}") \
       $GP_OPT_AVG

  else

    # Determine the previous template
    if [[ $iter -eq 0 ]]; then
      PREV_TEMPLATE_IMAGE=$INIT_TEMPLATE_IMAGE
      PREV_TEMPLATE_MASK=$INIT_TEMPLATE_MASK
    else
      set_template_vars $((iter-1))
      PREV_TEMPLATE_IMAGE=$TEMPLATE_IMAGE
      PREV_TEMPLATE_MASK=$TEMPLATE_MASK
      set_template_vars $iter
    fi

    # Propagate the previous template by copying or unwarping
    if [[ $ALL_TFORMS ]]; then

      # Perform unwarping on the template from the last iteration - this gets
      # kind of messy, and not sure why we are doing this
      greedy_template_average \
         -d $GP_DIM -i $PREV_TEMPLATE_IMAGE \
         $([ $GP_USE_MASKS ] && echo "-m $PREV_TEMPLATE_MASK $TEMPLATE_MASK") \
         -w ${ALL_TFORMS[*]} \
         $GP_OPT_AVG

    else

      cp $PREV_TEMPLATE_IMAGE $TEMPLATE_IMAGE
      [ $GP_USE_MASKS ] && cp $PREV_TEMPLATE_MASK $TEMPLATE_MASK || :

    fi
  fi
}

# Build the average and register everything to it
function build_basic_template()
{
  # User may limit the number of iterations
  NITER=$((GP_ITER))
  if [[ $RUN_ITER && $((RUN_ITER+START_ITER)) -lt $NITER ]]; then
    NITER=$((RUN_ITER+START_ITER))
  fi

  # Perform the iterations
  for ((iter=$START_ITER;iter<$NITER;iter++)); do

    if [[ $GP_USE_PYBATCH == "true" ]]; then

      # Compute the average
      pybatch -N "tempavg_${iter}" $GP_PYBATCH_OPT_AVG \
        $SCRIPT -d -i $MANIFEST -o $TEMPLATE_DIR -p $JSON \
        $([ $INIT_TEMPLATE_IMAGE ] && echo "-t $INIT_TEMPLATE_IMAGE") \
        $([ $INIT_TEMPLATE_MASK ] && echo "-m $INIT_TEMPLATE_MASK") \
        $([ $INIT_TEMPLATE_ID ] && echo "-T $INIT_TEMPLATE_ID") \
        $([ $INIT_TRANSFORM_MANIFEST ] && echo "-R $INIT_TRANSFORM_MANIFEST") \
        $([ $WRITE_INTERMEDIATE ] && echo "-W") \
        template_make_average ${iter}

      pybatch -w "tempavg_${iter}"

      # Perform registrations
      for((i=0;i<${#ALL_IDS[*]};i++)); do
        pybatch -N "tempreg_${iter}_${i}" $GP_PYBATCH_OPT_REG \
          $SCRIPT -d -i $MANIFEST -o $TEMPLATE_DIR -p $JSON \
          $([ $INIT_TEMPLATE_IMAGE ] && echo "-t $INIT_TEMPLATE_IMAGE") \
          $([ $INIT_TEMPLATE_MASK ] && echo "-m $INIT_TEMPLATE_MASK") \
          $([ $INIT_TEMPLATE_ID ] && echo "-T $INIT_TEMPLATE_ID") \
          $([ $INIT_TRANSFORM_MANIFEST ] && echo "-R $INIT_TRANSFORM_MANIFEST") \
          $([ $WRITE_INTERMEDIATE ] && echo "-W") \
          template_register_and_reslice ${ALL_IDS[i]} $iter
      done
      pybatch -w "tempreg_${iter}_*"

    else

      template_make_average $iter
      for((i=0;i<${#ALL_IDS[*]};i++)); do
        template_register_and_reslice ${ALL_IDS[i]} $iter
      done

    fi

  done
}

# Print schedule of registrations
function print_schedule()
{
  echo "Schedule:"
  NITER=$((GP_ITER))
  if [[ $RUN_ITER && $((RUN_ITER+START_ITER)) -lt $NITER ]]; then
    NITER=$((RUN_ITER+START_ITER))
  fi

  for ((iter=$START_ITER;iter<$NITER;iter++)); do
    set_iter_vars $iter
    echo "Iter $iter"
    printf "  Averaging:    %10s\n" "$([ $DO_AVERAGE ] && echo Yes || echo No)"
    printf "  Registration: %10s  %10s  %10s\n" \
      $([ $DO_RIGID ] && echo Rigid || echo " --- ") \
      $([ $DO_AFFINE ] && echo Affine || echo " --- ") \
      $([ $DO_DEFORM ] && echo Deformable || echo " --- ")
  done
}


# Check the initial template specification
set_iter_vars 0
if [[ ! $DO_AVERAGE ]]; then
  if [[ ! $INIT_TEMPLATE_IMAGE && ! $INIT_TEMPLATE_ID ]]; then
    echo "Initial template image must be specified (-t or -T)"
    exit 255
  fi
  if [[ $GP_USE_MASKS && ! $INIT_TEMPLATE_MASK && ! $INIT_TEMPLATE_ID ]]; then
    echo "Initial template mask must be specified (-m or -T)"
    exit 255
  fi
fi


if [[ $CMD_MAIN ]]; then
  if [[ $# -ge 1 ]]; then shift; fi
  $CMD_MAIN $@
elif [[ $# -ge 1 ]]; then
  CMD=${1}
  shift
  $CMD "$@"
else
  build_basic_template
fi
