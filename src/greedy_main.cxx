/*=========================================================================

  Program:   ALFABIS fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  This program is part of ALFABIS: Adaptive Large-Scale Framework for
  Automatic Biomedical Image Segmentation.

  ALFABIS development is funded by the NIH grant R01 EB017255.

  ALFABIS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ALFABIS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ALFABIS.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================*/

#include "GreedyAPI.h"
#include "CommandLineHelper.h"

#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cerrno>

#include "lddmm_common.h"
#include "lddmm_data.h"

#include <itkImageFileReader.h>
#include <itkAffineTransform.h>
#include <itkTransformFactory.h>
#include <itkTimeProbe.h>

#include "MultiImageRegistrationHelper.h"
#include "FastWarpCompositeImageFilter.h"
#include <vnl/vnl_cost_function.h>
#include <vnl/vnl_random.h>
#include <vnl/algo/vnl_powell.h>
#include <vnl/algo/vnl_svd.h>
#include <vnl/vnl_trace.h>

extern const char *GreedyVersionInfo;

int usage()
{
  printf("greedy: Paul's greedy diffeomorphic registration implementation\n");
  printf("Usage: \n");
  printf("  greedy [options]\n");
  printf("Required options: \n");
  printf("  -d DIM                 : Number of image dimensions\n");
  printf("  -i fix.nii mov.nii     : Image pair (may be repeated)\n");
  printf("  -o output.nii          : Output file\n");
  printf("Mode specification: \n");
  printf("  -a                     : Perform affine registration and save to output (-o)\n");
  printf("  -brute radius          : Perform a brute force search around each voxel \n");
  printf("  -moments <1|2>         : Perform moments of inertia rigid alignment of given order.\n");
  printf("                             order 1 matches center of mass only\n");
  printf("                             order 2 matches second-order moments of inertia tensors\n");
  printf("  -r [tran_spec]         : Reslice images instead of doing registration \n");
  printf("                               tran_spec is a series of warps, affine matrices\n");
  printf("  -iw inwarp outwarp     : Invert previously computed warp\n");
  printf("  -root inwarp outwarp N : Convert 2^N-th root of a warp \n");
  printf("Options in deformable / affine mode: \n");
  printf("  -w weight              : weight of the next -i pair\n");
  printf("  -m metric              : metric for the entire registration\n");
  printf("                               SSD:          sum of square differences (default)\n");
  printf("                               MI:           mutual information\n");
  printf("                               NMI:          normalized mutual information\n");
  printf("                               NCC <radius>: normalized cross-correlation\n");
  printf("  -e epsilon             : step size (default = 1.0), \n");
  printf("                               may also be specified per level (e.g. 0.3x0.1)\n");
  printf("  -n NxNxN               : number of iterations per level of multi-res (100x100) \n");
  printf("  -threads N             : set the number of allowed concurrent threads\n");
  printf("  -gm mask.nii           : mask for gradient computation\n");
  printf("  -mm mask.nii           : mask for the moving image\n");
  printf("  -it filenames          : sequence of transforms to apply to the moving image first \n");
  printf("Specific to deformable mode: \n");
  printf("  -tscale MODE           : time step behavior mode: CONST, SCALE [def], SCALEDOWN\n");
  printf("  -s sigma1 sigma2       : smoothing for the greedy update step. Must specify units,\n");
  printf("                           either `vox` or `mm`. Default: 1.732vox, 0.7071vox\n");
  printf("  -oinv image.nii        : compute and write the inverse of the warp field into image.nii\n");
  printf("  -invexp VALUE          : how many times to take the square root of the forward\n");
  printf("                                transform when computing inverse (default=2)\n");
  printf("  -wp VALUE              : Saved warp precision (in voxels; def=0.1; 0 for no compression).\n");
  printf("  -noise VALUE           : Standard deviation of white noise added to moving/fixed images when \n");
  printf("                           using NCC metric. Relative to intensity range. Def=0.001\n");
  printf("Initial transform specification: \n");
  printf("  -ia filename           : initial affine matrix for optimization (not the same as -it) \n");
  printf("  -ia-identity           : initialize affine matrix based on NIFTI headers \n");
  printf("  -ia-image-centers      : initialize affine matrix based on matching image centers \n");
  printf("  -ia-image-side CODE    : initialize affine matrix based on matching center of one image side \n");
  printf("  -ia-moments <1|2>      : initialize affine matrix based on matching moments of inertia\n");
  printf("Specific to affine mode (-a):\n");
  printf("  -dof N                 : Degrees of freedom for affine reg. 6=rigid, 12=affine\n");
  printf("  -jitter sigma          : Jitter (in voxel units) applied to sample points (def: 0.5)\n");
  printf("  -search N s_ang s_xyz  : Random search over rigid transforms (N iter) before starting optimization\n");
  printf("                           s_ang, s_xyz: sigmas for rot-n angle (degrees) and offset between image centers\n");
  printf("Specific to moments of inertia mode (-moments 2): \n");
  printf("  -det <-1|1>            : Force the determinant of transform to be either 1 (no flip) or -1 (flip)\n");
  printf("  -cov-id                : Assume identity covariance (match centers and do flips only, no rotation)\n");
  printf("Specific to reslice mode (-r): \n");
  printf("  -rf fixed.nii          : fixed image for reslicing\n");
  printf("  -rm mov.nii out.nii    : moving/output image pair (may be repeated)\n");
  printf("  -rs mov.vtk out.vtk    : moving/output surface pair (vertices are warped from fixed space to moving)\n");
  printf("  -ri interp_mode        : interpolation for the next pair (NN, LINEAR*, LABEL sigma)\n");
  printf("  -rc outwarp            : write composed transforms to outwarp \n");
  printf("  -rj outjacobian        : write Jacobian determinant image to outjacobian \n");
  printf("For developers: \n");
  printf("  -debug-deriv           : enable periodic checks of derivatives (debug) \n");
  printf("  -debug-deriv-eps       : epsilon for derivative debugging \n");
  printf("  -debug-aff-obj         : plot affine objective in neighborhood of -ia matrix \n");
  printf("  -dump-moving           : dump moving image at each iter\n");
  printf("  -dump-freq N           : dump frequency\n");
  printf("  -powell                : use Powell's method instead of LGBFS\n");
  printf("  -float                 : use single precision floating point (off by default)\n");
  printf("  -version               : print version info\n");

  return -1;
}



template <unsigned int VDim, typename TReal>
class GreedyRunner
{
public:
  static int Run(GreedyParameters &param)
  {
    GreedyApproach<VDim, TReal> greedy;
    return greedy.Run(param);

  }
};





int main(int argc, char *argv[])
{
  GreedyParameters param;
  GreedyParameters::SetToDefaults(param);

  double current_weight = 1.0;

  // reslice mode parameters
  InterpSpec interp_current;

  if(argc < 2)
    return usage();

  try
  {
    CommandLineHelper cl(argc, argv);
    while(!cl.is_at_end())
      {
      // Read the next command
      std::string arg = cl.read_command();

      if(arg == "-d")
        {
        param.dim = cl.read_integer();
        }
      else if(arg == "-float")
        {
        param.flag_float_math = true;
        }
      else if(arg == "-n")
        {
        param.iter_per_level = cl.read_int_vector();
        }
      else if(arg == "-w")
        {
        current_weight = cl.read_double();
        }
      else if(arg == "-e")
        {
        param.epsilon_per_level = cl.read_double_vector();
        }
      else if(arg == "-m")
        {
        std::string metric_name = cl.read_string();
        if(metric_name == "NCC" || metric_name == "ncc")
          {
          param.metric = GreedyParameters::NCC;
          param.metric_radius = cl.read_int_vector();
          }
        else if(metric_name == "MI" || metric_name == "mi")
          {
          param.metric = GreedyParameters::MI;
          }
        else if(metric_name == "NMI" || metric_name == "nmi")
          {
          param.metric = GreedyParameters::NMI;
          }
        }
      else if(arg == "-tscale")
        {
        std::string mode = cl.read_string();
        if(mode == "SCALE" || mode == "scale")
          param.time_step_mode = GreedyParameters::SCALE;
        else if(mode == "SCALEDOWN" || mode == "scaledown")
          param.time_step_mode = GreedyParameters::SCALEDOWN;
        }
      else if(arg == "-noise")
        {
        param.ncc_noise_factor = cl.read_double();
        }
      else if(arg == "-s")
        {
        param.sigma_pre.sigma = cl.read_scalar_with_units(param.sigma_pre.physical_units);
        param.sigma_post.sigma = cl.read_scalar_with_units(param.sigma_post.physical_units);
        }
      else if(arg == "-i")
        {
        ImagePairSpec ip;
        ip.weight = current_weight;
        ip.fixed = cl.read_existing_filename();
        ip.moving = cl.read_existing_filename();
        param.inputs.push_back(ip);
        }
      else if(arg == "-ia")
        {
        param.affine_init_mode = RAS_FILENAME;
        param.affine_init_transform = cl.read_transform_spec();
        }
      else if(arg == "-ia-identity" || arg == "-iaid" || arg == "-ia-id")
        {
        param.affine_init_mode = RAS_IDENTITY;
        }
      else if(arg == "-ia-image-centers" || arg == "-iaic" || arg == "-ia-ic")
        {
        param.affine_init_mode = IMG_CENTERS;
        }
      else if(arg == "-dof")
        {
        int dof = cl.read_integer();
        if(dof == 6)
          param.affine_dof = GreedyParameters::DOF_RIGID;
        else if(dof == 12)
            param.affine_dof = GreedyParameters::DOF_AFFINE;
        else throw GreedyException("DOF parameter only accepts 6 and 12 as values");
        }
      else if(arg == "-jitter")
        {
        param.affine_jitter = cl.read_double();
        }
      else if(arg == "-search")
        {
        param.rigid_search.iterations = cl.read_integer();
        param.rigid_search.sigma_angle = cl.read_double();
        param.rigid_search.sigma_xyz = cl.read_double();
        }
      else if(arg == "-it")
        {
        int nFiles = cl.command_arg_count();
        for(int i = 0; i < nFiles; i++)
          param.moving_pre_transforms.push_back(cl.read_transform_spec());
        }
      else if(arg == "-gm")
        {
        param.gradient_mask = cl.read_existing_filename();
        }
      else if(arg == "-mm")
        {
        param.moving_mask = cl.read_existing_filename();
        }
      else if(arg == "-o")
        {
        param.output = cl.read_output_filename();
        }
      else if(arg == "-dump-moving")
        {
        param.flag_dump_moving = true;
        }
      else if(arg == "-powell")
        {
        param.flag_powell = true;
        }
      else if(arg == "-dump-frequency" || arg == "-dump-freq")
        {
        param.dump_frequency = cl.read_integer();
        }
      else if(arg == "-debug-deriv")
        {
        param.flag_debug_deriv = true;
        }
      else if(arg == "-debug-deriv-eps")
        {
        param.deriv_epsilon = cl.read_double();
        }
      else if(arg == "-debug-aff-obj")
        {
        param.flag_debug_aff_obj = true;
        }
      else if(arg == "-threads")
        {
        param.threads = cl.read_integer();
        }
      else if(arg == "-version")
        {
        std::cout << GreedyVersionInfo << std::endl;
        exit(0);
        }
      else if(arg == "-a")
        {
        param.mode = GreedyParameters::AFFINE;
        }
      else if(arg == "-moments")
        {
        param.mode = GreedyParameters::MOMENTS;

        // For backward compatibility allow no parameter, which defaults to order 1
        param.moments_order = cl.command_arg_count() > 0 ? cl.read_integer() : 2;
        if(param.moments_order != 1 && param.moments_order != 2)
          throw GreedyException("Parameter to -moments must be 1 or 2");
        }
      else if(arg == "-brute")
        {
        param.mode = GreedyParameters::BRUTE;
        param.brute_search_radius = cl.read_int_vector();
        }
      else if(arg == "-r")
        {
        param.mode = GreedyParameters::RESLICE;
        int nFiles = cl.command_arg_count();
        for(int i = 0; i < nFiles; i++)
          param.reslice_param.transforms.push_back(cl.read_transform_spec());
        }
      else if(arg == "-iw")
        {
        param.mode = GreedyParameters::INVERT_WARP;
        param.invwarp_param.in_warp = cl.read_existing_filename();
        param.invwarp_param.out_warp = cl.read_output_filename();
        }
      else if(arg == "-root")
        {
        param.mode = GreedyParameters::ROOT_WARP;
        param.warproot_param.in_warp = cl.read_existing_filename();
        param.warproot_param.out_warp = cl.read_output_filename();
        param.warproot_param.exponent = cl.read_integer();
        }

      else if(arg == "-rm")
        {
        ResliceSpec rp;
        rp.interp = interp_current;
        rp.moving = cl.read_existing_filename();
        rp.output = cl.read_output_filename();
        param.reslice_param.images.push_back(rp);
        }
      else if(arg == "-rs")
        {
        ResliceMeshSpec rp;
        rp.fixed = cl.read_existing_filename();
        rp.output = cl.read_output_filename();
        param.reslice_param.meshes.push_back(rp);
        }
      else if(arg == "-rf")
        {
        param.reslice_param.ref_image = cl.read_existing_filename();
        }
      else if(arg == "-rc")
        {
        param.reslice_param.out_composed_warp = cl.read_output_filename();
        }
      else if(arg == "-rj")
        {
        param.reslice_param.out_jacobian_image = cl.read_output_filename();
        }
      else if(arg == "-oinv")
        {
        param.inverse_warp = cl.read_output_filename();
        }
      else if(arg == "-invexp")
        {
        param.inverse_exponent = cl.read_integer();
        }
      else if(arg == "-ri")
        {
        std::string mode = cl.read_string();
        if(mode == "nn" || mode == "NN" || mode == "0")
          {
          interp_current.mode = InterpSpec::NEAREST;
          }
        else if(mode == "linear" || mode == "LINEAR" || mode == "1")
          {
          interp_current.mode = InterpSpec::LINEAR;
          }
        else if(mode == "label" || mode == "LABEL")
          {
          interp_current.mode = InterpSpec::LABELWISE;
          interp_current.sigma.sigma = cl.read_scalar_with_units(interp_current.sigma.physical_units);
          }
        else
          {
          std::cerr << "Unknown interpolation mode" << std::endl;
          }
        }
      else if(arg == "-wp")
        {
        param.warp_precision = cl.read_double();
        }
      else if(arg == "-det")
        {
        int det_value = cl.read_integer();
        if(det_value != -1 && det_value != 1)
          {
          std::cerr << "Incorrect -det parameter value " << det_value << std::endl;
          return -1;
          }
        param.moments_flip_determinant = det_value;
        }
      else if(arg == "-cov-id")
        {
        param.flag_moments_id_covariance = true;
        }
      else
        {
        std::cerr << "Unknown parameter " << arg << std::endl;
        return -1;
        }
      }

    // Use the threads parameter
    if(param.threads > 0)
      {
      std::cout << "Limiting the number of threads to " << param.threads << std::endl;
      itk::MultiThreader::SetGlobalMaximumNumberOfThreads(param.threads);
      }
    else
      {
      std::cout << "Executing with the default number of threads: " << itk::MultiThreader::GetGlobalDefaultNumberOfThreads() << std::endl;

      }

    // Some parameters may be specified as either vector or scalar, and need to be verified
    if(param.epsilon_per_level.size() != param.iter_per_level.size())
      {
      if(param.epsilon_per_level.size() == 1)
        {
        param.epsilon_per_level = 
          std::vector<double>(param.iter_per_level.size(), param.epsilon_per_level.back());
        }
      else
        {
        throw GreedyException("Mismatch in size of vectors supplied with -n and -e options");
        }
      }

    // Run the main code
    if(param.flag_float_math)
      {
      switch(param.dim)
        {
        case 2: return GreedyRunner<2, float>::Run(param); break;
        case 3: return GreedyRunner<3, float>::Run(param); break;
        case 4: return GreedyRunner<4, float>::Run(param); break;
        default: throw GreedyException("Wrong number of dimensions requested: %d", param.dim);
        }
      }
    else
      {
      switch(param.dim)
        {
        case 2: return GreedyRunner<2, double>::Run(param); break;
        case 3: return GreedyRunner<3, double>::Run(param); break;
        case 4: return GreedyRunner<4, double>::Run(param); break;
        default: throw GreedyException("Wrong number of dimensions requested: %d", param.dim);
        }
      }
  }
  catch(std::exception &exc)
  {
    std::cerr << "ABORTING PROGRAM DUE TO RUNTIME EXCEPTION -- "
              << exc.what() << std::endl;
    return -1;
  }
}
