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
#ifndef GREEDYPARAMETERS_H
#define GREEDYPARAMETERS_H

#include <string>
#include <vector>
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include <ostream>

class CommandLineHelper;

struct ImagePairSpec
{
  std::string fixed;
  std::string moving;
  double weight;

  ImagePairSpec(std::string in_fixed, std::string in_moving, double in_weight = 1.0)
    : fixed(in_fixed), moving(in_moving), weight(in_weight) {}

  ImagePairSpec()
    : weight(1.0) {}
};

struct SmoothingParameters
{
  double sigma;
  bool physical_units;
  SmoothingParameters(double s, bool pu) : sigma(s), physical_units(pu) {}
  SmoothingParameters() : sigma(0.0), physical_units(true) {}

  bool operator != (const SmoothingParameters &other) {
    return sigma != other.sigma || physical_units != other.physical_units;
  }
};

enum RigidSearchRotationMode
{
  RANDOM_NORMAL_ROTATION,
  ANY_ROTATION,
  ANY_ROTATION_AND_FLIP
};

struct RigidSearchSpec
{
  RigidSearchRotationMode mode;
  int iterations;
  double sigma_xyz;
  double sigma_angle;

  RigidSearchSpec() : mode(RANDOM_NORMAL_ROTATION),
    iterations(0), sigma_xyz(0.0), sigma_angle(0.0) {}
};

struct InterpSpec
{
  enum InterpMode { LINEAR, NEAREST, LABELWISE };

  InterpMode mode;
  SmoothingParameters sigma;
  double outside_value;

  InterpSpec(InterpMode in_mode = LINEAR,
             double in_sigma = 0.5,
             bool in_sigma_physical_units = false,
             double in_outside_value = 0.0)
    : mode(in_mode), sigma(in_sigma, in_sigma_physical_units), outside_value(in_outside_value) {}
};

struct ResliceSpec
{
  std::string moving;
  std::string output;
  InterpSpec interp;

  ResliceSpec(const std::string &in_moving = "",
              const std::string &in_output = "",
              InterpSpec in_interp =  InterpSpec())
    : moving(in_moving), output(in_output), interp(in_interp) {}
};

struct ResliceMeshSpec
{
  std::string fixed;
  std::string output;
};

struct TransformSpec
{
  // Transform file
  std::string filename;

  // Optional exponent (-1 for inverse, 0.5 for square root)
  double exponent;

  // Constructor
  TransformSpec(const std::string in_filename = std::string(), double in_exponent = 1.0)
    : filename(in_filename), exponent(in_exponent) {}
};

enum AffineInitMode
{
  VOX_IDENTITY = 0, // Identity mapping in voxel space
  RAS_IDENTITY,     // Identity mapping in physical space (i.e., use headers)
  RAS_FILENAME,     // User-specified matrix in physical space
  IMG_CENTERS,      // Match image centers, identity rotation in voxel space
  IMG_SIDE,         // Match image sides,
  MOMENTS_1,        // Match centers of mass,
  MOMENTS_2         // Match inertia tensors
};

struct GreedyResliceParameters
{
  // For reslice mode
  std::vector<ResliceSpec> images;
  std::vector<ResliceMeshSpec> meshes;

  // Reference image
  std::string ref_image;

  // Chain of transforms
  std::vector<TransformSpec> transforms;

  // Output warp
  std::string out_composed_warp;

  // Output jacobian
  std::string out_jacobian_image;
};

// Parameters for inverse warp command
struct GreedyInvertWarpParameters
{
  std::string in_warp, out_warp;
};

struct GreedyJacobianParameters
{
  std::string in_warp, out_det_jac;
};


// Parameters for inverse warp command
struct GreedyWarpRootParameters
{
  std::string in_warp, out_warp;
};

// Parameters for the LBFGS optimizer
struct LBFGSParameters
{
  double ftol = 0.0, gtol = 0.0;
  int memory = 0;
};

template <class TAtomic>
class PerLevelSpec
{
public:
  PerLevelSpec() : m_UseCommon(false) {}
  PerLevelSpec(TAtomic common_value) { *this = common_value; }
  PerLevelSpec(std::vector<TAtomic> per_level_value) { *this = per_level_value; }

  TAtomic operator [] (unsigned int pos) const
  {
    return m_UseCommon ? m_CommonValue : m_ValueArray.at(pos);
  }

  PerLevelSpec<TAtomic> & operator = (TAtomic value)
  {
    m_CommonValue = value; m_UseCommon = true;
    return *this;
  }

  PerLevelSpec<TAtomic> & operator = (std::vector<TAtomic> per_level_value)
  {
    if(per_level_value.size() == 1)
      return (*this = per_level_value[0]);

    m_ValueArray = per_level_value; m_UseCommon = false;
    return *this;
  }

  bool CheckSize(unsigned int n_Levels) const
  {
    return m_UseCommon || m_ValueArray.size() == n_Levels;
  }

  bool operator != (const PerLevelSpec<TAtomic> &other)
  {
    if(m_UseCommon && other.m_UseCommon)
      {
      return m_CommonValue != m_CommonValue;
      }
    else if(!m_UseCommon && !other.m_UseCommon)
      {
      return m_ValueArray != other.m_ValueArray;
      }
    else return false;
  }

  void Print(std::ostream &oss) const 
    {
    if(m_UseCommon)
      oss << m_CommonValue;
    else
      {
      for(unsigned int i = 0; i < m_ValueArray.size(); i++)
        {
        if(i > 0) oss << "x";
        oss << m_ValueArray[i];
        }
      }
    }

protected:
  TAtomic m_CommonValue;
  std::vector<TAtomic> m_ValueArray;
  bool m_UseCommon;
};

template <class TAtomic>
std::ostream& operator << (std::ostream &oss, const PerLevelSpec<TAtomic> &val)
{
  val.Print(oss);
  return oss;
}

/**
 * An input set is a set of fixed/moving images with optional fixed/moving mask
 * and initial moving transform. More than one input set can be used when you want
 * to apply different masks to different inputs, for example when registering a
 * slice to neighbor slices in stack_greedy
 */
struct GreedyInputSet
{
  // Pairs of input images
  std::vector<ImagePairSpec> inputs;

  // Mask for the moving image
  std::string moving_mask;

  // Mask for the moving image
  std::string fixed_mask;

  // List of transforms to apply to the moving image before registration
  std::vector<TransformSpec> moving_pre_transforms;
};

struct GreedyParameters
{
  enum MetricType { SSD = 0, NCC, WNCC, MI, NMI, MAHALANOBIS };
  enum TimeStepMode { CONSTANT=0, SCALE, SCALEDOWN };
  enum Mode { GREEDY=0, AFFINE, BRUTE, RESLICE, INVERT_WARP, ROOT_WARP, JACOBIAN_WARP, MOMENTS, METRIC };
  enum AffineDOF { DOF_RIGID=6, DOF_SIMILARITY=7, DOF_AFFINE=12 };
  enum Verbosity { VERB_NONE=0, VERB_DEFAULT, VERB_VERBOSE, VERB_INVALID };

  // One or more input sets
  std::vector<GreedyInputSet> input_sets;

  // Output affine or warp
  std::string output;

  // Image dimension
  unsigned int dim = 2;

  // Number of threads to use, default 0 means all available
  int threads = 0;

  // Output for each iteration. This can be in the format "blah_%04d_%04d.mat" for
  // saving intermediate results into separate files. Or it can point to an object
  // in the GreedyAPI cache
  std::string output_intermediate;

  // Reslice parameters
  GreedyResliceParameters reslice_param;

  // Inversion parameters
  GreedyInvertWarpParameters invwarp_param;

  // Jacobian parameters
  GreedyJacobianParameters jacobian_param;

  // Root warp parameters
  GreedyWarpRootParameters warproot_param;

  // Registration mode
  Mode mode = GREEDY;

  // Debug dumo related
  bool flag_dump_moving = false, flag_dump_pyramid = false, flag_debug_deriv = false;
  int dump_frequency = 1;

  // Epsilon for derivative calculations (debug related)
  double deriv_epsilon = 1e-4;

  // Standard deviation of jitter noise added to the sampling grid during affine
  double affine_jitter = 0.5;

  // Background fill for reslicing operations
  double background = 0.0;

  // Smoothing parameters
  SmoothingParameters sigma_pre = { 1.7320508076, false };
  SmoothingParameters sigma_post = { 0.7071067812, false };

  // Which metric to use
  MetricType metric = SSD;

  // Time step scaling
  TimeStepMode time_step_mode = SCALE;

  // Iterations per level (i.e., 40x40x100)
  std::vector<int> iter_per_level = {{100, 100}};

  // Epsilon factor for each level
  PerLevelSpec<double> epsilon_per_level = 1.0;

  std::vector<int> metric_radius;

  std::vector<int> brute_search_radius;

  // An image used to specify the reference space
  std::string reference_space;

  // Amount of padding applied to the reference space when applying the moving pre-transform
  // This allows the moving image to extend past the fixed image when applying the moving
  // pre-transforms, so that during warping we can actually sample from outside of the fixed
  // image region.
  std::vector<int> reference_space_padding;

  // Initial affine transform mode
  AffineInitMode affine_init_mode = VOX_IDENTITY;

  // Degrees of freedom (rigid or affine)
  AffineDOF affine_dof = DOF_AFFINE;

  // Initial affine transform
  TransformSpec affine_init_transform;

  // Filename of initial warp
  std::string initial_warp;

  // Trim for the gradient mask
  std::vector<int> fixed_mask_trim_radius;

  // Whether the mask is dilated udint NCC
  bool flag_ncc_mask_dilate = false;

  // Inverse warp and root warp, for writing in deformable mode
  std::string inverse_warp, root_warp;

  // Exponent for scaling and squaring
  int warp_exponent = 6;

  // Precision for output warps (in voxel units, to save space)
  double warp_precision = 0.1;

  // Noise for NCC, relative to image intensity range
  double ncc_noise_factor = 0.001;

  // Debugging matrices
  bool flag_debug_aff_obj = false;

  // Rigid search
  RigidSearchSpec rigid_search;

  // Moments of inertia specification
  int moments_flip_determinant = 0;
  int moments_order = 1;
  bool flag_moments_id_covariance = false;

  // Stationary velocity (Vercauteren 2008 LogDemons) mode
  bool flag_stationary_velocity_mode = false;

  // Whether the Lie bracket is used in the y velocity update
  bool flag_stationary_velocity_mode_use_lie_bracket = false;

  // Incompressibility mode (Mansi 2011 iLogDemons)
  bool flag_incompressibility_mode = false;

  // Floating point precision?
  bool flag_float_math = false;

  // Whether to use an alternative solver for affine optimization
  bool flag_powell = false;

  // Weight applied to new image pairs
  double current_weight = 1.0;

  // Interpolation applied to new reslice image pairs
  InterpSpec current_interp;
  
  // Verbosity flag
  Verbosity verbosity = VERB_DEFAULT;

  // Optimization parameters
  LBFGSParameters lbfgs_param;

  // Where to store the metric gradient in metric computation mode
  std::string output_metric_gradient;

  // Data root (used when testing to provide relative paths)
  std::string data_root;

  // Data root (used when testing to provide relative paths)
  std::string dump_prefix;

  // Constructor
  GreedyParameters();

  // Read parameters from the
  bool ParseCommandLine(const std::string &cmd, CommandLineHelper &cl);

  // Get an existing filename
  std::string GetExistingFilename(CommandLineHelper &cl);

  // Generate a command line for current parameters
  std::string GenerateCommandLine();
};


#endif // GREEDYPARAMETERS_H
