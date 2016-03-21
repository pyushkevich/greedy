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

// Little helper functions
template <unsigned int VDim> class array_caster
{
public:
  template <class T> static itk::Size<VDim> to_itkSize(const T &t)
  {
    itk::Size<VDim> sz;
    for(int i = 0; i < VDim; i++)
      sz[i] = t[i];
    return sz;
  }
};

template <class TITKMatrix, class TVNLMatrix>
void vnl_matrix_to_itk_matrix(
    const TVNLMatrix &vmat,
    TITKMatrix &imat)
{
  for(int r = 0; r < TITKMatrix::RowDimensions; r++)
    for(int c = 0; c < TITKMatrix::ColumnDimensions; c++)
      imat(r,c) = static_cast<typename TITKMatrix::ValueType>(vmat(r,c));
}

template <class TITKVector, class TVNLVector>
void vnl_vector_to_itk_vector(
    const TVNLVector &vvec,
    TITKVector &ivec)
{
  for(int r = 0; r < TITKVector::Dimension; r++)
    ivec[r] = static_cast<typename TITKVector::ValueType>(vvec(r));
}

template <class TITKMatrix, class TVNL>
void itk_matrix_to_vnl_matrix(
    const TITKMatrix &imat,
    vnl_matrix_fixed<TVNL,TITKMatrix::RowDimensions,TITKMatrix::ColumnDimensions>  &vmat)
{
  for(int r = 0; r < TITKMatrix::RowDimensions; r++)
    for(int c = 0; c < TITKMatrix::ColumnDimensions; c++)
      vmat(r,c) = static_cast<TVNL>(imat(r,c));
}

template <class TITKMatrix, class TVNL>
void itk_matrix_to_vnl_matrix(
    const TITKMatrix &imat,
    vnl_matrix<TVNL>  &vmat)
{
  vmat.set_size(TITKMatrix::RowDimensions,TITKMatrix::ColumnDimensions);
  for(int r = 0; r < TITKMatrix::RowDimensions; r++)
    for(int c = 0; c < TITKMatrix::ColumnDimensions; c++)
      vmat(r,c) = static_cast<TVNL>(imat(r,c));
}

template <class TITKVector, class TVNL>
void itk_vector_to_vnl_vector(
    const TITKVector &ivec,
    vnl_vector_fixed<TVNL,TITKVector::Dimension> &vvec)
{
  for(int r = 0; r < TITKVector::Dimension; r++)
    vvec(r) = static_cast<TVNL>(ivec[r]);
}

template <class TITKVector, class TVNL>
void itk_vector_to_vnl_vector(
    const TITKVector &ivec,
    vnl_vector<TVNL> &vvec)
{
  vvec.set_size(TITKVector::Dimension);
  for(int r = 0; r < TITKVector::Dimension; r++)
    vvec(r) = static_cast<TVNL>(ivec[r]);
}





/**
 * A simple exception class with string formatting
 */
class GreedyException : public std::exception
{
public:

  GreedyException(const char *format, ...)
  {
    buffer = new char[4096];
    va_list args;
    va_start (args, format);
    vsprintf (buffer,format, args);
    va_end (args);
  }

  virtual const char* what() const throw() { return buffer; }

  virtual ~GreedyException() throw() { delete buffer; }

private:

  char *buffer;

};

int usage()
{
  printf("greedy: Paul's greedy diffeomorphic registration implementation\n");
  printf("Usage: \n");
  printf("  greedy [options]\n");
  printf("Required options: \n");
  printf("  -d DIM                 : Number of image dimensions\n");
  printf("  -i fix.nii mov.nii     : Image pair (may be repeated)\n");
  printf("  -o output.nii               : Output file\n");  
  printf("Mode specification: \n");
  printf("  -a                     : Perform affine registration and save to output (-o)\n");
  printf("  -brute radius          : Perform a brute force search around each voxel \n");
  printf("  -r [tran_spec]         : Reslice images instead of doing registration \n");
  printf("                               tran_spec is a series of warps, affine matrices\n");
  printf("Options in deformable / affine mode: \n");
  printf("  -w weight              : weight of the next -i pair\n");
  printf("  -m metric              : metric for the entire registration\n");
  printf("                               SSD:          sum of square differences (default)\n");
  printf("                               MI:           mutual information\n");
  printf("                               NMI:          normalized mutual information\n");
  printf("                               NCC <radius>: normalized cross-correlation\n");
  printf("  -e epsilon             : step size (default = 1.0)\n");
  printf("  -n NxNxN               : number of iterations per level of multi-res (100x100) \n");
  printf("  -threads N             : set the number of allowed concurrent threads\n");
  printf("  -gm mask.nii           : mask for gradient computation\n");
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
  printf("Specific to affine mode:\n");
  printf("  -dof N                 : Degrees of freedom for affine reg. 6=rigid, 12=affine\n");
  printf("  -jitter sigma          : Jitter (in voxel units) applied to sample points (def: 0.5)\n");
  printf("  -search N s_ang s_xyz  : Random search over rigid transforms (N iter) before starting optimization\n");
  printf("                           s_ang, s_xyz: sigmas for rot-n angle (degrees) and offset between image centers\n");
  printf("Specific to reslice mode: \n");
  printf("   -rf fixed.nii         : fixed image for reslicing\n");
  printf("   -rm mov.nii out.nii   : moving/output image pair (may be repeated)\n");
  printf("   -ri interp_mode       : interpolation for the next pair (NN, LINEAR*, LABEL sigma)\n");
  printf("For developers: \n");
  printf("  -debug-deriv           : enable periodic checks of derivatives (debug) \n");
  printf("  -debug-deriv-eps       : epsilon for derivative debugging \n");
  printf("  -debug-aff-obj         : plot affine objective in neighborhood of -ia matrix \n");
  printf("  -dump-moving           : dump moving image at each iter\n");
  printf("  -dump-freq N           : dump frequency\n");
  printf("  -powell                : use Powell's method instead of LGBFS\n");
  printf("  -float                 : use single precision floating point (off by default)\n");

  return -1;
}

struct ImagePairSpec
{
  std::string fixed;
  std::string moving;
  double weight;
};

struct SmoothingParameters
{
  double sigma;
  bool physical_units;
  SmoothingParameters(double s, bool pu) : sigma(s), physical_units(pu) {}
  SmoothingParameters() : sigma(0.0), physical_units(true) {}
};

struct RigidSearchSpec
{
  int iterations;
  double sigma_xyz;
  double sigma_angle;

  RigidSearchSpec() : iterations(0), sigma_xyz(0.0), sigma_angle(0.0) {}
};

struct InterpSpec
{
  enum InterpMode { LINEAR, NEAREST, LABELWISE };

  InterpMode mode;
  SmoothingParameters sigma;

  InterpSpec() : mode(LINEAR), sigma(0.5, false) {}
};

struct ResliceSpec
{
  std::string moving;
  std::string output;
  InterpSpec interp;
};

struct TransformSpec
{
  // Transform file
  std::string filename;

  // Optional exponent (-1 for inverse, 0.5 for square root)
  double exponent;
};

enum AffineInitMode
{
  VOX_IDENTITY = 0, // Identity mapping in voxel space
  RAS_IDENTITY,     // Identity mapping in physical space (i.e., use headers)
  RAS_FILENAME,     // User-specified matrix in physical space
  IMG_CENTERS       // Match image centers, identity rotation in voxel space
};

struct GreedyResliceParameters
{
  // For reslice mode
  std::vector<ResliceSpec> images;

  // Reference image
  std::string ref_image;

  // Chain of transforms
  std::vector<TransformSpec> transforms;
};

struct GreedyParameters
{
  enum MetricType { SSD = 0, NCC, MI, NMI };
  enum TimeStepMode { CONST=0, SCALE, SCALEDOWN };
  enum Mode { GREEDY=0, AFFINE, BRUTE, RESLICE };
  enum AffineDOF { DOF_RIGID=6, DOF_SIMILARITY=7, DOF_AFFINE=12 };

  std::vector<ImagePairSpec> inputs;
  std::string output;
  unsigned int dim; 

  // Reslice parameters
  GreedyResliceParameters reslice_param;

  // Registration mode
  Mode mode;

  bool flag_dump_moving, flag_debug_deriv, flag_powell;
  int dump_frequency, threads;
  double epsilon;
  double deriv_epsilon;

  double affine_jitter;

  // Smoothing parameters
  SmoothingParameters sigma_pre, sigma_post;

  MetricType metric;
  TimeStepMode time_step_mode;

  // Iterations per level (i.e., 40x40x100)
  std::vector<int> iter_per_level;

  std::vector<int> metric_radius;

  std::vector<int> brute_search_radius;

  // List of transforms to apply to the moving image before registration
  std::vector<TransformSpec> moving_pre_transforms;

  // Initial affine transform
  AffineInitMode affine_init_mode;
  AffineDOF affine_dof;
  TransformSpec affine_init_transform;

  // Mask for gradient
  std::string gradient_mask;

  // Inverse warp
  std::string inverse_warp;
  int inverse_exponent;

  // Precision for output warps
  double warp_precision;

  // Noise for NCC
  double ncc_noise_factor;

  // Debugging matrices
  bool flag_debug_aff_obj;

  // Rigid search
  RigidSearchSpec rigid_search;

  // Floating point precision?
  bool flag_float_math;
};



// Helper function to map from ITK coordiante space to RAS space
template<unsigned int VDim, class TMat, class TVec>
void
GetVoxelSpaceToNiftiSpaceTransform(itk::ImageBase<VDim> *image,
                                   TMat &A,
                                   TVec &b)
{
  // Generate intermediate terms
  typedef typename TMat::element_type TReal;
  vnl_matrix<double> m_dir, m_ras_matrix;
  vnl_diag_matrix<double> m_scale, m_lps_to_ras;
  vnl_vector<double> v_origin, v_ras_offset;

  // Compute the matrix
  m_dir = image->GetDirection().GetVnlMatrix();
  m_scale.set(image->GetSpacing().GetVnlVector());
  m_lps_to_ras.set(vnl_vector<double>(VDim, 1.0));
  m_lps_to_ras[0] = -1;
  m_lps_to_ras[1] = -1;
  A = m_lps_to_ras * m_dir * m_scale;

  // Compute the vector
  v_origin = image->GetOrigin().GetVnlVector();
  b = m_lps_to_ras * v_origin;
}

template <unsigned int VDim, typename TReal = double>
class GreedyApproach
{
public:

  typedef GreedyApproach<VDim, TReal> Self;

  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef typename LDDMMType::ImageBaseType ImageBaseType;
  typedef typename LDDMMType::ImageType ImageType;
  typedef typename LDDMMType::ImagePointer ImagePointer;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::VectorImagePointer VectorImagePointer;
  typedef typename LDDMMType::CompositeImageType CompositeImageType;
  typedef typename LDDMMType::CompositeImagePointer CompositeImagePointer;

  typedef MultiImageOpticalFlowHelper<TReal, VDim> OFHelperType;

  struct ImagePair {
    ImagePointer fixed, moving;
    VectorImagePointer grad_moving;
    double weight;
  };


  static int Run(GreedyParameters &param);

  static int RunDeformable(GreedyParameters &param);

  static int RunAffine(GreedyParameters &param);

  static int RunBrute(GreedyParameters &param);

  static int RunReslice(GreedyParameters &param);

protected:

  static void ReadImages(GreedyParameters &param, OFHelperType &ofhelper);

  static void ResampleImages(GreedyParameters &param,
                             const std::vector<ImagePair> &imgRaw,
                             std::vector<ImagePair> &img,
                             int level);

  static void ReadTransformChain(const std::vector<TransformSpec> &tran_chain,
                                 ImageBaseType *ref_space,
                                 VectorImagePointer &out_warp);

  static vnl_matrix<double> MapAffineToPhysicalRASSpace(
      OFHelperType &of_helper, int level,
      typename OFHelperType::LinearTransformType *tran);

  static void MapPhysicalRASSpaceToAffine(
      OFHelperType &of_helper, int level,
      vnl_matrix<double> &Qp,
      typename OFHelperType::LinearTransformType *tran);

  class AbstractAffineCostFunction : public vnl_cost_function
  {
  public:
    typedef typename OFHelperType::LinearTransformType TransformType;

    AbstractAffineCostFunction(int n_unknowns) : vnl_cost_function(n_unknowns) {}
    virtual vnl_vector<double> GetCoefficients(TransformType *tran) = 0;
    virtual void GetTransform(const vnl_vector<double> &coeff, TransformType *tran) = 0;
    virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g) = 0;
  };

  /**
   * Pure affine cost function - parameters are elements of N x N matrix M.
   * Transformation takes place in voxel coordinates - not physical coordinates (for speed)
   */
  class PureAffineCostFunction : public AbstractAffineCostFunction
  {
  public:
    typedef typename OFHelperType::LinearTransformType TransformType;

    // Construct the function
    PureAffineCostFunction(GreedyParameters *param, int level, OFHelperType *helper);

    // Get the parameters for the specified initial transform
    vnl_vector<double> GetCoefficients(TransformType *tran)
    {
      vnl_vector<double> x_true(this->get_number_of_unknowns());
      flatten_affine_transform(tran, x_true.data_block());
      return x_true;
    }

    // Get the transform for the specificed coefficients
    void GetTransform(const vnl_vector<double> &coeff, TransformType *tran)
    {
      unflatten_affine_transform(coeff.data_block(), tran);
    }

    // Get the preferred scaling for this function given image dimensions
    virtual vnl_vector<double> GetOptimalParameterScaling(const itk::Size<VDim> &image_dim);

    // Cost function computation
    virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g);

  protected:

    // Data needed to compute the cost function
    GreedyParameters *m_Param;
    OFHelperType *m_OFHelper;
    int m_Level;

    // Storage for the gradient of the similarity map
    VectorImagePointer m_Phi, m_GradMetric, m_GradMask;
    ImagePointer m_Metric, m_Mask;
  };

  /**
   * Physical space affine cost function - parameters are elements of affine transform in
   * physical RAS space.
   */
  class PhysicalSpaceAffineCostFunction : public AbstractAffineCostFunction
  {
  public:
    typedef typename OFHelperType::LinearTransformType TransformType;

    PhysicalSpaceAffineCostFunction(GreedyParameters *param, int level, OFHelperType *helper);
    virtual vnl_vector<double> GetCoefficients(TransformType *tran);
    virtual void GetTransform(const vnl_vector<double> &coeff, TransformType *tran);
    virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g);
    virtual vnl_vector<double> GetOptimalParameterScaling(const itk::Size<VDim> &image_dim);

    void map_phys_to_vox(const vnl_vector<double> &x_phys, vnl_vector<double> &x_vox);

  protected:
    PureAffineCostFunction m_PureFunction;

    // Voxel to physical transforms for fixed, moving image
    typedef vnl_matrix_fixed<double, VDim, VDim> Mat;
    typedef vnl_vector_fixed<double, VDim> Vec;

    Mat Q_fix, Q_mov, Q_fix_inv, Q_mov_inv;
    Vec b_fix, b_mov, b_fix_inv, b_mov_inv;

    vnl_matrix<double> J_phys_vox;
  };

  /** Abstract scaling cost function - wraps around another cost function and provides scaling */
  class ScalingCostFunction : public AbstractAffineCostFunction
  {
  public:
    typedef typename OFHelperType::LinearTransformType TransformType;

    // Construct the function
    ScalingCostFunction(AbstractAffineCostFunction *pure_function, const vnl_vector<double> &scaling)
      : AbstractAffineCostFunction(pure_function->get_number_of_unknowns()),
        m_PureFunction(pure_function), m_Scaling(scaling) {}

    // Get the parameters for the specified initial transform
    vnl_vector<double> GetCoefficients(TransformType *tran)
    {
      vnl_vector<double> x_true = m_PureFunction->GetCoefficients(tran);
      return element_product(x_true, m_Scaling);
    }

    // Get the transform for the specificed coefficients
    void GetTransform(const vnl_vector<double> &coeff, TransformType *tran)
    {
      vnl_vector<double> x_true = element_quotient(coeff, m_Scaling);
      m_PureFunction->GetTransform(x_true, tran);
    }

    // Cost function computation
    virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g)
    {
      // Scale the parameters so they are in unscaled units
      vnl_vector<double> x_scaled = element_quotient(x, m_Scaling);

      // Call the wrapped method
      if(g)
        {
        vnl_vector<double> g_scaled(x_scaled.size());
        m_PureFunction->compute(x_scaled, f, &g_scaled);
        *g = element_quotient(g_scaled, m_Scaling);
        }
      else
        {
        m_PureFunction->compute(x_scaled, f, g);
        }
    }

    const vnl_vector<double> &GetScaling() { return m_Scaling; }

  protected:

    // Data needed to compute the cost function
    AbstractAffineCostFunction *m_PureFunction;
    vnl_vector<double> m_Scaling;
  };

  /** Cost function for rigid registration */
  class RigidCostFunction : public AbstractAffineCostFunction
  {
  public:
    typedef typename OFHelperType::LinearTransformType TransformType;
    typedef vnl_vector_fixed<double, VDim> Vec3;
    typedef vnl_matrix_fixed<double, VDim, VDim> Mat3;


    RigidCostFunction(GreedyParameters *param, int level, OFHelperType *helper);
    vnl_vector<double> GetCoefficients(TransformType *tran);
    void GetTransform(const vnl_vector<double> &coeff, TransformType *tran);
    virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g);

    // Get the preferred scaling for this function given image dimensions
    virtual vnl_vector<double> GetOptimalParameterScaling(const itk::Size<VDim> &image_dim);

    // Create a random set of parameters, such that on average point C_fixed maps to point C_mov
    vnl_vector<double> GetRandomCoeff(const vnl_vector<double> &xInit, vnl_random &randy, double sigma_angle, double sigma_xyz,
                                     const Vec3 &C_fixed, const Vec3 &C_moving);

  protected:

    Mat3 GetRotationMatrix(const Vec3 &q);
    Vec3 GetAxisAngle(const Mat3 &R);

    // We wrap around a physical space affine function, since rigid in physical space is not
    // the same as rigid in voxel space
    PhysicalSpaceAffineCostFunction m_AffineFn;

  };
};

template <unsigned int VDim, typename TReal>
GreedyApproach<VDim, TReal>::PureAffineCostFunction
::PureAffineCostFunction(GreedyParameters *param, int level, OFHelperType *helper)
  : AbstractAffineCostFunction(VDim * (VDim + 1))
{
  // Store the data
  m_Param = param;
  m_OFHelper = helper;
  m_Level = level;

  // Allocate the working images
  m_Phi = VectorImageType::New();
  m_Phi->CopyInformation(helper->GetReferenceSpace(level));
  m_Phi->SetRegions(helper->GetReferenceSpace(level)->GetBufferedRegion());
  m_Phi->Allocate();

  m_GradMetric = VectorImageType::New();
  m_GradMetric->CopyInformation(helper->GetReferenceSpace(level));
  m_GradMetric->SetRegions(helper->GetReferenceSpace(level)->GetBufferedRegion());
  m_GradMetric->Allocate();

  m_GradMask = VectorImageType::New();
  m_GradMask->CopyInformation(helper->GetReferenceSpace(level));
  m_GradMask->SetRegions(helper->GetReferenceSpace(level)->GetBufferedRegion());
  m_GradMask->Allocate();

  m_Metric = ImageType::New();
  m_Metric->CopyInformation(helper->GetReferenceSpace(level));
  m_Metric->SetRegions(helper->GetReferenceSpace(level)->GetBufferedRegion());
  m_Metric->Allocate();

  m_Mask = ImageType::New();
  m_Mask->CopyInformation(helper->GetReferenceSpace(level));
  m_Mask->SetRegions(helper->GetReferenceSpace(level)->GetBufferedRegion());
  m_Mask->Allocate();
}


template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>::PureAffineCostFunction
::compute(const vnl_vector<double> &x, double *f, vnl_vector<double> *g)
{
  // Form a matrix/vector from x
  typename TransformType::Pointer tran = TransformType::New();

  // Set the components of the transform
  unflatten_affine_transform(x.data_block(), tran.GetPointer());

  // Compute the gradient
  double val = 0.0;
  if(g)
    {
    typename TransformType::Pointer grad = TransformType::New();

    if(m_Param->metric == GreedyParameters::SSD)
      {
      val = m_OFHelper->ComputeAffineMSDMatchAndGradient(
              m_Level, tran, m_Metric, m_Mask, m_GradMetric, m_GradMask, m_Phi, grad);

      flatten_affine_transform(grad.GetPointer(), g->data_block());
      }
    else if(m_Param->metric == GreedyParameters::NCC)
      {

      val = m_OFHelper->ComputeAffineNCCMatchAndGradient(
              m_Level, tran, array_caster<VDim>::to_itkSize(m_Param->metric_radius),
              m_Metric, m_Mask, m_GradMetric, m_GradMask, m_Phi, grad);

      flatten_affine_transform(grad.GetPointer(), g->data_block());

      // NCC should be maximized
      (*g) *= -10000.0;
      val *= -10000.0;
      }
    else if(m_Param->metric == GreedyParameters::MI || m_Param->metric == GreedyParameters::NMI)
      {
      val = m_OFHelper->ComputeAffineMIMatchAndGradient(
              m_Level, m_Param->metric == GreedyParameters::NMI,
              tran, m_Metric, m_Mask, m_GradMetric, m_GradMask, m_Phi, grad);

      flatten_affine_transform(grad.GetPointer(), g->data_block());

      val *= -10000.0;
      (*g) *= -10000.0;

      }
    }
  else
    {
    if(m_Param->metric == GreedyParameters::SSD)
      {
      val = m_OFHelper->ComputeAffineMSDMatchAndGradient(
              m_Level, tran, m_Metric, m_Mask, m_GradMetric, m_GradMask, m_Phi, NULL);
      }
    else if(m_Param->metric == GreedyParameters::NCC)
      {
      val = m_OFHelper->ComputeAffineNCCMatchAndGradient(
              m_Level, tran, array_caster<VDim>::to_itkSize(m_Param->metric_radius)
              , m_Metric, m_Mask, m_GradMetric, m_GradMask, m_Phi, NULL);

      // NCC should be maximized
      val *= -10000.0;
      }
    else if(m_Param->metric == GreedyParameters::MI || m_Param->metric == GreedyParameters::NMI)
      {
      val = m_OFHelper->ComputeAffineMIMatchAndGradient(
              m_Level, m_Param->metric == GreedyParameters::NMI,
              tran, m_Metric, m_Mask, m_GradMetric, m_GradMask, m_Phi, NULL);

      val *= -10000.0;
      }
    }

  if(f)
    *f = val;
}


template <unsigned int VDim, typename TReal>
vnl_vector<double>
GreedyApproach<VDim, TReal>::PureAffineCostFunction
::GetOptimalParameterScaling(const itk::Size<VDim> &image_dim)
{
  // Initialize the scaling vector
  vnl_vector<double> scaling(this->get_number_of_unknowns());

  // Set the scaling of the parameters based on image dimensions. This makes it
  // possible to set tolerances in units of voxels. The order of change in the
  // parameters is comparable to the displacement of any point inside the image
  typename TransformType::MatrixType matrix;
  typename TransformType::OffsetType offset;

  for(int i = 0; i < VDim; i++)
    {
    offset[i] = 1.0;
    for(int j = 0; j < VDim; j++)
      matrix(i, j) = image_dim[j];
    }

  typename TransformType::Pointer transform = TransformType::New();
  transform->SetMatrix(matrix);
  transform->SetOffset(offset);
  flatten_affine_transform(transform.GetPointer(), scaling.data_block());

  return scaling;
}

/**
 * PHYSICAL SPACE COST FUNCTION - WRAPS AROUND AFFINE
 */
template <unsigned int VDim, typename TReal>
GreedyApproach<VDim, TReal>::PhysicalSpaceAffineCostFunction
::PhysicalSpaceAffineCostFunction(GreedyParameters *param, int level, OFHelperType *helper)
  : AbstractAffineCostFunction(VDim * (VDim + 1)), m_PureFunction(param, level, helper)
{
  // The rigid transformation must be rigid in physical space, not in voxel space
  // So in the constructor, we must compute the mappings from the two spaces
  GetVoxelSpaceToNiftiSpaceTransform(helper->GetReferenceSpace(level), Q_fix, b_fix);
  GetVoxelSpaceToNiftiSpaceTransform(helper->GetMovingReferenceSpace(level), Q_mov, b_mov);

  // Compute the inverse transformations
  Q_fix_inv = vnl_matrix_inverse<double>(Q_fix);
  b_fix_inv = - Q_fix_inv * b_fix;

  Q_mov_inv = vnl_matrix_inverse<double>(Q_mov);
  b_mov_inv = - Q_mov_inv * b_mov;

  // Take advantage of the fact that the transformation is linear in A and b to compute
  // the Jacobian of the transformation ahead of time, and "lazily", using finite differences
  int n = VDim * (VDim + 1);
  J_phys_vox.set_size(n, n);
  vnl_vector<double> x_phys(n, 0), x_vox_0(n), x_vox(n);

  // Voxel parameter vector corresponding to zero transform
  this->map_phys_to_vox(x_phys, x_vox_0);

  // Compute each column of the jacobian
  for(int i = 0; i < n; i++)
    {
    x_phys.fill(0);
    x_phys[i] = 1;
    this->map_phys_to_vox(x_phys, x_vox);
    J_phys_vox.set_column(i, x_vox - x_vox_0);
    }


}

template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>::PhysicalSpaceAffineCostFunction
::map_phys_to_vox(const vnl_vector<double> &x_phys, vnl_vector<double> &x_vox)
{
  Mat A_phys;
  Vec b_phys;

  // unflatten the input parameters into A and b
  unflatten_affine_transform(x_phys.data_block(), A_phys, b_phys);

  // convert into voxel-space affine transform
  Mat A_vox = Q_mov_inv * A_phys * Q_fix;
  Vec b_vox = Q_mov_inv * (A_phys * b_fix + b_phys) + b_mov_inv;

  // Flatten back
  x_vox.set_size(m_PureFunction.get_number_of_unknowns());
  flatten_affine_transform(A_vox, b_vox, x_vox.data_block());
}


template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>::PhysicalSpaceAffineCostFunction
::compute(const vnl_vector<double> &x, double *f, vnl_vector<double> *g)
{
  // Map to voxel space
  vnl_vector<double> x_vox(m_PureFunction.get_number_of_unknowns());
  this->map_phys_to_vox(x, x_vox);

  // Do we need the gradient?
  if(g)
    {
    // Compute the function and gradient wrt voxel parameters
    vnl_vector<double> g_vox(m_PureFunction.get_number_of_unknowns());
    m_PureFunction.compute(x_vox, f, &g_vox);

    // Transform voxel-space gradient into physical-space gradient
    *g = J_phys_vox.transpose() * g_vox;
    }
  else
    {
    // Just compute the function
    m_PureFunction.compute(x_vox, f, NULL);
    }
}

template <unsigned int VDim, typename TReal>
vnl_vector<double>
GreedyApproach<VDim, TReal>::PhysicalSpaceAffineCostFunction
::GetOptimalParameterScaling(const itk::Size<VDim> &image_dim)
{
  // TODO: work out scaling for this
  return m_PureFunction.GetOptimalParameterScaling(image_dim);
}

template <unsigned int VDim, typename TReal>
vnl_vector<double>
GreedyApproach<VDim, TReal>::PhysicalSpaceAffineCostFunction
::GetCoefficients(TransformType *tran)
{
  // The input transform is in voxel space, we must return parameters in physical space
  Mat A_vox, A_phys;
  Vec b_vox, b_phys;

  itk_matrix_to_vnl_matrix(tran->GetMatrix(), A_vox);
  itk_vector_to_vnl_vector(tran->GetOffset(), b_vox);

  // convert into physical-space affine transform
  A_phys = Q_mov * A_vox * Q_fix_inv;
  b_phys = Q_mov * (b_vox - b_mov_inv) - A_phys * b_fix;

  // Flatten
  vnl_vector<double> x(m_PureFunction.get_number_of_unknowns());
  flatten_affine_transform(A_phys, b_phys, x.data_block());

  return x;
}

template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>::PhysicalSpaceAffineCostFunction
::GetTransform(const vnl_vector<double> &x, TransformType *tran)
{
  // Get voxel-space tranform corresponding to the parameters x
  vnl_vector<double> x_vox(m_PureFunction.get_number_of_unknowns());
  this->map_phys_to_vox(x, x_vox);

  // Unflatten into a transform
  unflatten_affine_transform(x_vox.data_block(), tran);
}



/**
 * RIGID COST FUNCTION - WRAPS AROUND AFFINE
 */
template <unsigned int VDim, typename TReal>
GreedyApproach<VDim, TReal>::RigidCostFunction
::RigidCostFunction(GreedyParameters *param, int level, OFHelperType *helper)
  : AbstractAffineCostFunction(VDim * 2), m_AffineFn(param, level, helper)
{
}

template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>::RigidCostFunction
::compute(const vnl_vector<double> &x, double *f, vnl_vector<double> *g)
{
  // Place parameters into q and b
  Vec3 q, b;
  q[0] = x[0]; q[1] = x[1]; q[2] = x[2];
  b[0] = x[3]; b[1] = x[4]; b[2] = x[5];

  // Compute theta
  double theta = q.magnitude();

  // Predefine the rotation matrix
  Mat3 R; R.set_identity();

  // Create the Q matrix
  Mat3 Qmat; Qmat.fill(0.0);
  Qmat(0,1) = -q[2]; Qmat(1,0) =  q[2];
  Qmat(0,2) =  q[1]; Qmat(2,0) = -q[1];
  Qmat(1,2) = -q[0]; Qmat(2,1) =  q[0];

  // Compute the square of the matrix
  Mat3 QQ = vnl_matrix_fixed_mat_mat_mult(Qmat, Qmat);

  // A small epsilon for which a better approximation is R = I + Q
  double eps = 1.0e-4;
  double a1, a2;

  // When theta = 0, rotation is identity
  if(theta > eps)
    {
    // Compute the constant terms in the Rodriguez formula
    a1 = sin(theta) / theta;
    a2 = (1 - cos(theta)) / (theta * theta);

    // Compute the rotation matrix
    R += a1 * Qmat + a2 * QQ;
    }
  else
    {
    R += Qmat;
    }

  // Now we have a rotation and a translation, convert to parameters for the affine function
  vnl_vector<double> x_affine(m_AffineFn.get_number_of_unknowns());
  flatten_affine_transform(R, b, x_affine.data_block());

  // Split depending on whether there is gradient to compute
  if(g)
    {
    // Create a vector to store the affine gradient
    vnl_vector<double> g_affine(m_AffineFn.get_number_of_unknowns());
    m_AffineFn.compute(x_affine, f, &g_affine);

    // Compute the matrices d_Qmat
    Mat3 d_Qmat[3], d_R[3];
    d_Qmat[0].fill(0); d_Qmat[0](1,2) = -1; d_Qmat[0](2,1) =  1;
    d_Qmat[1].fill(0); d_Qmat[1](0,2) =  1; d_Qmat[1](2,0) = -1;
    d_Qmat[2].fill(0); d_Qmat[2](0,1) = -1; d_Qmat[2](1,0) =  1;

    // Compute partial derivatives of R wrt q
    if(theta > eps)
      {
      // Compute the scaling factors in the Rodriguez formula
      double d_a1 = (theta * cos(theta) - sin(theta)) / (theta * theta * theta);
      double d_a2 = (theta * sin(theta) + 2 * cos(theta) - 2) /
                    (theta * theta * theta * theta);

      // Loop over the coordinate and compute the derivative of the rotation matrix wrt x
      for(int p = 0; p < 3; p++)
        {
        // Compute the gradient of the rotation with respect to q[p]
        d_R[p] = d_a1 * q[p] * Qmat +
                 a1 * d_Qmat[p] +
                 d_a2 * q[p] * vnl_matrix_fixed_mat_mat_mult(Qmat, Qmat)
                 + a2 * (vnl_matrix_fixed_mat_mat_mult(d_Qmat[p], Qmat) +
                         vnl_matrix_fixed_mat_mat_mult(Qmat, d_Qmat[p]));
        }
      }
    else
      {
      for(int p = 0; p < 3; p++)
        d_R[p] = d_Qmat[p];
      }

    // Create a matrix to hold the jacobian
    vnl_matrix<double> jac(m_AffineFn.get_number_of_unknowns(), 6);
    jac.fill(0.0);

    // Zero vector
    Vec3 zero_vec; zero_vec.fill(0.0);
    Mat3 zero_mat; zero_mat.fill(0.0);

    // Fill out the jacobian
    for(int p = 0; p < 3; p++)
      {
      // Fill the corresponding column
      vnl_vector<double> jac_col_q(m_AffineFn.get_number_of_unknowns());
      flatten_affine_transform(d_R[p], zero_vec, jac_col_q.data_block());
      jac.set_column(p, jac_col_q);

      // Also set column on the right (wrt translation)
      vnl_vector<double> jac_col_b(m_AffineFn.get_number_of_unknowns());
      Vec3 ep; ep.fill(0.0); ep[p] = 1;
      flatten_affine_transform(zero_mat, ep, jac_col_b.data_block());
      jac.set_column(p+3, jac_col_b);
      }

    // Multiply the gradient by the jacobian
    *g = jac.transpose() * g_affine;
    }
  else
    {
    m_AffineFn.compute(x_affine, f, NULL);
    }
}

template <unsigned int VDim, typename TReal>
vnl_vector<double>
GreedyApproach<VDim, TReal>::RigidCostFunction
::GetOptimalParameterScaling(const itk::Size<VDim> &image_dim)
{
  // Initialize the scaling vector
  vnl_vector<double> scaling(this->get_number_of_unknowns());

  // Scaling is harder for rotations. The rotation parameters are in units of
  // radians. We must figure out how many radians are equivalent to a point in
  // the image moving by a single voxel. That actually works out to be 1/dim.

  // So we take the average of the image dimensions and use that as scaling
  double mean_dim = 0;
  for(int i = 0; i < VDim; i++)
    mean_dim += image_dim[i] / VDim;
  scaling[0] = scaling[1] = scaling[2] = mean_dim;
  scaling[3] = scaling[4] = scaling[5] = 1.0;

  return scaling;
}

template <unsigned int VDim, typename TReal>
vnl_vector<double>
GreedyApproach<VDim, TReal>::RigidCostFunction
::GetCoefficients(TransformType *tran)
{
  // This affine transform is in voxel space. We must first map it into physical
  vnl_vector<double> x_aff_phys = m_AffineFn.GetCoefficients(tran);
  Mat3 A; Vec3 b;
  unflatten_affine_transform(x_aff_phys.data_block(), A, b);

  // Compute polar decomposition of the affine matrix
  vnl_svd<double> svd(A);
  Mat3 R = svd.U() * svd.V().transpose();
  Vec3 q = this->GetAxisAngle(R);

  // Make result
  vnl_vector<double> x(6);
  x[0] = q[0]; x[1] = q[1]; x[2] = q[2];
  x[3] = b[0]; x[4] = b[1]; x[5] = b[2];

  return x;
}

template <unsigned int VDim, typename TReal>
typename GreedyApproach<VDim, TReal>::RigidCostFunction::Vec3
GreedyApproach<VDim, TReal>::RigidCostFunction
::GetAxisAngle(const Mat3 &R)
{
  double eps = 1e-4;
  double f_thresh = cos(eps);

  // Compute the matrix logarithm of R
  double f = (vnl_trace(R) - 1) / 2;
  Vec3 q;
  if(f >= f_thresh)
    {
    q[0] = R(2,1) - R(1,2);
    q[1] = R(0,2) - R(2,0);
    q[2] = R(1,0) - R(0,1);
    q *= 0.5;
    }
  else
    {
    double theta = acos(f);
    double sin_theta = sqrt(1 - f * f);
    q[0] = R(2,1) - R(1,2);
    q[1] = R(0,2) - R(2,0);
    q[2] = R(1,0) - R(0,1);
    q *= theta / (2 * sin_theta);
    }

  return q;
}



template <unsigned int VDim, typename TReal>
vnl_vector<double>
GreedyApproach<VDim, TReal>::RigidCostFunction
::GetRandomCoeff(const vnl_vector<double> &xInit, vnl_random &randy, double sigma_angle, double sigma_xyz,
                 const Vec3 &C_fixed, const Vec3 &C_moving)
{
  // Generate a random rotation using given angles
  Vec3 q;
  for(int d = 0; d < 3; d++)
    q[d] = randy.normal() * sigma_angle;
  Mat3 R = this->GetRotationMatrix(q);

  // Generate a rotation matrix for the initial parameters
  Vec3 qInit;
  for(int d = 0; d < 3; d++)
    qInit[d] = xInit[d];
  Mat3 R_init = this->GetRotationMatrix(qInit);

  // Combined rotation
  Mat3 R_comb = R * R_init;

  // Take the log map
  Vec3 q_comb = this->GetAxisAngle(R_comb);

  // Generate the offset
  Vec3 b = C_moving - R_comb * C_fixed;

  // Apply random offset
  for(int d = 0; d < 3; d++)
    b[d] += randy.normal() * sigma_xyz;

  // Generate output vector
  vnl_vector<double> x(6);
  x[0] = q_comb[0];
  x[1] = q_comb[1];
  x[2] = q_comb[2];
  x[3] = b[0];
  x[4] = b[1];
  x[5] = b[2];

  return x;
}

template <unsigned int VDim, typename TReal>
typename GreedyApproach<VDim, TReal>::RigidCostFunction::Mat3
GreedyApproach<VDim, TReal>::RigidCostFunction
::GetRotationMatrix(const Vec3 &q)
{
  // Compute theta
  double theta = q.magnitude();

  // Predefine the rotation matrix
  Mat3 R; R.set_identity();

  // Create the Q matrix
  Mat3 Qmat; Qmat.fill(0.0);
  Qmat(0,1) = -q[2]; Qmat(1,0) =  q[2];
  Qmat(0,2) =  q[1]; Qmat(2,0) = -q[1];
  Qmat(1,2) = -q[0]; Qmat(2,1) =  q[0];

  // Compute the square of the matrix
  Mat3 QQ = vnl_matrix_fixed_mat_mat_mult(Qmat, Qmat);

  // When theta = 0, rotation is identity
  double eps = 1e-4;

  if(theta > eps)
    {
    // Compute the constant terms in the Rodriguez formula
    double a1 = sin(theta) / theta;
    double a2 = (1 - cos(theta)) / (theta * theta);

    // Compute the rotation matrix
    R += a1 * Qmat + a2 * QQ;
    }
  else
    {
    R += Qmat;
    }

  return R;
}

template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>::RigidCostFunction
::GetTransform(const vnl_vector<double> &x, TransformType *tran)
{
  // Place parameters into q and b
  Vec3 q, b;
  q[0] = x[0]; q[1] = x[1]; q[2] = x[2];
  b[0] = x[3]; b[1] = x[4]; b[2] = x[5];

  // Get the rotation matrix
  Mat3 R = this->GetRotationMatrix(q);

  // This gives us the physical space affine matrices. Flatten and map to voxel space
  vnl_vector<double> x_aff_phys(m_AffineFn.get_number_of_unknowns());
  flatten_affine_transform(R, b, x_aff_phys.data_block());
  m_AffineFn.GetTransform(x_aff_phys, tran);
}


/*
template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, double>::AffineCostFunction
::compute(const vnl_vector<double> &x, double *f, vnl_vector<double> *g)
{
  // Form a matrix/vector from x
  typename TransformType::Pointer tran = TransformType::New();

  // Set the components of the transform
  unflatten_affine_transform(x.data_block(), tran.GetPointer());

  // Compute the gradient
  double val = 0.0;
  if(g)
    {
    typename TransformType::Pointer grad = TransformType::New();
    val = m_OFHelper->ComputeAffineMatchAndGradient(m_Level, tran, grad);
    flatten_affine_transform(grad.GetPointer(), g->data_block());
    }
  else
    {
    val = m_OFHelper->ComputeAffineMatchAndGradient(m_Level, tran, NULL);
    }

  if(f)
    *f = val;
}
*
*
*/

#include "itkTransformFileReader.h"

template <typename TReal, unsigned int VDim>
vnl_matrix<double> ReadAffineMatrix(const TransformSpec &ts)
{
  // Physical (RAS) space transform matrix
  vnl_matrix<double> Qp(VDim+1, VDim+1);

  // Open the file and read the first line
  std::ifstream fin(ts.filename.c_str());
  std::string header_line, itk_header = "#Insight Transform File";
  std::getline(fin, header_line);

  if(header_line.substr(0, itk_header.size()) == itk_header)
    {
    fin.close();
    try
      {
      // First we try to load the transform using ITK code
      // This code is from c3d_affine_tool
      typedef itk::MatrixOffsetTransformBase<double, VDim, VDim> MOTBType;
      typedef itk::AffineTransform<double, VDim> AffTran;
      itk::TransformFactory<MOTBType>::RegisterTransform();
      itk::TransformFactory<AffTran>::RegisterTransform();

      itk::TransformFileReader::Pointer fltReader = itk::TransformFileReader::New();
      fltReader->SetFileName(ts.filename.c_str());
      fltReader->Update();

      itk::TransformBase *base = fltReader->GetTransformList()->front();
      typedef itk::MatrixOffsetTransformBase<double, VDim, VDim> MOTBType;
      MOTBType *motb = dynamic_cast<MOTBType *>(base);

      Qp.set_identity();
      if(motb)
        {
        for(size_t r = 0; r < VDim; r++)
          {
          for(size_t c = 0; c < VDim; c++)
            {
            Qp(r,c) = motb->GetMatrix()(r,c);
            }
          Qp(r,3) = motb->GetOffset()[r];
          }

        // RAS - LPI nonsense
        if(VDim == 3)
          {
          Qp(2,0) *= -1; Qp(2,1) *= -1;
          Qp(0,2) *= -1; Qp(1,2) *= -1;
          Qp(0,3) *= -1; Qp(1,3) *= -1;
          }
        }
      }
    catch(...)
      {
      throw GreedyException("Unable to read ITK transform file %s", ts.filename.c_str());
      }
    }
  else
    {
    // Try reading C3D matrix format
    fin.seekg(0);
    for(size_t i = 0; i < VDim+1; i++)
      for(size_t j = 0; j < VDim+1; j++)
        if(fin.good())
          {
          fin >> Qp[i][j];
          }
    fin.close();
    }

  // Compute the exponent
  if(ts.exponent == 1.0)
    {
    return Qp;
    }
  else if(ts.exponent == -1.0)
    {
    return vnl_matrix_inverse<double>(Qp);
    }
  else
    {
    throw GreedyException("Transform exponent values of +1 and -1 are the only ones currently supported");
    }

  return Qp;
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::ReadImages(GreedyParameters &param, OFHelperType &ofhelper)
{
  // If the parameters include a sequence of transforms, apply it first
  VectorImagePointer moving_pre_warp;

  // Read the input images and stick them into an image array
  for(int i = 0; i < param.inputs.size(); i++)
    {
    // Read fixed
    typedef itk::ImageFileReader<CompositeImageType> ReaderType;
    typename ReaderType::Pointer readfix = ReaderType::New();
    readfix->SetFileName(param.inputs[i].fixed);
    readfix->Update();

    // Read moving
    typedef itk::ImageFileReader<CompositeImageType> ReaderType;
    typename ReaderType::Pointer readmov = ReaderType::New();
    readmov->SetFileName(param.inputs[i].moving);
    readmov->Update();

    // Read the pre-warps (only once)
    if(param.moving_pre_transforms.size() && moving_pre_warp.IsNull())
      {
      ReadTransformChain(param.moving_pre_transforms, readfix->GetOutput(), moving_pre_warp);
      }

    if(moving_pre_warp.IsNotNull())
      {
      // Create an image to store the warp
      CompositeImagePointer warped_moving;
      LDDMMType::alloc_cimg(warped_moving, readfix->GetOutput(),
                            readmov->GetOutput()->GetNumberOfComponentsPerPixel());

      // Interpolate the moving image using the transform chain
      LDDMMType::interp_cimg(readmov->GetOutput(), moving_pre_warp, warped_moving, false, true);

      // Add the image pair to the helper
      ofhelper.AddImagePair(readfix->GetOutput(), warped_moving, param.inputs[i].weight);
      }
    else
      {
      // Add to the helper object
      ofhelper.AddImagePair(readfix->GetOutput(), readmov->GetOutput(), param.inputs[i].weight);
      }
    }

  // Read the masks
  if(param.gradient_mask.size())
    {
    // Read gradient mask
    typedef itk::ImageFileReader<typename OFHelperType::FloatImageType> ReaderType;
    typename ReaderType::Pointer readmask = ReaderType::New();
    readmask->SetFileName(param.gradient_mask);
    readmask->Update();

    ofhelper.SetGradientMask(readmask->GetOutput());
    }

  // Generate the optimized composite images. For the NCC metric, we add random noise to
  // the composite images, specified in units of the interquartile intensity range.
  double noise = (param.metric == GreedyParameters::NCC) ? param.ncc_noise_factor : 0.0;

  // Build the composite images
  ofhelper.BuildCompositeImages(noise);

  // If the metric is NCC, then also apply special processing to the gradient masks
  if(param.metric == GreedyParameters::NCC)
    ofhelper.DilateCompositeGradientMasksForNCC(array_caster<VDim>::to_itkSize(param.metric_radius));
}

#include <vnl/algo/vnl_lbfgs.h>

template <unsigned int VDim, typename TReal>
vnl_matrix<double>
GreedyApproach<VDim, TReal>
::MapAffineToPhysicalRASSpace(
    OFHelperType &of_helper, int level,
    typename OFHelperType::LinearTransformType *tran)
{
  // Map the transform to NIFTI units
  vnl_matrix<double> T_fix, T_mov, Q, A;
  vnl_vector<double> s_fix, s_mov, p, b;

  GetVoxelSpaceToNiftiSpaceTransform(of_helper.GetReferenceSpace(level), T_fix, s_fix);
  GetVoxelSpaceToNiftiSpaceTransform(of_helper.GetMovingReferenceSpace(level), T_mov, s_mov);

  itk_matrix_to_vnl_matrix(tran->GetMatrix(), A);
  itk_vector_to_vnl_vector(tran->GetOffset(), b);

  Q = T_mov * A * vnl_matrix_inverse<double>(T_fix);
  p = T_mov * b + s_mov - Q * s_fix;

  vnl_matrix<double> Qp(VDim+1, VDim+1);
  Qp.set_identity();
  for(int i = 0; i < VDim; i++)
    {
    Qp(i, VDim) = p(i);
    for(int j = 0; j < VDim; j++)
      Qp(i,j) = Q(i,j);
    }

  return Qp;
}

template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>
::MapPhysicalRASSpaceToAffine(
    OFHelperType &of_helper, int level,
    vnl_matrix<double> &Qp,
    typename OFHelperType::LinearTransformType *tran)
{
  // Map the transform to NIFTI units
  vnl_matrix<double> T_fix, T_mov, Q(VDim, VDim), A;
  vnl_vector<double> s_fix, s_mov, p(VDim), b;

  GetVoxelSpaceToNiftiSpaceTransform(of_helper.GetReferenceSpace(level), T_fix, s_fix);
  GetVoxelSpaceToNiftiSpaceTransform(of_helper.GetMovingReferenceSpace(level), T_mov, s_mov);

  for(int i = 0; i < VDim; i++)
    {
    p(i) = Qp(i, VDim);
    for(int j = 0; j < VDim; j++)
      Q(i,j) = Qp(i,j);
    }

  // A = vnl_matrix_inverse<double>(T_mov) * (Q * T_fix);
  // b = vnl_matrix_inverse<double>(T_mov) * (p - s_mov + Q * s_fix);
  A=vnl_svd<double>(T_mov).solve(Q * T_fix);
  b=vnl_svd<double>(T_mov).solve(p - s_mov + Q * s_fix);

  typename OFHelperType::LinearTransformType::MatrixType tran_A;
  typename OFHelperType::LinearTransformType::OffsetType tran_b;

  vnl_matrix_to_itk_matrix(A, tran_A);
  vnl_vector_to_itk_vector(b, tran_b);

  tran->SetMatrix(tran_A);
  tran->SetOffset(tran_b);
}

/**
 * Find a plane of symmetry in an image
 */
/*
template <unsigned int VDim, typename TReal>
vnl_vector<double>
GreedyApproach<VDim, TReal>
::FindSymmetryPlane(ImageType *image, int N, int n_search_pts)
{
  typedef vnl_vector_fixed<double, 3> Vec3;
  typedef vnl_matrix_fixed<double, 3, 3> Mat3;

  // Loop over direction on a sphere, using the Saff & Kuijlaars algorithm
  // https://perswww.kuleuven.be/~u0017946/publications/Papers97/art97a-Saff-Kuijlaars-MI/Saff-Kuijlaars-MathIntel97.pdf
  double phi = 0.0;
  double spiral_const = 3.6 / sqrt(N);
  for(int k = 0; k < n_sphere_pts; k++)
    {
    // Height of the k-th point
    double cos_theta = -1 * (2 * k) / (N - 1);
    double sin_theta = sqrt(1 - cos_theta * cos_theta);

    // Phase of the k-th point
    if(k > 0 && k < N-1)
      phi = fmod(phi_last + spiral_const / sin_theta, vnl_math::pi * 2);
    else
      phi = 0.0;

    // We now have the polar coordinates of the points, get cartesian coordinates
    Vec3 q;
    q[0] = sin_theta * cos(phi);
    q[1] = sin_theta * sin(phi);
    q[2] = cos_theta;

    // Now q * (x,y,z) = 0 defines a plane through the origin. We will test whether the image
    // is symmetric across this plane. We first construct the reflection matrix
    Mat3 R;
    R(0,0) =  1 - q[0] * q[0]; R(0,1) = -2 * q[1] * q[0]; R(0,2) = -2 * q[2] * q[0];
    R(1,0) = -2 * q[0] * q[1]; R(1,1) =  1 - q[1] * q[1]; R(1,2) = -2 * q[2] * q[1];
    R(2,0) = -2 * q[0] * q[2]; R(2,1) = -2 * q[1] * q[2]; R(2,2) =  1 - q[2] * q[2];

    // We must find the reasonable range of intercepts to search for. An intercept is reasonable
    // if the plane cuts the image volume in at least a 80/20 ratio (let's say)


    // This is a test axis of rotation. We will now measure the symmetry of the image across this axis
    // To do so, we will construct a series of flips across this direction

    }
}
*/

/**
 * This method performs initial alignment by first searching for a plane of symmetry
 * in each image, and then finding the transform between the planes of symmetry.
 *
 * The goal is to have an almost sure-fire method for brain alignment, yet generic
 * enough to work for other data as well.
 */
/*
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::SymmetrySearch(GreedyParameters &param, int level, OFHelperType *of_helper)
{

}
*/




template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunAffine(GreedyParameters &param)
{
  // Create an optical flow helper object
  OFHelperType of_helper;

  // Set the scaling factors for multi-resolution
  of_helper.SetDefaultPyramidFactors(param.iter_per_level.size());

  // Add random sampling jitter for affine stability at voxel edges
  of_helper.SetJitterSigma(param.affine_jitter);

  // Read the image pairs to register - this will also build the composite pyramids
  ReadImages(param, of_helper);

  // Matrix describing current transform in physical space
  vnl_matrix<double> Q_physical;

  // The number of resolution levels
  int nlevels = param.iter_per_level.size();

  // Iterate over the resolution levels
  for(unsigned int level = 0; level < nlevels; ++level)
    {
    // Define the affine cost function
    AbstractAffineCostFunction *pure_acf, *acf;
    if(param.affine_dof == GreedyParameters::DOF_RIGID)
      {
      RigidCostFunction *rigid_acf = new RigidCostFunction(&param, level, &of_helper);
      acf = new ScalingCostFunction(
              rigid_acf,
              rigid_acf->GetOptimalParameterScaling(
                of_helper.GetReferenceSpace(level)->GetBufferedRegion().GetSize()));
      pure_acf = rigid_acf;
      }
    else
      {
      //  PureAffineCostFunction *affine_acf = new PureAffineCostFunction(&param, level, &of_helper);
      PhysicalSpaceAffineCostFunction *affine_acf = new PhysicalSpaceAffineCostFunction(&param, level, &of_helper);
      acf = new ScalingCostFunction(
              affine_acf,
              affine_acf->GetOptimalParameterScaling(
                of_helper.GetReferenceSpace(level)->GetBufferedRegion().GetSize()));
      pure_acf = affine_acf;
      }

    // Current transform
    typedef typename OFHelperType::LinearTransformType TransformType;
    typename TransformType::Pointer tLevel = TransformType::New();

    // Set up the initial transform
    if(level == 0)
      {
      // Get the coefficients corresponding to the identity transform in voxel space
      tLevel->SetIdentity();
      vnl_vector<double> xIdent = acf->GetCoefficients(tLevel);

      // Use the provided initial affine as the starting point
      if(param.affine_init_mode == RAS_FILENAME)
        {
        // Read the initial affine transform from a file
        vnl_matrix<double> Qp = ReadAffineMatrix<double, VDim>(param.affine_init_transform);

        // Map this to voxel space
        MapPhysicalRASSpaceToAffine(of_helper, level, Qp, tLevel);
        }
      else if(param.affine_init_mode == RAS_IDENTITY)
        {
        // Physical space transform
        vnl_matrix<double> Qp(VDim+1, VDim+1); Qp.set_identity();

        // Map this to voxel space
        MapPhysicalRASSpaceToAffine(of_helper, level, Qp, tLevel);
        }

      // Get the new coefficients
      vnl_vector<double> xInit = acf->GetCoefficients(tLevel);

      // If the voxel-space transform is identity, apply a little bit of jitter
      if((xIdent - xInit).inf_norm() < 1e-4)
        {
        // Apply jitter
        vnl_random rndy(12345);
        for(int i = 0; i < xInit.size(); i++)
          xInit[i] += rndy.drand32(-0.4, 0.4);

        // Map back into transform format
        acf->GetTransform(xInit, tLevel);
        }

      // If the uses asks for rigid search, do it!
      if(param.rigid_search.iterations > 0)
        {
        // Create a pure rigid acf
        RigidCostFunction search_fun(&param, level, &of_helper);

        // Get the parameters corresponding to the current transform
        vnl_vector<double> xRigidInit = search_fun.GetCoefficients(tLevel);

        // Get center of fixed and moving images in physical space
        itk::Point<double, VDim> ctr_Fixed, ctr_Moving;
        itk::Index<VDim> idx_Fixed = of_helper.GetReferenceSpace(level)->GetBufferedRegion().GetIndex();
        for(int d = 0; d < 3; d++)
          idx_Fixed[d] += of_helper.GetReferenceSpace(level)->GetBufferedRegion().GetSize()[d] / 2;
        of_helper.GetReferenceSpace(level)->TransformIndexToPhysicalPoint(idx_Fixed, ctr_Fixed);

        itk::Index<VDim> idx_Moving = of_helper.GetMovingReferenceSpace(level)->GetBufferedRegion().GetIndex();
        for(int d = 0; d < 3; d++)
          idx_Moving[d] += of_helper.GetMovingReferenceSpace(level)->GetBufferedRegion().GetSize()[d] / 2;
        of_helper.GetMovingReferenceSpace(level)->TransformIndexToPhysicalPoint(idx_Moving, ctr_Moving);

        // At random, try a whole bunch of transforms, around 5 degrees
        vnl_random randy(12345);

        // TODO: make a heap of k best tries
        double fBest;
        vnl_vector<double> xBest = xRigidInit;
        search_fun.compute(xBest, &fBest, NULL);

        // Report the initial best
        std::cout << "Rigid search -> Initial best: " << fBest << " " << xBest << std::endl;

        for(int i = 0; i < param.rigid_search.iterations; i++)
          {
          // Get random coefficient
          // Compute a random rotation
          vnl_vector<double> xTry = search_fun.GetRandomCoeff(xRigidInit, randy,
                                                              param.rigid_search.sigma_angle,
                                                              param.rigid_search.sigma_xyz,
                                                              ctr_Fixed.GetVnlVector(), ctr_Moving.GetVnlVector());

          // Evaluate this transform
          double f;
          search_fun.compute(xTry, &f, NULL);

          if(f < fBest)
            {
            fBest = f;
            xBest = xTry;
            std::cout << "New best: " << fBest << " " << xBest << std::endl;
            }
          }

        xInit = xBest;
        search_fun.GetTransform(xInit, tLevel);
        }
      }
    else
      {
      // Update the transform from the last level
      MapPhysicalRASSpaceToAffine(of_helper, level, Q_physical, tLevel);
      }

    // Test derivatives
    // Convert to a parameter vector
    vnl_vector<double> xLevel = acf->GetCoefficients(tLevel.GetPointer());

    if(param.flag_debug_deriv)
      {
      // Test the gradient computation
      vnl_vector<double> xGrad(acf->get_number_of_unknowns(), 0.0);
      double f0;
      acf->compute(xLevel, &f0, &xGrad);

      // Propagate the jitter to the transform
      Q_physical = MapAffineToPhysicalRASSpace(of_helper, level, tLevel);
      std::cout << "Initial RAS Transform: " << std::endl << Q_physical  << std::endl;

      printf("ANL gradient: ");
      for(int i = 0; i < xGrad.size(); i++)
        printf("%11.4f ", xGrad[i]);
      printf("\n");

      vnl_vector<double> xGradN(acf->get_number_of_unknowns(), 0.0);
      for(int i = 0; i < acf->get_number_of_unknowns(); i++)
        {
        // double eps = (i % VDim == 0) ? 1.0e-2 : 1.0e-5;
        double eps = param.deriv_epsilon;
        double f1, f2, f3, f4;
        vnl_vector<double> x1 = xLevel, x2 = xLevel, x3 = xLevel, x4 = xLevel;
        x1[i] -= 2 * eps; x2[i] -= eps; x3[i] += eps; x4[i] += 2 * eps;

        // Four-point derivative computation
        acf->compute(x1, &f1, NULL);
        acf->compute(x2, &f2, NULL);
        acf->compute(x3, &f3, NULL);
        acf->compute(x4, &f4, NULL);

        xGradN[i] = (f1 - 8 * f2 + 8 * f3 - f4) / (12 * eps);
        }

      printf("NUM gradient: ");
      for(int i = 0; i < xGradN.size(); i++)
        printf("%11.4f ", xGradN[i]);
      printf("\n");

      std::cout << "f = " << f0 << std::endl;

      acf->GetTransform(xGrad, tLevel.GetPointer());
      std::cout << "A: " << std::endl
                << tLevel->GetMatrix() << std::endl
                << tLevel->GetOffset() << std::endl;

      acf->GetTransform(xGradN, tLevel.GetPointer());
      std::cout << "N: " << std::endl
                << tLevel->GetMatrix() << std::endl
                << tLevel->GetOffset() << std::endl;
      }

    if(param.flag_debug_aff_obj)
      {
      for(int k = -50; k < 50; k++)
        {
        printf("Obj\t%d\t", k);
        for(int i = 0; i < acf->get_number_of_unknowns(); i++)
          {
          vnl_vector<double> xTest = xLevel;
          xTest[i] = xLevel[i] + k * param.deriv_epsilon;
          double f; acf->compute(xTest, &f, NULL);
          printf("%12.8f\t", f);
          }
        printf("\n");
        }
        {
        vnl_vector<double> xTest = xLevel;
          {
          }
        printf("\n");
        }
      }

    // Run the minimization
    if(param.iter_per_level[level] > 0)
      {
      if(param.flag_powell)
        {
        // Set up the optimizer
        vnl_powell *optimizer = new vnl_powell(acf);
        optimizer->set_f_tolerance(1e-9);
        optimizer->set_x_tolerance(1e-4);
        optimizer->set_g_tolerance(1e-6);
        optimizer->set_trace(true);
        optimizer->set_verbose(true);
        optimizer->set_max_function_evals(param.iter_per_level[level]);

        optimizer->minimize(xLevel);
        delete optimizer;

        }
      else
        {
        // Set up the optimizer
        vnl_lbfgs *optimizer = new vnl_lbfgs(*acf);
        optimizer->set_f_tolerance(1e-9);
        optimizer->set_x_tolerance(1e-4);
        optimizer->set_g_tolerance(1e-6);
        optimizer->set_trace(true);
        optimizer->set_max_function_evals(param.iter_per_level[level]);

        optimizer->minimize(xLevel);
        delete optimizer;
        }

      // Get the final transform
      typename TransformType::Pointer tFinal = TransformType::New();
      acf->GetTransform(xLevel, tFinal.GetPointer());
      Q_physical = MapAffineToPhysicalRASSpace(of_helper, level, tFinal);
      }

    std::cout << "Final RAS Transform: " << std::endl << Q_physical << std::endl;

    delete acf;
    delete pure_acf;
    }

  // Write the final affine transform
  std::ofstream matrixFile;
  matrixFile.open(param.output.c_str());
  matrixFile << Q_physical;
  matrixFile.close();


  return 0;
}

#include "itkStatisticsImageFilter.h"

template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::Run(GreedyParameters &param)
{
  switch(param.mode)
    {
    case GreedyParameters::GREEDY:
      return Self::RunDeformable(param);
    case GreedyParameters::AFFINE:
      return Self::RunAffine(param);
    case GreedyParameters::BRUTE:
      return Self::RunBrute(param);
    case GreedyParameters::RESLICE:
      return Self::RunReslice(param);
    }

  return -1;
}


/**
 * This is the main function of the GreedyApproach algorithm
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunDeformable(GreedyParameters &param)
{
  // Create an optical flow helper object
  OFHelperType of_helper;

  // Set the scaling factors for multi-resolution
  of_helper.SetDefaultPyramidFactors(param.iter_per_level.size());

  // Read the image pairs to register
  ReadImages(param, of_helper);

  // An image pointer desribing the current estimate of the deformation
  VectorImagePointer uLevel = NULL;

  // The number of resolution levels
  int nlevels = param.iter_per_level.size();

  // Iterate over the resolution levels
  for(unsigned int level = 0; level < nlevels; ++level)
    {
    // Reference space
    ImageBaseType *refspace = of_helper.GetReferenceSpace(level);

    // Smoothing factors for this level, in physical units
    typename LDDMMType::Vec sigma_pre_phys =
        of_helper.GetSmoothingSigmasInPhysicalUnits(level, param.sigma_pre.sigma,
                                                    param.sigma_pre.physical_units);

    typename LDDMMType::Vec sigma_post_phys =
        of_helper.GetSmoothingSigmasInPhysicalUnits(level, param.sigma_post.sigma,
                                                    param.sigma_post.physical_units);

    // Report the smoothing factors used
    std::cout << "LEVEL " << level+1 << " of " << nlevels << std::endl;
    std::cout << "  Smoothing sigmas: " << sigma_pre_phys << ", " << sigma_post_phys << std::endl;

    // Set up timers for different critical components of the optimization
    itk::TimeProbe tm_Gradient, tm_Gaussian1, tm_Gaussian2, tm_Iteration;

    // Intermediate images
    ImagePointer iTemp = ImageType::New();
    VectorImagePointer viTemp = VectorImageType::New();
    VectorImagePointer uk = VectorImageType::New();
    VectorImagePointer uk1 = VectorImageType::New();

    // Allocate the intermediate data
    LDDMMType::alloc_vimg(uk, refspace);
    LDDMMType::alloc_img(iTemp, refspace);
    LDDMMType::alloc_vimg(viTemp, refspace);
    LDDMMType::alloc_vimg(uk1, refspace);

    // Initialize the deformation field from last iteration
    if(uLevel.IsNotNull())
      {
      LDDMMType::vimg_resample_identity(uLevel, refspace, uk);
      LDDMMType::vimg_scale_in_place(uk, 2.0);
      uLevel = uk;
      }
    else if(param.affine_init_mode != VOX_IDENTITY)
      {
      typename OFHelperType::LinearTransformType::Pointer tran =
          OFHelperType::LinearTransformType::New();

      if(param.affine_init_mode == RAS_FILENAME)
        {
        // Read the initial affine transform from a file
        vnl_matrix<double> Qp = ReadAffineMatrix<double, VDim>(param.affine_init_transform);

        // Map this to voxel space
        MapPhysicalRASSpaceToAffine(of_helper, level, Qp, tran);
        }
      else if(param.affine_init_mode == RAS_IDENTITY)
        {
        // Physical space transform
        vnl_matrix<double> Qp(VDim+1, VDim+1); Qp.set_identity();

        // Map this to voxel space
        MapPhysicalRASSpaceToAffine(of_helper, level, Qp, tran);
        }

      // Create an initial warp
      OFHelperType::AffineToField(tran, uk);
      uLevel = uk;

      itk::Index<VDim> test; test.Fill(24);
      std::cout << "Index 24x24x24 maps to " << uk->GetPixel(test) << std::endl;
      }

    // Iterate for this level
    for(unsigned int iter = 0; iter < param.iter_per_level[level]; iter++)
      {
      // Start the iteration timer
      tm_Iteration.Start();

      // Compute the gradient of objective
      double total_energy;

      if(param.metric == GreedyParameters::SSD)
        {
        // Begin gradient computation
        tm_Gradient.Start();

        vnl_vector<double> all_metrics =
            of_helper.ComputeOpticalFlowField(level, uk, iTemp, uk1, param.epsilon)  / param.epsilon;

        // If there is a mask, multiply the gradient by the mask
        if(param.gradient_mask.size())
          LDDMMType::vimg_multiply_in_place(uk1, of_helper.GetGradientMask(level));

        // End gradient computation
        tm_Gradient.Stop();

        printf("Lev:%2d  Itr:%5d  Met:[", level, iter);
        total_energy = 0.0;
        for(int i = 0;  i < all_metrics.size(); i++)
          {
          printf("  %8.6f", all_metrics[i]);
          total_energy += all_metrics[i];
          }
        printf("]  Tot: %8.6f\n", total_energy);
        }

      else if(param.metric == GreedyParameters::MI || param.metric == GreedyParameters::NMI)
        {
        // Begin gradient computation
        tm_Gradient.Start();

        vnl_vector<double> all_metrics =
            of_helper.ComputeMIFlowField(level, param.metric == GreedyParameters::NMI, uk, iTemp, uk1, param.epsilon);

        // If there is a mask, multiply the gradient by the mask
        if(param.gradient_mask.size())
          LDDMMType::vimg_multiply_in_place(uk1, of_helper.GetGradientMask(level));

        // End gradient computation
        tm_Gradient.Stop();

        printf("Lev:%2d  Itr:%5d  Met:[", level, iter);
        total_energy = 0.0;
        for(int i = 0;  i < all_metrics.size(); i++)
          {
          printf("  %8.6f", all_metrics[i]);
          total_energy += all_metrics[i];
          }
        printf("]  Tot: %8.6f\n", total_energy);
        }

      else
        {
        itk::Size<VDim> radius = array_caster<VDim>::to_itkSize(param.metric_radius);

        // Test derivative
        // total_energy = of_helper.ComputeNCCMetricAndGradient(level, uk, uk1, radius, param.epsilon);

        /*
        if(iter == 0)
          {

          // Perform a derivative check!

          itk::Index<VDim> test; test.Fill(24);
          typename VectorImageType::PixelType vtest = uk->GetPixel(test), vv;

          itk::ImageRegion<VDim> region = uk1->GetBufferedRegion();
          // region.ShrinkByRadius(1);

          double eps = param.epsilon;
          for(int d = 0; d < VDim; d++)
            {
            vv.Fill(0.5); vv[d] -= eps; uk->FillBuffer(vv);
            of_helper.ComputeNCCMetricImage(level, uk, radius, iTemp, uk1, 1.0);

            double a1 = 0.0;
            typedef itk::ImageRegionConstIterator<ImageType> Iter;
            for(Iter it(iTemp, region); !it.IsAtEnd(); ++it)
              {
              a1 += it.Get();
              }


            vv.Fill(0.5); vv[d] += eps; uk->FillBuffer(vv);
            of_helper.ComputeNCCMetricImage(level, uk, radius, iTemp, uk1, 1.0);

            double a2 = 0.0;
            typedef itk::ImageRegionConstIterator<ImageType> Iter;
            for(Iter it(iTemp, region); !it.IsAtEnd(); ++it)
              {
              a2 += it.Get();
              }

            std::cout << "NUM:" << (a2 - a1) / (2*eps) << std::endl;

            }

          vv.Fill(0.5); uk->FillBuffer(vv);
          total_energy = of_helper.ComputeNCCMetricImage(level, uk, radius, iTemp, uk1, 1.0);
          for(int d = 0; d < VDim; d++)
            {

            double ader = 0.0;
            typedef itk::ImageRegionConstIterator<VectorImageType> Iter;
            for(Iter it(uk1, region); !it.IsAtEnd(); ++it)
              {
              ader += it.Get()[d];
              }

            // itk::Index<VDim> test; test.Fill(24);
            // std::cout << "ANA:" << uk1->GetPixel(test) << std::endl;

            std::cout << "ANA:" << ader << std::endl;
            }
          }
          */

        // Begin gradient computation
        tm_Gradient.Start();

        // Compute the metric - no need to multiply by the mask, this happens already in the NCC metric code
        total_energy = of_helper.ComputeNCCMetricImage(level, uk, radius, iTemp, uk1, param.epsilon) / param.epsilon;

        // End gradient computation
        tm_Gradient.Stop();

        printf("Level %5d,  Iter %5d:    Energy = %8.4f\n", level, iter, total_energy);
        fflush(stdout);
        }

      // Dump the gradient image if requested
      if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
        {
        char fname[256];
        sprintf(fname, "dump_gradient_lev%02d_iter%04d.nii.gz", level, iter);
        LDDMMType::vimg_write(uk1, fname);
        }

      // We have now computed the gradient vector field. Next, we smooth it
      tm_Gaussian1.Start();
      LDDMMType::vimg_smooth_withborder(uk1, viTemp, sigma_pre_phys, 1);
      tm_Gaussian1.Stop();

      // After smoothing, compute the maximum vector norm and use it as a normalizing
      // factor for the displacement field
      if(param.time_step_mode == GreedyParameters::SCALE)
        LDDMMType::vimg_normalize_to_fixed_max_length(viTemp, iTemp, param.epsilon, false);
      else if (param.time_step_mode == GreedyParameters::SCALEDOWN)
        LDDMMType::vimg_normalize_to_fixed_max_length(viTemp, iTemp, param.epsilon, true);

      // Dump the smoothed gradient image if requested
      if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
        {
        char fname[256];
        sprintf(fname, "dump_optflow_lev%02d_iter%04d.nii.gz", level, iter);
        LDDMMType::vimg_write(viTemp, fname);
        }

      // Compute the updated deformation field - in uk1
      LDDMMType::interp_vimg(uk, viTemp, 1.0, uk1);
      LDDMMType::vimg_add_in_place(uk1, viTemp);

      // Dump if requested
      if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
        {
        char fname[256];
        sprintf(fname, "dump_uk1_lev%02d_iter%04d.nii.gz", level, iter);
        LDDMMType::vimg_write(uk1, fname);
        }

      // Another layer of smoothing
      tm_Gaussian2.Start();
      LDDMMType::vimg_smooth_withborder(uk1, uk, sigma_post_phys, 1);
      tm_Gaussian2.Stop();

      tm_Iteration.Stop();
      }

    // Store the end result
    uLevel = uk;

    // Compute the jacobian of the deformation field
    LDDMMType::field_jacobian_det(uk, iTemp);
    TReal jac_min, jac_max;
    LDDMMType::img_min_max(iTemp, jac_min, jac_max);
    printf("END OF LEVEL %5d    DetJac Range: %8.4f  to %8.4f \n", level, jac_min, jac_max);

    // Print timing information
    printf("  Avg. Gradient Time  : %6.4fs  %5.2f%% \n", tm_Gradient.GetMean(), tm_Gradient.GetMean() * 100.0 / tm_Iteration.GetMean());
    printf("  Avg. Gaussian Time  : %6.4fs  %5.2f%% \n", tm_Gaussian1.GetMean() + tm_Gaussian2.GetMean(),
           (tm_Gaussian1.GetMean() + tm_Gaussian2.GetMean()) * 100.0 / tm_Iteration.GetMean());
    printf("  Avg. Iteration Time : %6.4fs \n", tm_Iteration.GetMean());

    }

  // The transformation field is in voxel units. To work with ANTS, it must be mapped
  // into physical offset units - just scaled by the spacing?

  // Write the resulting transformation field
  of_helper.WriteCompressedWarpInPhysicalSpace(nlevels - 1, uLevel, param.output.c_str(), param.warp_precision);

  // If an inverse is requested, compute the inverse using the Chen 2008 fixed method.
  // A modification of this method is that if convergence is slow, we take the square
  // root of the forward transform.
  //
  // TODO: it would be more efficient to check the Lipschitz condition rather than
  // the brute force approach below
  //
  // TODO: the maximum checks should only be done over the region where the warp is
  // not going outside of the image. Right now, they are meaningless and we are doing
  // extra work when computing the inverse.
  if(param.inverse_warp.size())
    {
    // Compute the inverse
    VectorImagePointer uInverse = VectorImageType::New();
    LDDMMType::alloc_vimg(uInverse, uLevel);
    of_helper.ComputeDeformationFieldInverse(uLevel, uInverse, param.inverse_exponent);

    // Write the warp using compressed format
    of_helper.WriteCompressedWarpInPhysicalSpace(nlevels - 1, uInverse, param.inverse_warp.c_str(), param.warp_precision);
    }
  return 0;
}

/**
 * This function performs brute force search for similar patches. It generates a discrete displacement
 * field where every pixel in the fixed image is matched to the most similar pixel in the moving image
 * within a certain radius
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunBrute(GreedyParameters &param)
{
  // Check for valid parameters
  if(param.metric != GreedyParameters::NCC)
    {
    std::cerr << "Brute force search requires NCC metric only" << std::endl;
    return -1;
    }

  if(param.brute_search_radius.size() != VDim)
    {
    std::cerr << "Brute force search radius must be same dimension as the images" << std::endl;
    return -1;
    }

  // Create an optical flow helper object
  OFHelperType of_helper;

  // No multi-resolution
  of_helper.SetDefaultPyramidFactors(1);

  // Read the image pairs to register
  ReadImages(param, of_helper);

  // Reference space
  ImageBaseType *refspace = of_helper.GetReferenceSpace(0);

  // Intermediate images
  VectorImagePointer u_best = VectorImageType::New();
  VectorImagePointer u_curr = VectorImageType::New();
  ImagePointer m_curr = ImageType::New();
  ImagePointer m_best = ImageType::New();

  // Allocate the intermediate data
  LDDMMType::alloc_vimg(u_best, refspace);
  LDDMMType::alloc_vimg(u_curr, refspace);
  LDDMMType::alloc_img(m_best, refspace);
  LDDMMType::alloc_img(m_curr, refspace);

  // Allocate m_best to a negative value
  m_best->FillBuffer(-100.0);

  // Create a neighborhood for computing offsets
  itk::Neighborhood<float, VDim> dummy_nbr;
  itk::Size<VDim> search_rad = array_caster<VDim>::to_itkSize(param.brute_search_radius);
  itk::Size<VDim> metric_rad = array_caster<VDim>::to_itkSize(param.metric_radius);
  dummy_nbr.SetRadius(search_rad);

  // Iterate over all offsets
  for(int k = 0; k < dummy_nbr.Size(); k++)
    {
    // Get the offset corresponding to this iteration
    itk::Offset<VDim> offset = dummy_nbr.GetOffset(k);

    // Fill the deformation field with this offset
    typename LDDMMType::Vec vec_offset;
    for(int i = 0; i < VDim; i++)
      vec_offset[i] = offset[i];
    u_curr->FillBuffer(vec_offset);

    // Perform interpolation and metric computation
    of_helper.ComputeNCCMetricImage(0, u_curr, metric_rad, m_curr);

    // Temp: keep track of number of updates
    unsigned long n_updates = 0;

    // Out of laziness, just take a quick pass over the images
    typename VectorImageType::RegionType rgn = refspace->GetBufferedRegion();
    itk::ImageRegionIterator<VectorImageType> it_u(u_best, rgn);
    itk::ImageRegionConstIterator<ImageType> it_m_curr(m_curr, rgn);
    itk::ImageRegionIterator<ImageType> it_m_best(m_best, rgn);
    for(; !it_m_best.IsAtEnd(); ++it_m_best, ++it_m_curr, ++it_u)
      {
      float v_curr = it_m_curr.Value();
      if(v_curr > it_m_best.Value())
        {
        it_m_best.Set(v_curr);
        it_u.Set(vec_offset);
        ++n_updates;
        }
      }

    std::cout << "offset: " << offset << "     updates: " << n_updates << std::endl;
    }

  LDDMMType::vimg_write(u_best, param.output.c_str());
  LDDMMType::img_write(m_best, "mbest.nii.gz");

  return 0;
}


#include "itkWarpVectorImageFilter.h"
#include "itkWarpImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"


template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::ReadTransformChain(const std::vector<TransformSpec> &tran_chain,
                     ImageBaseType *ref_space,
                     VectorImagePointer &out_warp)
{
  // Create the initial transform and set it to zero
  out_warp = VectorImageType::New();
  LDDMMType::alloc_vimg(out_warp, ref_space);

  // Read the sequence of transforms
  for(int i = 0; i < tran_chain.size(); i++)
    {
    // Read the next parameter
    std::string tran = tran_chain[i].filename;

    // Determine if it's an affine transform
    if(itk::ImageIOFactory::CreateImageIO(tran.c_str(), itk::ImageIOFactory::ReadMode))
      {
      // Create a temporary warp
      VectorImagePointer warp_tmp = VectorImageType::New();
      LDDMMType::alloc_vimg(warp_tmp, ref_space);

      // Read the next warp
      VectorImagePointer warp_i = VectorImageType::New();
      LDDMMType::vimg_read(tran.c_str(), warp_i);

      // Now we need to compose the current transform and the overall warp.
      LDDMMType::interp_vimg(warp_i, out_warp, 1.0, warp_tmp, false, true);
      LDDMMType::vimg_add_in_place(out_warp, warp_tmp);
      }
    else
      {
      // Read the transform as a matrix
      vnl_matrix<double> mat = ReadAffineMatrix<double, VDim>(tran_chain[i]);
      vnl_matrix<double>  A = mat.extract(VDim, VDim);
      vnl_vector<double> b = mat.get_column(VDim).extract(VDim), q;

      // TODO: stick this in a filter to take advantage of threading!
      typedef itk::ImageRegionIteratorWithIndex<VectorImageType> IterType;
      for(IterType it(out_warp, out_warp->GetBufferedRegion()); !it.IsAtEnd(); ++it)
        {
        itk::Point<double, VDim> pt, pt2;
        typename VectorImageType::IndexType idx = it.GetIndex();

        // Get the physical position
        // TODO: this calls IsInside() internally, which limits efficiency
        out_warp->TransformIndexToPhysicalPoint(idx, pt);

        // Add the displacement (in DICOM coordinates) and
        for(int i = 0; i < VDim; i++)
          pt2[i] = pt[i] + it.Value()[i];

        // Switch to NIFTI coordinates
        pt2[0] = -pt2[0]; pt2[1] = -pt2[1];

        // Apply the matrix - get the transformed coordinate in DICOM space
        q = A * pt2.GetVnlVector() + b;
        q[0] = -q[0]; q[1] = -q[1];

        // Compute the difference in DICOM space
        for(int i = 0; i < VDim; i++)
          it.Value()[i] = q[i] - pt[i];
        }
      }
    }
}

#include "itkBinaryThresholdImageFilter.h"
//#include "itkRecursiveGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkNaryFunctorImageFilter.h"

template <class TInputImage, class TOutputImage>
class NaryLabelVotingFunctor
{
public:
  typedef NaryLabelVotingFunctor<TInputImage,TOutputImage> Self;
  typedef typename TInputImage::PixelType InputPixelType;
  typedef typename TOutputImage::PixelType OutputPixelType;
  typedef std::vector<OutputPixelType> LabelArray;

  NaryLabelVotingFunctor(const LabelArray &labels)
    : m_LabelArray(labels), m_Size(labels.size()) {}

  NaryLabelVotingFunctor() : m_Size(0) {}


  OutputPixelType operator() (const std::vector<InputPixelType> &pix)
  {
    InputPixelType best_val = pix[0];
    int best_index = 0;
    for(int i = 1; i < m_Size; i++)
      if(pix[i] > best_val)
        {
        best_val = pix[i];
        best_index = i;
        }

    return m_LabelArray[best_index];
  }

  bool operator != (const Self &other)
    { return other.m_LabelArray != m_LabelArray; }

protected:
  LabelArray m_LabelArray;
  int m_Size;
};

/**
 * Run the reslice code - simply apply a warp or set of warps to images
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunReslice(GreedyParameters &param)
{
  typedef typename OFHelperType::LinearTransformType TransformType;

  GreedyResliceParameters r_param = param.reslice_param;

  // Check the parameters
  if(!r_param.ref_image.size())
    throw GreedyException("A reference image (-rf) option is required for reslice commands");

  if(!r_param.images.size())
    throw GreedyException("At least one pair of moving/output images (-rm) is required for reslice commands");

  // Read the fixed as a plain image (we don't care if it's composite)
  ImagePointer ref = ImageType::New();
  LDDMMType::img_read(r_param.ref_image.c_str(), ref);
  itk::ImageBase<VDim> *ref_space = ref;

  // Read the transform chain
  VectorImagePointer warp;
  ReadTransformChain(param.reslice_param.transforms, ref_space, warp);

  // Process image pairs
  for(int i = 0; i < r_param.images.size(); i++)
    {
    const char *filename = r_param.images[i].moving.c_str();

    // Handle the special case of multi-label images
    if(r_param.images[i].interp.mode == InterpSpec::LABELWISE)
      {
      // The label image assumed to be an image of shorts
      typedef itk::Image<short, VDim> LabelImageType;
      typedef itk::ImageFileReader<LabelImageType> LabelReaderType;

      // Create a reader
      typename LabelReaderType::Pointer reader = LabelReaderType::New();
      reader->SetFileName(filename);
      reader->Update();
      typename LabelImageType::Pointer moving = reader->GetOutput();

      // Scan the unique labels in the image
      std::set<short> label_set;
      short *labels = moving->GetBufferPointer();
      int n_pixels = moving->GetPixelContainer()->Size();

      // Get the list of unique pixels
      short last_pixel = 0;
      for(int j = 0; j < n_pixels; j++)
        {
        short pixel = labels[j];
        if(last_pixel != pixel || i == 0)
          {
          label_set.insert(pixel);
          last_pixel = pixel;
          if(label_set.size() > 1000)
            throw GreedyException("Label wise interpolation not supported for image %s "
                                  "which has over 1000 distinct labels", filename);
          }
        }

      // Turn this set into an array
      std::vector<short> label_array(label_set.begin(), label_set.end());

      // Create a N-way voting filter
      typedef NaryLabelVotingFunctor<ImageType, LabelImageType> VotingFunctor;
      VotingFunctor vf(label_array);

      typedef itk::NaryFunctorImageFilter<ImageType, LabelImageType, VotingFunctor> VotingFilter;
      typename VotingFilter::Pointer fltVoting = VotingFilter::New();
      fltVoting->SetFunctor(vf);

      // Create a mini-pipeline of streaming filters
      for(int j = 0; j < label_array.size(); j++)
        {
        // Set up a threshold filter for this label
        typedef itk::BinaryThresholdImageFilter<LabelImageType, ImageType> ThresholdFilterType;
        typename ThresholdFilterType::Pointer fltThreshold = ThresholdFilterType::New();
        fltThreshold->SetInput(moving);
        fltThreshold->SetLowerThreshold(label_array[j]);
        fltThreshold->SetUpperThreshold(label_array[j]);
        fltThreshold->SetInsideValue(1.0);
        fltThreshold->SetOutsideValue(0.0);

        // Set up a smoothing filter for this label
        typedef itk::SmoothingRecursiveGaussianImageFilter<ImageType, ImageType> SmootherType;
        typename SmootherType::Pointer fltSmooth = SmootherType::New();
        fltSmooth->SetInput(fltThreshold->GetOutput());

        // Work out the sigmas for the filter
        if(r_param.images[i].interp.sigma.physical_units)
          {
          fltSmooth->SetSigma(r_param.images[i].interp.sigma.sigma);
          }
        else
          {
          typename SmootherType::SigmaArrayType sigma_array;
          for(int d = 0; d < VDim; d++)
            sigma_array[d] = r_param.images[i].interp.sigma.sigma * moving->GetSpacing()[d];
          fltSmooth->SetSigmaArray(sigma_array);
          }

        // TODO: we should really be coercing the output into a vector image to speed up interpolation!
        typedef FastWarpCompositeImageFilter<ImageType, ImageType, VectorImageType> InterpFilter;
        typename InterpFilter::Pointer fltInterp = InterpFilter::New();
        fltInterp->SetMovingImage(fltSmooth->GetOutput());
        fltInterp->SetDeformationField(warp);
        fltInterp->SetUsePhysicalSpace(true);

        fltInterp->Update();

        // Add to the voting filter
        fltVoting->SetInput(j, fltInterp->GetOutput());
        }

      // TODO: test out streaming!
      // Run this big pipeline
      fltVoting->Update();

      // Save
      typedef itk::ImageFileWriter<LabelImageType> WriterType;
      typename WriterType::Pointer writer = WriterType::New();
      writer->SetFileName(r_param.images[i].output.c_str());
      writer->SetInput(fltVoting->GetOutput());
      writer->Update();
      }
    else
      {
      // Read the input image
      CompositeImagePointer moving, warped;
      itk::ImageIOBase::IOComponentType comp =
          LDDMMType::cimg_read(filename, moving);

      // Allocate the warped image
      LDDMMType::alloc_cimg(warped, ref_space, moving->GetNumberOfComponentsPerPixel());

      // Perform the warp
      LDDMMType::interp_cimg(moving, warp, warped,
                             r_param.images[i].interp.mode == InterpSpec::NEAREST,
                             true);

      // Write, casting to the input component type
      LDDMMType::cimg_write(warped, r_param.images[i].output.c_str(), comp);
      }
    }


  return 0;
}




#include "itksys/SystemTools.hxx"

class CommandLineHelper
{
public:
  CommandLineHelper(int argc, char *argv[])
  {
    this->argc = argc;
    this->argv = argv;
    i = 1;
  }

  bool is_at_end()
  {
    return i >= argc;
  }

  /**
   * Just read the next arg (used internally)
   */
  const char *read_arg()
  {
    if(i >= argc)
      throw GreedyException("Unexpected end of command line arguments.");

    return argv[i++];
  }

  /**
   * Read a command (something that starts with a '-')
   */
  std::string read_command()
  {
    current_command = read_arg();
    if(current_command[0] != '-')
      throw GreedyException("Expected a command at position %d, instead got '%s'.", i, current_command.c_str());
    return current_command;
  }

  /**
   * Read a string that is not a command (may not start with a -)
   */
  std::string read_string()
  {
    std::string arg = read_arg();
    if(arg[0] == '-')
      throw GreedyException("Expected a string argument as parameter to '%s', instead got '%s'.", current_command.c_str(), arg.c_str());

    return arg;
  }


  /**
   * Get the number of free arguments to the current command. Use only for commands with
   * a priori unknown number of arguments. Otherwise, just use the get_ commands
   */
  int command_arg_count(int min_required = 0)
  {
    // Count the number of arguments
    int n_args = 0;
    for(int j = i; j < argc; j++, n_args++)
      if(argv[j][0] == '-')
        break;

    // Test for minimum required
    if(n_args < min_required)
      throw GreedyException(
          "Expected at least %d arguments to '%s', instead got '%d'",
          min_required, current_command.c_str(), n_args);

    return n_args;
  }

  /**
   * Read an existing filename
   */
  std::string read_existing_filename()
  {
    std::string file = read_arg();
    if(!itksys::SystemTools::FileExists(file.c_str()))
      throw GreedyException("File '%s' does not exist", file.c_str());

    return file;
  }

  /**
   * Read a transform specification, format file,number
   */
  TransformSpec read_transform_spec()
  {
    std::string spec = read_arg();
    size_t pos = spec.find_first_of(',');

    TransformSpec ts;
    ts.filename = spec.substr(0, pos);
    ts.exponent = 1.0;

    if(!itksys::SystemTools::FileExists(ts.filename.c_str()))
      throw GreedyException("File '%s' does not exist", ts.filename.c_str());

    if(pos != std::string::npos)
      {
      errno = 0; char *pend;
      std::string expstr = spec.substr(pos+1);
      ts.exponent = std::strtod(expstr.c_str(), &pend);

      if(errno || *pend)
        throw GreedyException("Expected a floating point number after comma in transform specification '%s', instead got '%s'",
                              current_command.c_str(), spec.substr(pos).c_str());

      }

    return ts;
  }

  /**
   * Read an output filename
   */
  std::string read_output_filename()
  {
    std::string file = read_arg();
    return file;
  }

  /**
   * Read a floating point value
   */
  double read_double()
  {
    std::string arg = read_arg();

    errno = 0; char *pend;
    double val = std::strtod(arg.c_str(), &pend);

    if(errno || *pend)
      throw GreedyException("Expected a floating point number as parameter to '%s', instead got '%s'",
                            current_command.c_str(), arg.c_str());

    return val;
  }

  /**
   * Check if a string ends with another string and return the
   * substring without the suffix
   */
  bool check_suffix(const std::string &source, const std::string &suffix, std::string &out_prefix)
  {
    int n = source.length(), m = suffix.length();
    if(n < m)
      return false;

    if(source.substr(n-m, m) != suffix)
      return false;

    out_prefix = source.substr(0, n-m);
    return true;
  }

  /**
   * Read a floating point value with units (mm or vox)
   */
  double read_scalar_with_units(bool &physical_units)
  {
    std::string arg = read_arg();
    std::string scalar;

    if(check_suffix(arg, "vox", scalar))
      physical_units = false;
    else if(check_suffix(arg, "mm", scalar))
      physical_units = true;
    else
      throw GreedyException("Parameter to '%s' should include units, e.g. '3vox' or '3mm', instead got '%s'",
                            current_command.c_str(), arg.c_str());

    errno = 0; char *pend;
    double val = std::strtod(scalar.c_str(), &pend);

    if(errno || *pend)
      throw GreedyException("Expected a floating point number as parameter to '%s', instead got '%s'",
                            current_command.c_str(), scalar.c_str());

    return val;
  }

  /**
   * Read an integer value
   */
  long read_integer()
  {
    std::string arg = read_arg();

    errno = 0; char *pend;
    long val = std::strtol(arg.c_str(), &pend, 10);

    if(errno || *pend)
      throw GreedyException("Expected an integer as parameter to '%s', instead got '%s'",
                            current_command.c_str(), arg.c_str());

    return val;
  }

  /**
   * Read one of a list of strings. The optional parameters to this are in the form
   * int, string, int, string, int, string. Each string may in turn contain a list
   * of words (separated by space) that are acceptable. So for example. NULL string
   * is used to refer to the default option.
   *
   * enum Mode { NORMAL, BAD, SILLY }
   * Mode m = X.read_option(NORMAL, "NORMAL normal", BAD, "bad BAD", SILLY, NULL);
   */
  /*
  template <class TOption>
  TOption read_option(TOption opt1, const char *str1, ...)
  {
    not implemented yet
  }
  */

  /**
   * Read a vector in the format 1.0x0.2x0.6
   */
  std::vector<double> read_double_vector()
  {
    std::string arg = read_arg();
    std::istringstream f(arg);
    std::string s;
    std::vector<double> vector;
    while (getline(f, s, 'x'))
      {
      errno = 0; char *pend;
      double val = std::strtod(s.c_str(), &pend);

      if(errno || *pend)
        throw GreedyException("Expected a floating point vector as parameter to '%s', instead got '%s'",
                              current_command.c_str(), arg.c_str());
      vector.push_back(val);
      }

    if(!vector.size())
      throw GreedyException("Expected a floating point vector as parameter to '%s', instead got '%s'",
                            current_command.c_str(), arg.c_str());

    return vector;
  }

  std::vector<int> read_int_vector()
  {
    std::string arg = read_arg();
    std::istringstream f(arg);
    std::string s;
    std::vector<int> vector;
    while (getline(f, s, 'x'))
      {
      errno = 0; char *pend;
      long val = std::strtol(s.c_str(), &pend, 10);

      if(errno || *pend)
        throw GreedyException("Expected an integer vector as parameter to '%s', instead got '%s'",
                              current_command.c_str(), arg.c_str());
      vector.push_back((int) val);
      }

    if(!vector.size())
      throw GreedyException("Expected an integer vector as parameter to '%s', instead got '%s'",
                            current_command.c_str(), arg.c_str());

    return vector;
  }





private:
  int argc, i;
  char **argv;
  std::string current_command;
};

int main(int argc, char *argv[])
{
  GreedyParameters param;
  double current_weight = 1.0;

  param.dim = 2;
  param.mode = GreedyParameters::GREEDY;
  param.flag_dump_moving = false;
  param.flag_debug_deriv = false;
  param.flag_debug_aff_obj = false;
  param.dump_frequency = 1;
  param.epsilon = 1.0;
  param.sigma_pre.sigma = sqrt(3.0);
  param.sigma_pre.physical_units = false;
  param.sigma_post.sigma = sqrt(0.5);
  param.sigma_post.physical_units = false;
  param.threads = 0;
  param.metric = GreedyParameters::SSD;
  param.time_step_mode = GreedyParameters::SCALE;
  param.deriv_epsilon = 1e-4;
  param.flag_powell = false;
  param.inverse_exponent = 2;
  param.warp_precision = 0.1;
  param.ncc_noise_factor = 0.001;
  param.affine_init_mode = VOX_IDENTITY;
  param.affine_dof = GreedyParameters::DOF_AFFINE;
  param.affine_jitter = 0.5;
  param.flag_float_math = false;

  // reslice mode parameters
  InterpSpec interp_current;

  param.iter_per_level.push_back(100);
  param.iter_per_level.push_back(100);

  if(argc < 3)
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
        param.epsilon = cl.read_double();
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
      else if(arg == "-a")
        {
        param.mode = GreedyParameters::AFFINE;
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
      else if(arg == "-rm")
        {
        ResliceSpec rp;
        rp.interp = interp_current;
        rp.moving = cl.read_existing_filename();
        rp.output = cl.read_output_filename();
        param.reslice_param.images.push_back(rp);
        }
      else if(arg == "-rf")
        {
        param.reslice_param.ref_image = cl.read_existing_filename();
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

    // Run the main code
    if(param.flag_float_math)
      {
      switch(param.dim)
        {
        case 2: return GreedyApproach<2, float>::Run(param); break;
        case 3: return GreedyApproach<3, float>::Run(param); break;
        case 4: return GreedyApproach<4, float>::Run(param); break;
        default: throw GreedyException("Wrong number of dimensions requested: %d", param.dim);
        }
      }
    else
      {
      switch(param.dim)
        {
        case 2: return GreedyApproach<2, double>::Run(param); break;
        case 3: return GreedyApproach<3, double>::Run(param); break;
        case 4: return GreedyApproach<4, double>::Run(param); break;
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
