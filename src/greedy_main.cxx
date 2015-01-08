#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

#include "lddmm_common.h"
#include "lddmm_data.h"

#include <itkImageFileReader.h>
#include <itkGaussianInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkIdentityTransform.h>
#include <itkShrinkImageFilter.h>

#include "MultiImageSimpleWarpImageFilter.h"
#include <vnl/vnl_cost_function.h>
#include <vnl/vnl_random.h>



int usage()
{
  printf("greedy: Paul's greedy diffeomorphic registration implementation\n");
  printf("Usage: \n");
  printf("  greedy [options]\n");
  printf("Required options: \n");
  printf("  -d DIM                      : Number of image dimensions\n");
  printf("  -i fixed.nii moving.nii     : Image pair (may be repeated)\n");
  printf("  -o output.nii               : Output file\n");
  printf("Optional: \n");
  printf("  -a                          : Perform affine registration and save to output (-o)\n");
  printf("  -w weight                   : weight of the next -i pair\n");
  printf("  -e epsilon                  : step size (default = 1.0)\n");
  printf("  -s sigma1 sigma2            : smoothing for the greedy update step (3.0, 1.0)\n");
  printf("  -n NxNxN                    : number of iterations per level of multi-res (100x100) \n");
  printf("  -dump-moving                : dump moving image at each iter\n");
  printf("  -dump-freq N                : dump frequency\n");
  printf("  -threads N                  : set the number of allowed concurrent threads\n");
  return -1;
}

struct ImagePairSpec
{
  std::string fixed;
  std::string moving;
  double weight;
};

struct GreedyParameters
{
  std::vector<ImagePairSpec> inputs;
  std::string output;
  unsigned int dim; 

  bool flag_optimize_affine;
  bool flag_dump_moving;
  int dump_frequency, threads;
  double epsilon, sigma_pre, sigma_post;

  // Iterations per level (i.e., 40x40x100)
  std::vector<int> iter_per_level;
};


// Helper function to map from ITK coordiante space to RAS space
template<unsigned int VDim>
void
GetVoxelSpaceToNiftiSpaceTransform(itk::ImageBase<VDim> *image,
                                   vnl_matrix<double> &A,
                                   vnl_vector<double> &b)
{
  // Generate intermediate terms
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

  static int RunAffine(GreedyParameters &param);

protected:

  static void ReadImages(GreedyParameters &param, OFHelperType &ofhelper);
  static void ReadImages(GreedyParameters &param, std::vector<ImagePair> &imgRaw);
  static void ResampleImages(GreedyParameters &param,
                             const std::vector<ImagePair> &imgRaw,
                             std::vector<ImagePair> &img,
                             int level);

  static vnl_matrix<double> MapAffineToPhysicalRASSpace(
      OFHelperType &of_helper, int level,
      typename OFHelperType::LinearTransformType *tran);

  /** Cost function used for conjugate gradient descent */
  class AffineCostFunction : public vnl_cost_function
  {
  public:


    // Construct the function
    AffineCostFunction(GreedyParameters *param, int level, OFHelperType *helper);

    // Cost function computation
    virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g);

  protected:

    typedef typename OFHelperType::LinearTransformType TransformType;

    // Data needed to compute the cost function
    GreedyParameters *m_Param;
    OFHelperType *m_OFHelper;
    int m_Level;

    // Storage for the gradient of the similarity map
    VectorImagePointer *m_GradSim;
  };
};

template <unsigned int VDim, typename TReal>
GreedyApproach<VDim, TReal>::AffineCostFunction
::AffineCostFunction(GreedyParameters *param, int level, OFHelperType *helper)
  : vnl_cost_function(VDim * (VDim + 1))
{
  // Store the data
  m_Param = param;
  m_OFHelper = helper;
  m_Level = level;

  // Initialize the image data
  // m_GradSim = VectorImageType::New();
  // LDDMMType::alloc_vimg(m_GradSim, helper->GetReferenceSpace(level));
}

template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>::AffineCostFunction
::compute(const vnl_vector<double> &x, double *f, vnl_vector<double> *g)
{
  // Form a matrix/vector from x
  typename TransformType::Pointer tran = TransformType::New();

  // Set the components of the transform
  itk::unflatten_affine_transform(x.data_block(), tran.GetPointer());

  // Compute the gradient
  double val = 0.0;
  if(g)
    {
    typename TransformType::Pointer grad = TransformType::New();
    val = m_OFHelper->ComputeAffineMatchAndGradient(m_Level, tran, grad);
    itk::flatten_affine_transform(grad.GetPointer(), g->data_block());
    }
  else
    {
    val = m_OFHelper->ComputeAffineMatchAndGradient(m_Level, tran, NULL);
    }

  if(f)
    *f = val;
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::ReadImages(GreedyParameters &param, OFHelperType &ofhelper)
{
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

    // Add to the helper object
    ofhelper.AddImagePair(readfix->GetOutput(), readmov->GetOutput(), param.inputs[i].weight);
    }
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
  A = tran->GetMatrix().GetVnlMatrix();
  b = tran->GetOffset().GetVnlVector();

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
int GreedyApproach<VDim, TReal>
::RunAffine(GreedyParameters &param)
{
  // Create an optical flow helper object
  OFHelperType of_helper;

  // Set the scaling factors for multi-resolution
  of_helper.SetDefaultPyramidFactors(param.iter_per_level.size());

  // Read the image pairs to register
  ReadImages(param, of_helper);

  // Generate the optimized composite images
  of_helper.BuildCompositeImages();

  // The number of resolution levels
  int nlevels = param.iter_per_level.size();

  // Iterate over the resolution levels
  for(unsigned int level = 0; level < nlevels; ++level)
    {
    // Reference space
    ImageBaseType *refspace = of_helper.GetReferenceSpace(level);

    // Define the affine cost function
    AffineCostFunction acf(&param, level, &of_helper);

    // Perform the optimization
    vnl_lbfgs optimizer(acf);
    optimizer.set_f_tolerance(1e-4);
    optimizer.set_x_tolerance(1e-3);
    optimizer.set_g_tolerance(1e-2);
    optimizer.set_trace(true);
    //  optimizer.set_check_derivatives(1);

    // Set the initial parameter vector
    typedef typename OFHelperType::LinearTransformType TransformType;
    typename TransformType::Pointer tInit = TransformType::New();
    tInit->SetIdentity();

    typename TransformType::OffsetType offset = tInit->GetOffset();
    typename TransformType::MatrixType matrix = tInit->GetMatrix();
    vnl_random rndy;
    for(int i = 0; i < VDim; i++)
      {
      offset[i] += rndy.drand32(-4.0, 4.0);
      for(int j = 0; j < VDim; j++)
        matrix(i,j) += rndy.drand32(-0.04, 0.04);
      }
    tInit->SetOffset(offset);
    tInit->SetMatrix(matrix);

    vnl_vector<double> xInit(acf.get_number_of_unknowns(), 0.0);
    itk::flatten_affine_transform(tInit.GetPointer(), xInit.data_block());

    // Test the function
    vnl_vector<double> xGrad(acf.get_number_of_unknowns(), 0.0);
    double f0;
    acf.compute(xInit, &f0, &xGrad);
    std::cout << "Analytic gradient: " << xGrad << std::endl;

    vnl_vector<double> xGradN(acf.get_number_of_unknowns(), 0.0);
    for(int i = 0; i < acf.get_number_of_unknowns(); i++)
      {
      double eps = 1.0e-4, f1, f2;
      vnl_vector<double> x1 = xInit, x2 = xInit;
      x1[i] -= eps; x2[i] += eps;

      acf.compute(x1, &f1, NULL);
      acf.compute(x2, &f2, NULL);

      xGradN[i] = (f2 - f1) / (2 * eps);
      }

    std::cout << "Numeric gradient: " << xGradN << std::endl;

    std::cout << "f = " << f0 << std::endl;

    vnl_matrix<double> Qi = MapAffineToPhysicalRASSpace(of_helper, level, tInit);
    std::cout << "Initial RAS Transform: " << std::endl << Qi  << std::endl;


    optimizer.minimize(xInit);

    // Get the final transform
    typename TransformType::Pointer tFinal = TransformType::New();
    itk::unflatten_affine_transform(xInit.data_block(), tFinal.GetPointer());

    vnl_matrix<double> Qf = MapAffineToPhysicalRASSpace(of_helper, level, tFinal);
    std::cout << "Final RAS Transform: " << std::endl << Qf << std::endl;
    }

  return 0;
}

/**
 * This is the main function of the GreedyApproach algorithm
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::Run(GreedyParameters &param)
{
  // Create an optical flow helper object
  OFHelperType of_helper;

  // Set the scaling factors for multi-resolution
  of_helper.SetDefaultPyramidFactors(param.iter_per_level.size());

  // Read the image pairs to register
  ReadImages(param, of_helper);

  // Generate the optimized composite images
  of_helper.BuildCompositeImages();

  // An image pointer desribing the current estimate of the deformation
  VectorImagePointer uLevel = NULL;

  // The number of resolution levels
  int nlevels = param.iter_per_level.size();

  // Iterate over the resolution levels
  for(unsigned int level = 0; level < nlevels; ++level)
    {
    // The scaling factor
    int shrink_factor = 1 << (nlevels - (1 + level));

    // Reference space
    ImageBaseType *refspace = of_helper.GetReferenceSpace(level);

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
      }

    // Iterate for this level
    for(unsigned int iter = 0; iter < param.iter_per_level[level]; iter++)
      {

      // Compute the gradient of objective
      double total_energy = of_helper.ComputeOpticalFlowField(level, uk, uk1, param.epsilon);

      // Dump the gradient image if requested
      if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
        {
        char fname[256];
        sprintf(fname, "dump_gradient_lev%02d_iter%04d.nii.gz", level, iter);
        LDDMMType::vimg_write(uk1, fname);
        }

      // We have now computed the gradient vector field. Next, we smooth it
      LDDMMType::vimg_smooth(uk1, viTemp, param.sigma_pre * shrink_factor);

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

      // Swap uk and uk1 pointers
      // VectorImagePointer tmpptr = uk1; uk1 = uk; uk = tmpptr;

      // Another layer of smoothing - really?
      LDDMMType::vimg_smooth(uk1, uk, param.sigma_post * shrink_factor);

      // Compute the Jacobian determinant of the updated field (temporary)
          /*
        */

      // Report the energy
      // printf("Iter %5d:    Energy = %8.4f     DetJac Range: %8.4f  to %8.4f \n", iter, total_energy, jac_min, jac_max);
      printf("Level %5d,  Iter %5d:    Energy = %8.4f\n", level, iter, total_energy);
      }

    // Store the end result
    uLevel = uk;

    // Compute the jacobian of the deformation field
    LDDMMType::field_jacobian_det(uk, iTemp);
    TReal jac_min, jac_max;
    LDDMMType::img_min_max(iTemp, jac_min, jac_max);
    printf("END OF LEVEL %5d    DetJac Range: %8.4f  to %8.4f \n", level, jac_min, jac_max);

    }

  // The transformation field is in voxel units. To work with ANTS, it must be mapped
  // into physical offset units - just scaled by the spacing?

  // Write the resulting transformation field
  VectorImagePointer uPhys = VectorImageType::New();
  LDDMMType::alloc_vimg(uPhys, uLevel);
  LDDMMType::warp_voxel_to_physical(uLevel, uPhys);
  LDDMMType::vimg_write(uPhys, param.output.c_str());

  return 0;
}




int main(int argc, char *argv[])
{
  GreedyParameters param;
  double current_weight = 1.0;

  param.dim = 2;
  param.flag_dump_moving = false;
  param.flag_optimize_affine = false;
  param.dump_frequency = 1;
  param.epsilon = 1.0;
  param.sigma_pre = 3.0;
  param.sigma_post = 1.0;
  param.threads = 0;

  param.iter_per_level.push_back(100);
  param.iter_per_level.push_back(100);

  if(argc < 3)
    return usage();

  for(int i = 1; i < argc; ++i)
    {
    std::string arg = argv[i];
    if(arg == "-d")
      {
      param.dim = atoi(argv[++i]);
      }
    else if(arg == "-n")
      {
      std::istringstream f(argv[++i]);
      std::string s;
      param.iter_per_level.clear();
      while (getline(f, s, 'x'))
        param.iter_per_level.push_back(atoi(s.c_str()));
      }
    else if(arg == "-w")
      {
      current_weight = atof(argv[++i]);
      }
    else if(arg == "-e")
      {
      param.epsilon = atof(argv[++i]);
      }
    else if(arg == "-s")
      {
      param.sigma_pre = atof(argv[++i]);
      param.sigma_post = atof(argv[++i]);
      }
    else if(arg == "-i")
      {
      ImagePairSpec ip;
      ip.weight = current_weight;
      ip.fixed = argv[++i];
      ip.moving = argv[++i];
      param.inputs.push_back(ip);
      }
    else if(arg == "-o")
      {
      param.output = argv[++i];
      }
    else if(arg == "-dump-moving")
      {
      param.flag_dump_moving = true;
      }
    else if(arg == "-dump-frequency" || arg == "-dump-freq")
      {
      param.dump_frequency = atoi(argv[++i]);
      }
    else if(arg == "-threads")
      {
      param.threads = atoi(argv[++i]);
      }
    else if(arg == "-a")
      {
      param.flag_optimize_affine = true;
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

  if(param.flag_optimize_affine)
    {
    switch(param.dim)
      {
      case 2: return GreedyApproach<2>::RunAffine(param); break;
      case 3: return GreedyApproach<3>::RunAffine(param); break;
      case 4: return GreedyApproach<4>::RunAffine(param); break;
      default:
            std::cerr << "Wrong dimensionality" << std::endl;
            return -1;
      }
    }
  else
    {
    switch(param.dim)
      {
      case 2: return GreedyApproach<2>::Run(param); break;
      case 3: return GreedyApproach<3>::Run(param); break;
      case 4: return GreedyApproach<4>::Run(param); break;
      default:
            std::cerr << "Wrong dimensionality" << std::endl;
            return -1;
      }
    }
}
