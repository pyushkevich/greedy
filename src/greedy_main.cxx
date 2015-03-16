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

#include "MultiImageRegistrationHelper.h"
#include <vnl/vnl_cost_function.h>
#include <vnl/vnl_random.h>

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
  printf("  -brute radius               : Perform a brute force search around each voxel \n");
  printf("  -w weight                   : weight of the next -i pair\n");
  printf("  -m metric                   : metric for the registration (SSD or NCC 3x3x3)");
  printf("  -e epsilon                  : step size (default = 1.0)\n");
  printf("  -tscale MODE                : time step behavior mode: CONST [def], SCALE, SCALEDOWN");
  printf("  -s sigma1 sigma2            : smoothing for the greedy update step (3.0, 1.0)\n");
  printf("  -n NxNxN                    : number of iterations per level of multi-res (100x100) \n");
  printf("  -dump-moving                : dump moving image at each iter\n");
  printf("  -dump-freq N                : dump frequency\n");
  printf("  -threads N                  : set the number of allowed concurrent threads\n");
  printf("  -ia filename                : initial affine transform (c3d format)\n");
  printf("  -gm mask.nii                : mask for gradient computation\n");
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
  enum MetricType { SSD = 0, NCC };
  enum TimeStepMode { CONST=0, SCALE, SCALEDOWN };
  enum Mode { GREEDY=0, AFFINE, BRUTE };



  std::vector<ImagePairSpec> inputs;
  std::string output;
  unsigned int dim; 

  // Registration mode
  Mode mode;

  bool flag_dump_moving;
  int dump_frequency, threads;
  double epsilon, sigma_pre, sigma_post;

  MetricType metric;
  TimeStepMode time_step_mode;

  // Iterations per level (i.e., 40x40x100)
  std::vector<int> iter_per_level;

  std::vector<int> metric_radius;

  std::vector<int> brute_search_radius;

  // Initial affine transform
  std::string initial_affine;

  // Mask for gradient
  std::string gradient_mask;
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

  static int RunBrute(GreedyParameters &param);



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

  static void MapPhysicalRASSpaceToAffine(
      OFHelperType &of_helper, int level,
      vnl_matrix<double> &Qp,
      typename OFHelperType::LinearTransformType *tran);

  /** Cost function used for conjugate gradient descent */
  class AffineCostFunction : public vnl_cost_function
  {
  public:
    typedef typename OFHelperType::LinearTransformType TransformType;


    // Construct the function
    AffineCostFunction(GreedyParameters *param, int level, OFHelperType *helper);

    // Get the parameters for the specified initial transform
    vnl_vector<double> GetCoefficients(TransformType *tran)
    {
      vnl_vector<double> x_true(this->get_number_of_unknowns());
      flatten_affine_transform(tran, x_true.data_block());
      return element_product(x_true, scaling);
    }

    // Get the transform for the specificed coefficients
    void GetTransform(const vnl_vector<double> &coeff, TransformType *tran)
    {
      vnl_vector<double> x_true = element_quotient(coeff, scaling);
      unflatten_affine_transform(x_true.data_block(), tran);
    }

    // Cost function computation
    virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g);

    const vnl_vector<double> &GetScaling() { return scaling; }

  protected:


    // Data needed to compute the cost function
    GreedyParameters *m_Param;
    OFHelperType *m_OFHelper;
    int m_Level;
    vnl_vector<double> scaling;

    // Storage for the gradient of the similarity map
    VectorImagePointer m_Phi, m_GradMetric, m_GradMask;
    ImagePointer m_Metric, m_Mask;
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

  // Set the scaling of the parameters based on image dimensions. This makes it
  // possible to set tolerances in units of voxels. The order of change in the
  // parameters is comparable to the displacement of any point inside the image
  scaling.set_size(this->get_number_of_unknowns());

  typename TransformType::MatrixType matrix;
  typename TransformType::OffsetType offset;
  for(int i = 0; i < VDim; i++)
    {
    offset[i] = 1.0;
    for(int j = 0; j < VDim; j++)
      matrix(i, j) = helper->GetReferenceSpace(level)->GetBufferedRegion().GetSize()[j];
    }

  typename TransformType::Pointer transform = TransformType::New();
  transform->SetMatrix(matrix);
  transform->SetOffset(offset);
  flatten_affine_transform(transform.GetPointer(), scaling.data_block());

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
GreedyApproach<VDim, TReal>::AffineCostFunction
::compute(const vnl_vector<double> &x, double *f, vnl_vector<double> *g)
{
  // Form a matrix/vector from x
  typename TransformType::Pointer tran = TransformType::New();

  // Divide x by the scaling
  vnl_vector<double> x_scaled = element_quotient(x, scaling);

  // Set the components of the transform
  unflatten_affine_transform(x_scaled.data_block(), tran.GetPointer());

  // Compute the gradient
  double val = 0.0;
  if(g)
    {
    vnl_vector<double> g_scaled(x_scaled.size());
    typename TransformType::Pointer grad = TransformType::New();

    if(m_Param->metric == GreedyParameters::SSD)
      {
      val = m_OFHelper->ComputeAffineMSDMatchAndGradient(
              m_Level, tran, m_Metric, m_Mask, m_GradMetric, m_GradMask, m_Phi, grad);

      flatten_affine_transform(grad.GetPointer(), g_scaled.data_block());
      *g = element_quotient(g_scaled, scaling);
      }
    else if(m_Param->metric == GreedyParameters::NCC)
      {

      val = m_OFHelper->ComputeAffineNCCMatchAndGradient(
              m_Level, tran, array_caster<VDim>::to_itkSize(m_Param->metric_radius),
              m_Metric, m_Mask, m_GradMetric, m_GradMask, m_Phi, grad);

      flatten_affine_transform(grad.GetPointer(), g_scaled.data_block());
      *g = element_quotient(g_scaled, scaling);

      // NCC should be maximized
      // *g *= -10000.0;
      // val *= -10000.0;
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
      // val *= -10000.0;
      }
    }

  if(f)
    *f = val;
}


/*
template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>::AffineCostFunction
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
*/

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

  tran_A = A;
  tran_b.SetVnlVector(b);

  tran->SetMatrix(tran_A);
  tran->SetOffset(tran_b);
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
  of_helper.BuildCompositeImages(param.metric == GreedyParameters::NCC);

  // Matrix describing current transform in physical space
  vnl_matrix<double> Q_physical;

  // The number of resolution levels
  int nlevels = param.iter_per_level.size();

  // Iterate over the resolution levels
  for(unsigned int level = 0; level < nlevels; ++level)
    {
    // Define the affine cost function
    AffineCostFunction acf(&param, level, &of_helper);

    // Set up the optimizer
    vnl_lbfgs optimizer(acf);
    optimizer.set_f_tolerance(1e-9);
    optimizer.set_x_tolerance(1e-4);
    optimizer.set_g_tolerance(1e-6);
    optimizer.set_trace(true);

    // Current transform
    typedef typename OFHelperType::LinearTransformType TransformType;
    typename TransformType::Pointer tLevel = TransformType::New();

    // Set up the initial transform
    if(level == 0)
      {
      // Set the initial transform
      tLevel->SetIdentity();

      // Map into the middle of the moving image
      itk::Matrix<double, VDim, VDim> mat;
      mat.SetIdentity();
      mat *= 0.6;
      tLevel->SetMatrix(mat);

      itk::Vector<double, VDim> offset;
      for(int i=0;i<VDim;i++)
        offset[i] = 0.2 * of_helper.GetReferenceSpace(level)->GetBufferedRegion().GetSize()[i];
      tLevel->SetOffset(offset);

      // Apply some random jitter to the initial transform
      vnl_vector<double> xInit = acf.GetCoefficients(tLevel);

      // Apply small amount of jitter to the vector
      vnl_random rndy(12345);
      for(int i = 0; i < xInit.size(); i++)
        xInit[i] += rndy.drand32(-0.4, 0.4);

      // Test the derivative of the cost function
      vnl_vector<double> grad_analytic(VDim * (VDim + 1), 0.0);
      vnl_vector<double> grad_numeric(VDim * (VDim + 1), 0.0);
      double f;
      acf.compute(xInit, &f, &grad_analytic);

      for(int i = 0; i < grad_analytic.size(); i++)
        {
        double eps = 1.0e-5;
        vnl_vector<double> x1 = xInit, x2 = xInit;
        double f1, f2;
        x1[i] += eps; x2[i] -= eps;
        acf.compute(x1, &f1, NULL);
        acf.compute(x2, &f2, NULL);
        grad_numeric[i] = (f1 - f2) / (2 * eps);
        }

      std::cout << "grad-a: " << grad_analytic << std::endl;
      std::cout << "grad-n: " << grad_numeric << std::endl;



      // Map back into transform format
      acf.GetTransform(xInit, tLevel);
      }
    else
      {
      // Update the transform from the last level
      MapPhysicalRASSpaceToAffine(of_helper, level, Q_physical, tLevel);
      }

    // Test derivatives
    // Convert to a parameter vector
    vnl_vector<double> xLevel = acf.GetCoefficients(tLevel.GetPointer());

    // Propagate the jitter to the transform
    Q_physical = MapAffineToPhysicalRASSpace(of_helper, level, tLevel);
    std::cout << "Initial RAS Transform: " << std::endl << Q_physical  << std::endl;

    // Run the minimization
    optimizer.minimize(xLevel);

    // Get the final transform
    typename TransformType::Pointer tFinal = TransformType::New();
    acf.GetTransform(xLevel, tFinal.GetPointer());

    Q_physical = MapAffineToPhysicalRASSpace(of_helper, level, tFinal);
    std::cout << "Final RAS Transform: " << std::endl << Q_physical << std::endl;
    }

  // Write the final affine transform
  std::ofstream matrixFile;
  matrixFile.open(param.output.c_str());
  matrixFile << Q_physical;
  matrixFile.close();


  return 0;
}

#include "itkStatisticsImageFilter.h"

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
  of_helper.BuildCompositeImages(param.metric == GreedyParameters::NCC);

  // An image pointer desribing the current estimate of the deformation
  VectorImagePointer uLevel = NULL;

  // The number of resolution levels
  int nlevels = param.iter_per_level.size();

  std::cout << "SIGMAS: " << param.sigma_pre << ", " << param.sigma_post << std::endl;

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
      LDDMMType::vimg_scale_in_place(uk, 2.0);
      uLevel = uk;
      }
    else if(param.initial_affine.length())
      {
      // Read the initial affine transform from a file
      vnl_matrix<double> Qp(VDim+1, VDim+1);
      std::ifstream fin(param.initial_affine.c_str());
      for(size_t i = 0; i < VDim+1; i++)
        for(size_t j = 0; j < VDim+1; j++)
          if(fin.good())
            {
            fin >> Qp[i][j];
            }
      fin.close();

      // Convert the transform to voxel units
      typename OFHelperType::LinearTransformType::Pointer tran = OFHelperType::LinearTransformType::New();
      MapPhysicalRASSpaceToAffine(of_helper, level, Qp, tran);

      // Create an initial warp
      OFHelperType::AffineToField(tran, uk);
      uLevel = uk;

      itk::Index<VDim> test; test.Fill(24);
      std::cout << "Index 24x24x24 maps to " << uk->GetPixel(test) << std::endl;
      }

    // Iterate for this level
    for(unsigned int iter = 0; iter < param.iter_per_level[level]; iter++)
      {

      // Compute the gradient of objective
      double total_energy;

      if(param.metric == GreedyParameters::SSD)
        {
        vnl_vector<double> all_metrics =
            of_helper.ComputeOpticalFlowField(level, uk, iTemp, uk1, param.epsilon);

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

        total_energy = of_helper.ComputeNCCMetricImage(level, uk, radius, iTemp, uk1, param.epsilon);
        printf("Level %5d,  Iter %5d:    Energy = %8.4f\n", level, iter, total_energy);
        }

      // If there is a mask, multiply the gradient by the mask
      if(param.gradient_mask.size())
        LDDMMType::vimg_multiply_in_place(uk1, of_helper.GetGradientMask(level));

      // Dump the gradient image if requested
      if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
        {
        char fname[256];
        sprintf(fname, "dump_gradient_lev%02d_iter%04d.nii.gz", level, iter);
        LDDMMType::vimg_write(uk1, fname);
        }

      // We have now computed the gradient vector field. Next, we smooth it
      LDDMMType::vimg_smooth(uk1, viTemp, param.sigma_pre * shrink_factor);

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
        // LDDMMType::vimg_write(uk1, fname);
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
      // printf("Level %5d,  Iter %5d:    Energy = %8.4f\n", level, iter, total_energy);
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
  of_helper.VoxelWarpToPhysicalWarp(nlevels - 1, uLevel, uPhys);
  LDDMMType::vimg_write(uPhys, param.output.c_str());

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

  // Generate the optimized composite images
  of_helper.BuildCompositeImages(true);

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



template<class TVector>
void read_cmdl_vector(char *arg, TVector &vector)
{
  std::istringstream f(arg);
  std::string s;
  vector.clear();
  while (getline(f, s, 'x'))
    vector.push_back(atoi(s.c_str()));
}

int main(int argc, char *argv[])
{
  GreedyParameters param;
  double current_weight = 1.0;

  param.dim = 2;
  param.mode = GreedyParameters::GREEDY;
  param.flag_dump_moving = false;
  param.dump_frequency = 1;
  param.epsilon = 1.0;
  param.sigma_pre = 3.0;
  param.sigma_post = 1.0;
  param.threads = 0;
  param.metric = GreedyParameters::SSD;
  param.time_step_mode = GreedyParameters::CONST;

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
      read_cmdl_vector(argv[++i], param.iter_per_level);
      }
    else if(arg == "-w")
      {
      current_weight = atof(argv[++i]);
      }
    else if(arg == "-e")
      {
      param.epsilon = atof(argv[++i]);
      }
    else if(arg == "-m")
      {
      std::string metric_name = argv[++i];
      if(metric_name == "NCC" || metric_name == "ncc")
        {
        param.metric = GreedyParameters::NCC;
        read_cmdl_vector(argv[++i], param.metric_radius);
        }
      }
    else if(arg == "-tscale")
      {
      std::string mode = argv[++i];
      if(mode == "SCALE" || mode == "scale")
        param.time_step_mode = GreedyParameters::SCALE;
      else if(mode == "SCALEDOWN" || mode == "scaledown")
        param.time_step_mode = GreedyParameters::SCALEDOWN;
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
    else if(arg == "-ia")
      {
      param.initial_affine = argv[++i];
      }
    else if(arg == "-gm")
      {
      param.gradient_mask = argv[++i];
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
      param.mode = GreedyParameters::AFFINE;
      }
    else if(arg == "-brute")
      {
      param.mode = GreedyParameters::BRUTE;
      read_cmdl_vector(argv[++i], param.brute_search_radius);
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

  if(param.mode == GreedyParameters::AFFINE)
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
  else if(param.mode == GreedyParameters::GREEDY)
    {
    switch(param.dim)
      {
      case 2: return GreedyApproach<2, float>::Run(param); break;
      case 3: return GreedyApproach<3, double>::Run(param); break;
      case 4: return GreedyApproach<4, float>::Run(param); break;
      default:
            std::cerr << "Wrong dimensionality" << std::endl;
            return -1;
      }
    }
  else if(param.mode == GreedyParameters::BRUTE)
    {
    switch(param.dim)
      {
      case 2: return GreedyApproach<2, float>::RunBrute(param); break;
      case 3: return GreedyApproach<3, double>::RunBrute(param); break;
      case 4: return GreedyApproach<4, float>::RunBrute(param); break;
      default:
            std::cerr << "Wrong dimensionality" << std::endl;
            return -1;
      }
    }
  else
    {
    return -1;
    }
}
