#include <GreedyAPI.h>
#include <CommandLineHelper.h>
#include <itksys/SystemTools.hxx>
#include <itkMatrixOffsetTransformBase.h>
#include <GreedyException.h>
#include <lddmm_data.h>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>
#include <MultiImageRegistrationHelper.h>
#include <OneDimensionalInPlaceAccumulateFilter.h>
#include <FastLinearInterpolator.h>
#include <MultiComponentWeightedNCCImageMetric.h>
#include <itkMultiThreaderBase.h>
#include "TetraMeshConstraints.h"
#include "DifferentiableScalingAndSquaring.h"
#include "GreedyMeshIO.h"
#include "vtkUnstructuredGrid.h"
#include "itkTimeProbe.h"

// Global variable storing the test data root
std::string data_root;

int usage()
{
  printf("test_greedy: GreedyReg test executable\n");
  printf("usage: \n");
  printf("  test_greedy <command> [options] \n");
  printf("commands: \n");
  printf("  phantom <1|2|3> <1|2|3> <NCC|NMI|SSD> <6|12> <0|1> \n");
  printf("        : run block phantom tests with selected fixed and moving phantoms\n");
  printf("          metric, affine degrees of freedom and masking. \n");
  printf("  masked_interpolation_test <2|3>\n");
  printf("        : test FastLinearInterpolator gradients in 2D or 3D. \n");
  printf("  ncc_gradient_vs_matlab <0|1>\n");
  printf("        : test new (2021) NCC metric gradients vs. MATLAB code. \n");
  printf("          0 for unweighted, 1 for weighted\n");
  printf("  grad_metric_phi <2|3> <eps> <tol> <greedy_opts>\n");
  printf("        : check gradients of various metrics with respect to phi\n");
  printf("          Greedy options: -i, -it, -id, -m, -gm, -mm \n");
  printf("  grad_metric_aff <2|3> <eps> <tol> <greedy_opts>\n");
  printf("        : check gradients of various metrics with respect to affine transforms\n");
  printf("          Greedy options: -i, -ia, -m, -gm, -mm \n");
  printf("  reg_2d_3d <aff|def> <metric_value> <tol> <greedy_opts>\n");
  printf("        : Test 2D/3D registration using phantom images\n");
  printf("          Greedy options: -i, -ia, -m, -n \n");
  printf("  tet_jac_reg <2|3> [refimage] [mesh] \n");
  printf("        : Test derivatives of the tetrahedral jacobian regularization term\n");
  printf("  comp_layer <2|3> \n");
  printf("        : Test derivatives of the warp composition layer\n");
  printf("  ssq_layer <2|3> [noise_amplitude==8.0] [noise_sigma=1.0] \n");
  printf("        : Test derivatives of the scaling and squaring layer\n");
  printf("  svf_smoothness_reg <2|3>\n");
  printf("        : Test derivatives of SVF smoothness regularizer\n");
  printf("  fast_smoothing <2|3> <fn_src> <fn_target> <sigma>\n");
  printf("        : Test fast smoothing code\n");
  return -1;
}

std::string GetFileName(const char *pattern, ...)
{
  // Fill out the filename
  va_list args;
  va_start(args, pattern);
  char filename_local[2048];
  vsprintf(filename_local, pattern, args);
  va_end(args);

  // Prepend the root
  return itksys::SystemTools::CollapseFullPath(filename_local, data_root);
}

/**
 * Test FastLinearInterpolator to make sure it returns correct gradients in
 * masked regions.
 */
template <unsigned int VDim = 3>
int RunMaskedInterpolationTest()
{
  typedef LDDMMData<double, VDim> LDDMMType;

  // Create a 40x40 reference image
  itk::Size<VDim> size;
  for(unsigned int d = 0; d < VDim; d++)
    size[d] = 40;
  typename LDDMMType::ImagePointer ref_space = LDDMMType::ImageType::New();
  ref_space->SetRegions(typename LDDMMType::RegionType(size));

  // Create an image to interpolate
  typename LDDMMType::CompositeImagePointer M = LDDMMType::new_cimg(ref_space, 1);

  // Set some random values at scattered locations
  vnl_random randy;
  for(unsigned int k = 0; k < M->GetPixelContainer()->Size() / 10; k++)
    {
    itk::Index<VDim> pos;
    for(unsigned int d = 0; d < VDim; d++)
      pos[d] = randy.lrand32(0, size[d]-1);

    typename LDDMMType::CompositeImageType::InternalPixelType val = randy.drand32(0, 256);
    typename LDDMMType::CompositeImageType::PixelType pixel(&val, 1);
    M->SetPixel(pos, pixel);
    }

  // Smooth the image
  typename LDDMMType::Vec sigma; sigma.Fill(2.0);
  LDDMMType::cimg_smooth(M, M, sigma);

  // Create a random mask
  typename LDDMMType::ImagePointer W = LDDMMType::new_img(ref_space);
  for(unsigned int k = 0; k < W->GetPixelContainer()->Size() / 10; k++)
    {
    itk::Index<VDim> pos;
    for(unsigned int d = 0; d < VDim; d++)
      pos[d] = randy.lrand32(0, size[d]-1);

    typename LDDMMType::ImageType::PixelType pixel = 1;
    W->SetPixel(pos, pixel);
    }

  // Dilate to create actual mask (smooth and threshold works fine)
  LDDMMType::img_smooth(W, W, typename LDDMMType::Vec(3.0));
  LDDMMType::img_threshold_in_place(W, 0.1, 1.1, 1.0, 0.0);

  // Multiply the moving image by the mask, so that we consistently return W*M for
  // outside pixels and zero-mask pixels
  itk::ImageRegionIteratorWithIndex<typename LDDMMType::CompositeImageType> it_M(M, M->GetBufferedRegion());
  itk::ImageRegionIteratorWithIndex<typename LDDMMType::ImageType> it_W(W, W->GetBufferedRegion());
  while(!it_M.IsAtEnd())
    {
    it_M.Set(it_M.Get() * it_W.Get());
    ++it_M; ++it_W;
    }

  // Create a fast interpolator
  typedef FastLinearInterpolator<
      typename LDDMMType::CompositeImageType, double, VDim,
      typename LDDMMType::ImageType> InterpType;

  // Create interpolator
  InterpType interp(M, W);

  // Status
  int retval = 0;

  // Sample at various locations inside and outside of the image
  int n_inside = 0, n_border = 0, n_outside = 0;
  for(unsigned int s = 0; s < 1000; s++)
    {
    // Random location
    vnl_vector<double> cix(VDim);
    for(unsigned int d = 0; d < VDim; d++)
      cix[d] = randy.drand32(-4.0, size[d] + 4.0);

    // Interpolate moving image and gradient
    vnl_vector<typename InterpType::OutputComponentType> M_grad(VDim, 0.0);
    typename InterpType::OutputComponentType M_sample, *M_grad_ptr = M_grad.data_block();
    auto status = interp.InterpolateWithGradient(cix.data_block(), &M_sample, &M_grad_ptr);

    // Ingore outside voxels, gradient is messed up there
    if(status == InterpType::OUTSIDE)
      {
      n_outside++;
      // continue;
      }
    else if(status == InterpType::BORDER)
      {
      n_border++;
      }
    else
      {
      n_inside++;
      }

    // Get mask value and gradient
    vnl_vector<double> W_grad(VDim, 0.0);
    double W_sample = interp.GetMaskAndGradient(W_grad.data_block());

    // Compute numerical gradient of both
    double eps = 1.0e-5, tol = 1.0e-3;
    double err_m = 0.0, err_w = 0.0;

    // Numeric gradients
    vnl_vector<double> M_num(VDim, 0.0), W_num(VDim, 0.0);
    for(unsigned int d = 0; d < VDim; d++)
      {
      vnl_vector<double> cix1 = cix; cix1[d] -= eps;
      vnl_vector<double> cix2 = cix; cix2[d] += eps;
      typename InterpType::OutputComponentType m1 = 0.0, m2 = 0.0;
      double w1 = 0.0, w2 = 0.0;

      interp.Interpolate(cix1.data_block(), &m1);
      w1 = interp.GetMask();

      interp.Interpolate(cix2.data_block(), &m2);
      w2 = interp.GetMask();

      M_num[d] = (m2 - m1) / (2*eps);
      W_num[d] = (w2 - w1) / (2*eps);

      err_m += fabs(M_grad[d] - M_num[d]);
      err_w += fabs(W_grad[d] - W_num[d]);
      }

    if(err_m > tol || err_w > tol)
      {
      std::cerr << "Derivative error at sample " << s << " index " << cix
                << " M = " << M_sample << " W = " << W_sample
                << " err_M = " << err_m
                << " err_W = " << err_w
                << std::endl;
      std::cerr << "  Grad_M  An: " << M_grad << "  Nu: " << M_num << std::endl;
      std::cerr << "  Grad_W  An: " << W_grad << "  Nu: " << W_num << std::endl;
      retval = -1;
      }
    }

  if(!retval)
    std::cout
        << "Success ("
        << "inside: " << n_inside
        << "; border: " << n_border
        << "; outside: " << n_outside << ")"
        << std::endl;

  return retval ;
}

int RunPhantomTest(CommandLineHelper &cl)
{
  // Set up greedy parameters for this test
  GreedyParameters gp;
  gp.dim = 3;
  gp.mode = GreedyParameters::AFFINE;

  // Which phantom to use
  int phantom_fixed_idx = cl.read_integer();
  int phantom_moving_idx = cl.read_integer();

  // Read the metric - this determines which image pair to use
  std::string metric = cl.read_string();
  int dof = cl.read_integer();
  int use_mask = cl.read_integer();

  // Configure the degrees of freedom
  if(metric == "NCC")
    {
    gp.metric = GreedyParameters::NCC;
    gp.metric_radius = std::vector<int>(3, 2);
    }
  else if(metric == "WNCC")
    {
    gp.metric = GreedyParameters::WNCC;
    gp.metric_radius = std::vector<int>(3, 2);
    }
  else if(metric == "SSD")
    gp.metric = GreedyParameters::SSD;
  else if(metric == "NMI")
    gp.metric = GreedyParameters::NMI;

  // Set up the input filenames
  std::string fn_fix = GetFileName("phantom%02d_fixed.nii.gz", phantom_fixed_idx);
  std::string fn_mov = GetFileName("phantom%02d_moving.nii.gz", phantom_moving_idx);
  std::string fn_mask = GetFileName("phantom01_mask.nii.gz");

  gp.input_groups.back().inputs.push_back(ImagePairSpec(fn_fix, fn_mov));
  if(use_mask)
    gp.input_groups.back().fixed_mask = fn_mask;

  gp.affine_dof = dof == 6 ? GreedyParameters::DOF_RIGID : (
                               dof == 7 ? GreedyParameters::DOF_SIMILARITY :
                                          GreedyParameters::DOF_AFFINE);

  // Set number of steps
  gp.iter_per_level = {{100, 60, 20}};

  // Store transform somewhere
  gp.output = "my_transform";

  // Run the affine registration
  typedef GreedyApproach<3> GreedyAPI;
  typedef GreedyAPI::LinearTransformType TransformType;
  GreedyAPI api_reg;

  // Output transform
  TransformType::Pointer tran = TransformType::New();
  api_reg.AddCachedOutputObject("my_transform", tran.GetPointer());

  // Report parameters
  std::cout << "Running affine registration with parameters " << gp.GenerateCommandLine() << std::endl;

  // Run the optimization
  api_reg.Run(gp);

  std::cout << "Affine registration complete" << std::endl;

  // Phantom source and ground truth transform
  std::string fn_source = GetFileName("phantom01_source.nii.gz");
  std::string fn_true_rigid = GetFileName("phantom01_rigid.mat");

  // Load the original image
  typedef GreedyAPI::LDDMMType LDDMMType;
  LDDMMType::CompositeImagePointer img_source = LDDMMType::cimg_read(fn_source.c_str());

  // Set reslicing parameters
  GreedyParameters gp_res;
  gp_res.dim = 3;
  gp_res.mode = GreedyParameters::RESLICE;

  gp_res.reslice_param.ref_image = fn_fix;
  gp_res.reslice_param.images.push_back(
        ResliceSpec("my_source", "my_resliced", InterpSpec(InterpSpec::LABELWISE, 0.1)));

  gp_res.reslice_param.transforms.push_back(TransformSpec("my_transform"));
  gp_res.reslice_param.transforms.push_back(TransformSpec(fn_true_rigid));

  // Allocate the result image
  typedef GreedyAPI::CompositeImageType CImageType;
  typedef itk::Image<short,3> LabelImageType;
  LabelImageType::Pointer img_reslice = LabelImageType::New();

  GreedyAPI api_reslice;
  api_reslice.AddCachedInputObject("my_source", img_source.GetPointer());
  api_reslice.AddCachedInputObject("my_transform", tran.GetPointer());
  api_reslice.AddCachedOutputObject("my_resliced", img_reslice.GetPointer());

  // Report parameters
  std::cout << "Running reslicing with parameters " << gp_res.GenerateCommandLine() << std::endl;

  api_reslice.Run(gp_res);
  std::cout << "Reslicing complete" << std::endl;

  typedef itk::VectorIndexSelectionCastImageFilter<CImageType, LabelImageType> CastFilter;
  CastFilter::Pointer fltCastSource = CastFilter::New();
  fltCastSource->SetInput(img_source);
  fltCastSource->SetIndex(0);
  fltCastSource->Update();
  std::cout << fltCastSource->GetOutput()->GetBufferedRegion() << std::endl;
  std::cout << img_reslice->GetBufferedRegion() << std::endl;

  // Compute the generalized dice overlap
  typedef itk::LabelOverlapMeasuresImageFilter<LabelImageType> OverlapFilter;
  OverlapFilter::Pointer fltOverlap = OverlapFilter::New();
  fltOverlap->SetSourceImage(fltCastSource->GetOutput());
  fltOverlap->SetTargetImage(img_reslice);
  fltOverlap->UpdateLargestPossibleRegion();

  // Get the Dice overlap
  double gen_dice = fltOverlap->GetDiceCoefficient();
  std::cout << "Generalized Dice after registration: " << gen_dice << std::endl;
  if(gen_dice < 0.92)
    {
    std::cout << "Test failed, insufficient Dice" << std::endl;
    return -1;
    }
  else return 0;
}

template <unsigned int VDim>
std::string printf_index(const char *format, itk::Index<VDim> index)
{
  std::string result;
  for(unsigned int i = 0; i < VDim; i++)
    {
    char buf[256];
    sprintf(buf, format, index[i]);
    result += buf;
    if(i < VDim - 1)
      result += ",";
    }
  return result;
}

template <unsigned int VDim, typename T>
std::string printf_vec(const char *format, T *arr)
{
  std::string result;
  for(unsigned int i = 0; i < VDim; i++)
    {
    char buf[256];
    sprintf(buf, format, arr[i]);
    result += buf;
    if(i < VDim - 1)
      result += ",";
    }
  return result;
}

int BasicWeightedNCCGradientTest(bool weighted)
{
  itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(1);
  itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1);

  // Fixed and moving values from MATLAB
  double F[] = {
    2.113031e+00, -3.101215e-01, -3.471530e-01, -1.320801e+00, 2.162966e-01,
    5.459689e-01, 1.370024e+00, 8.066662e-01, 6.574276e-01, 2.767253e-01,
    1.226758e+00, 4.277553e-01, -1.156259e+00, -1.076492e+00, -8.767345e-01,
    5.533616e-01, 1.579562e+00, 1.219700e+00, 2.123254e+00, 2.615404e-01,
    6.205432e-01, 1.165775e+00, -8.470456e-02, -6.608801e-01, -1.019336e-01,
    1.038028e+00, 1.210400e+00, 7.068342e-01, 9.945488e-01, -8.603150e-01,
    -2.029896e+00, 1.148114e+00, 1.795308e+00, -2.387594e+00, -3.709374e-02,
    1.338605e-01, 5.942942e-01, -1.633092e+00, 1.729625e+00, 3.522852e-01,
    6.835319e-01, 5.943463e-04, 1.190853e+00, -1.224254e+00, -7.837978e-01,
    -8.645776e-01, -1.310285e-01, -1.331426e-01, 7.973239e-01, -1.654186e+00};

  double M[] = {
    2.916250e-02, -1.509852e+00, 7.208623e-01, 1.019153e+00, 6.266861e-02,
    -1.003633e+00, -4.145445e-02, -3.256169e-01, 1.065722e+00, 3.084227e-01,
    -2.546284e-01, -4.698158e-01, 2.934224e-01, -1.706190e+00, -5.690751e-01,
    7.496825e-01, 1.847690e+00, 2.835588e-01, 2.671740e+00, 7.121727e-01,
    1.072848e+00, -1.155038e-02, -6.153552e-01, 5.367157e-01, 5.604590e-03,
    -1.378147e+00, -1.083360e+00, 1.144130e+00, -1.391504e-02, 1.689501e+00,
    1.630319e+00, 8.642794e-02, -6.570671e-01, 8.015981e-01, -1.635533e-01,
    -1.893970e-03, -4.032401e-01, -6.377865e-01, 1.951104e-01, -3.930638e-01,
    4.793461e-01, 1.058682e+00, 5.882104e-01, 9.996348e-01, -1.126601e+00,
    6.517136e-01, -3.994088e-01, 8.355329e-01, 1.635483e+00, -6.955057e-01};

  double W[] = {
    1.957950e-01, 2.945013e-01, 6.269999e-01, 8.622311e-02, 1.429450e-01,
    5.158265e-01, 6.893413e-01, 8.566258e-01, 6.473617e-01, 5.816187e-01,
    7.111160e-01, 2.524169e-01, 9.001597e-01, 4.422937e-01, 2.052082e-02,
    9.596610e-01, 6.522254e-01, 5.132063e-01, 6.823564e-01, 4.895404e-01,
    9.264902e-01, 5.158798e-01, 7.215988e-02, 5.675083e-01, 6.152432e-01,
    9.415463e-01, 4.153634e-01, 2.644400e-01, 9.739317e-02, 4.858442e-01,
    4.646629e-01, 2.975932e-02, 6.942775e-01, 7.169471e-01, 7.298114e-01,
    4.143510e-01, 1.509884e-02, 9.089752e-01, 7.893787e-01, 1.651992e-01,
    3.127860e-01, 6.109453e-01, 3.644903e-01, 1.560386e-01, 1.773038e-01,
    8.678897e-01, 2.900947e-01, 5.851796e-01, 4.539949e-01, 4.111781e-01
  };

  // Metric computation mask (-gm equivalent)
  double K[] = {
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0};

  // Expected NCC and grad
  double expected_NCC_unw[] = {
    2.191734e-02, -4.686602e-02, -4.689990e-02, -3.523790e-01, -5.125640e-01,
    -5.055979e-01, 2.746831e-04, -4.532824e-03, -2.465061e-01, -5.062250e-02,
    -2.544271e-02, 7.996050e-02, 4.980022e-02, 2.196680e-01, 6.996894e-01,
    8.121999e-01, 7.931090e-01, 6.271049e-01, 5.748011e-01, 3.513032e-01,
    6.314152e-01, 2.657034e-03, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, -7.536540e-01, -7.952989e-01, -9.636332e-01,
    -2.401571e-01, 6.672283e-01, 6.234391e-01, 4.795613e-01, 1.788605e-01,
    -4.141935e-02, -1.729231e-01, 4.338921e-02, 2.322679e-02, 1.020001e-03,
    0, 0, 0, 0, 0};

  double expected_Gradient_unw[] = {
    -7.931726e-01, -8.890974e-01, 5.681757e-03, 9.513810e-01, -2.459646e-01,
    -4.508113e-01, -4.315916e-01, -1.255892e-01, -2.088492e-01, 4.116573e-01,
    -2.114728e-01, 2.278992e-01, 2.346085e+00, -9.872185e-03, -7.339386e-01,
    -4.921512e-02, -2.819553e-01, 3.258282e+00, -6.645711e-01, -3.861992e-01,
    5.766248e-01, -3.170130e-01, -7.935631e-02, 2.782447e-02, 0,
    0, 0, 0, 0, 0,
    -2.459917e-01, -6.717985e-01, 1.940539e-01, 7.625819e-01, -1.291876e-01,
    3.724049e-01, -5.650816e-01, -2.302952e+00, -9.892850e-01, 4.828562e-01,
    -1.186675e-01, 1.636822e-01, 3.388455e-01, 1.495978e+00, -1.261360e-01,
    6.263267e-02, 5.878134e-03, 0, 0, 0};

  double expected_NCC_wgt[] = {
    -3.387199e-06, -7.008272e-04, -8.822945e-04, -3.494058e-02, -5.015853e-02,
    -3.836841e-03, 1.449632e-04, -4.290233e-03, -1.121110e-01, -5.188574e-02,
    -9.400475e-03, 3.520195e-03, 1.874769e-04, 6.177238e-02, 2.203093e-01,
    2.208611e-01, 2.288297e-01, 2.995991e-01, 2.633902e-01, 1.546403e-01,
    1.571882e-01, -2.455656e-03, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, -2.253506e-01, -2.575235e-01, -2.574806e-01,
    -4.077543e-02, 3.064733e-01, 1.896074e-01, 1.492842e-01, 8.882288e-02,
    -7.555127e-02, -1.883038e-02, 2.868920e-03, 4.256522e-03, -8.818908e-05,
    0, 0, 0, 0, 0};

  double expected_Gradient_wgt[] = {
    -2.528125e-02, -1.329148e-01, 8.359985e-02, 4.259147e-02, -1.541739e-02,
    -7.276558e-02, -1.552503e-01, -1.459809e-01, -8.088831e-02, 2.298587e-01,
    7.944193e-02, -1.273084e-02, 3.623903e-01, -2.268911e-01, -4.689008e-02,
    -3.271123e-01, -2.504951e-01, 1.671103e+00, -4.946397e-01, 2.022376e-02,
    3.372227e-02, -2.516246e-03, 8.430814e-02, 2.404325e-02, 0,
    0, 0, 0, 0, 0,
    5.992563e-02, -4.320686e-01, 1.330648e-01, 1.425765e-01, 1.107269e-01,
    2.091409e-02, -3.343489e-01, -2.935259e-01, -3.762383e-01, 8.257934e-02,
    -8.143800e-02, 9.420128e-02, 4.291498e-02, 1.379273e-01, 1.024425e-02,
    1.082144e-02, 7.409055e-04, 0, 0, 0};

  // Generate images from this data
  typedef LDDMMData<double, 2> LDDMMType;

  // Create a reference space
  typedef itk::Image<unsigned char, 2> RefImageType;
  RefImageType::Pointer ref = RefImageType::New();
  RefImageType::RegionType region;
  region.SetSize(0, 50);
  region.SetSize(1, 1);
  ref->SetRegions(region);

  // Set metric radius
  itk::Size<2> radius = {{2,0}};

  // Load test data
  LDDMMType::CompositeImagePointer fix = LDDMMType::new_cimg(ref.GetPointer(), 1);
  LDDMMType::CompositeImagePointer mov = LDDMMType::new_cimg(ref.GetPointer(), 1);
  LDDMMType::ImagePointer wgt = LDDMMType::new_img(ref.GetPointer());
  LDDMMType::ImagePointer ncc_mask = LDDMMType::new_img(ref.GetPointer());
  for(unsigned int i = 0; i < 50; i++)
    {
    if(weighted)
      {
      // The filter expects the iterator to return (m*w, w) pairs, so to match
      // matlab we multiply m by w here
      wgt->GetBufferPointer()[i] = W[i];
      mov->GetBufferPointer()[i] = M[i] * W[i];
      }
    else
      {
      mov->GetBufferPointer()[i] = M[i];
      }
    fix->GetBufferPointer()[i] = F[i];
    ncc_mask->GetBufferPointer()[i] = K[i];
    }

  // Process the ncc mask like in MultiImageOpticalFlowHelper
  LDDMMType::img_threshold_in_place(ncc_mask, 0.5, 1e100, 0.5, 0);
  LDDMMType::ImagePointer mask_copy = LDDMMType::new_img(ref.GetPointer());
  LDDMMType::img_copy(ncc_mask, mask_copy);
  LDDMMType::ImagePointer mask_accum =
      AccumulateNeighborhoodSumsInPlace(mask_copy.GetPointer(), radius);
  LDDMMType::img_threshold_in_place(mask_accum, 0.25, 1e100, 0.5, 0);
  LDDMMType::img_add_in_place(ncc_mask, mask_accum);

  // Create zero deformation
  LDDMMType::VectorImagePointer phi = LDDMMType::new_vimg(ref.GetPointer());

  // Create outputs
  LDDMMType::ImagePointer nccmap = LDDMMType::new_img(ref.GetPointer());
  LDDMMType::VectorImagePointer grad = LDDMMType::new_vimg(ref.GetPointer());

  // Create the filter
  typedef DefaultMultiComponentImageMetricTraits<double, 2> TraitsType;
  typedef MultiComponentWeightedNCCImageMetric<TraitsType> MetricType;

  // Create the working image, filter will allocate
  LDDMMType::CompositeImagePointer work = LDDMMType::CompositeImageType::New();

  // Run the filter
  MetricType::Pointer metric = MetricType::New();
  metric->SetFixedImage(fix);
  metric->SetMovingImage(mov);
  metric->SetFixedMaskImage(ncc_mask);
  metric->SetDeformationField(phi);
  metric->SetWeights(vnl_vector<float>(1, 1.0));
  metric->SetComputeGradient(true);
  metric->GetMetricOutput()->Graft(nccmap);
  metric->GetDeformationGradientOutput()->Graft(grad);
  metric->SetRadius(radius);
  metric->SetWorkingImage(work);

  // Set the moving mask image for weighted mode
  if(weighted)
    {
    metric->SetWeighted(weighted);
    metric->SetWeightScalingExponent(2);
    metric->SetMovingMaskImage(wgt);
    }

  metric->Update();

  // Check the results
  double test_eps = 1e-5;
  int status = 0;
  double *expected_NCC = weighted ? expected_NCC_wgt : expected_NCC_unw;
  double *expected_Gradient = weighted ? expected_Gradient_wgt : expected_Gradient_unw;
  for(unsigned int i = 0; i < 50; i++)
    {
    itk::Index<2> pos = {{i,0}};
    if(fabs(nccmap->GetPixel(pos) - expected_NCC[i]) > test_eps)
      {
      std::cerr << "NCC mismatch: expected " <<
                   expected_NCC[i] << " got " <<
                   nccmap->GetPixel(pos) << std::endl;
      status = -1;
      }
    if(fabs(grad->GetPixel(pos)[0] - expected_Gradient[i]) > test_eps)
      {
      std::cerr << "Gradient mismatch: expected " <<
                   expected_Gradient[i] << " got " <<
                   grad->GetPixel(pos)[0] << std::endl;
      status = -1;
      }
    }

  if(status == 0)
    std::cout << "Success" << std::endl;

  return status;
}

template <unsigned int VDim>
int RunMetricVoxelwiseGradientTest(CommandLineHelper &cl)
{
  // Set up greedy parameters for this test
  GreedyParameters gp;
  gp.dim = VDim;
  gp.mode = GreedyParameters::GREEDY;

  // Read required parameters
  double epsilon = cl.read_double();
  double tol = cl.read_double();

  bool minimization_mode = true;

  // List of greedy commands that are recognized by this test command
  std::set<std::string> greedy_cmd {
    "-m", "-threads", "-i", "-it", "-gm", "-mm", "-ia", "-bg", "-id"
  };

  // Parse the parameters
  std::string arg;
  while(cl.read_command(arg))
    {
    if(greedy_cmd.find(arg) != greedy_cmd.end())
      gp.ParseCommandLine(arg, cl);
    else
      throw GreedyException("Unknown test parameter: %s", arg.c_str());
    }

  // Create a helper
  typedef GreedyApproach<VDim> GreedyAPI;
  typedef typename GreedyAPI::LDDMMType LDDMMType;
  GreedyAPI api;
  typename GreedyAPI::OFHelperType of_helper;

  // Configure threading
  api.ConfigThreads(gp);

  // Initialize for one level
  of_helper.SetDefaultPyramidFactors(1);

  // Read the data
  api.ReadImages(gp, of_helper, true);

  // Generate the initial deformation field
  typename GreedyAPI::ImageBaseType *refspace = of_helper.GetReferenceSpace(0);
  typename GreedyAPI::VectorImagePointer phi = GreedyAPI::LDDMMType::new_vimg(refspace);
  typename GreedyAPI::VectorImagePointer grad_metric = GreedyAPI::LDDMMType::new_vimg(refspace);
  typename GreedyAPI::ImagePointer img_metric_1 = GreedyAPI::LDDMMType::new_img(refspace);
  typename GreedyAPI::ImagePointer img_metric_2 = GreedyAPI::LDDMMType::new_img(refspace);

  // Initialize phi to some dummy value
  api.LoadInitialTransform(gp, of_helper, 0, phi);

  // Report RMS displacement
  double rms = sqrt(LDDMMType::vimg_euclidean_norm_sq(phi) / phi->GetBufferedRegion().GetNumberOfPixels());
  printf("RMS displacement: %12.8f\n", rms);

  // Compute the metric and gradient, using minimization mode, which should ensure that
  // the gradient is scaled correctly relative to the metric
  MultiComponentMetricReport metric_report;
  api.EvaluateMetricForDeformableRegistration(gp, of_helper, 0, phi, metric_report, img_metric_1, grad_metric, 1.0, minimization_mode);

  // Interpolator to figure out what kind of sample it is
  typedef LDDMMData<double, VDim> LDDMMType;
  typedef FastLinearInterpolator<
      typename LDDMMType::CompositeImageType, double, VDim,
      typename LDDMMType::ImageType> InterpType;

  InterpType interp(of_helper.GetMovingComposite(0, 0));

  // Choose a set of locations to evaluate the metric
  unsigned int n_samples = 20;
  struct SampleData {
    itk::Index<VDim> pos;
    double fixed_value = 0.0, moving_value = 0.0, weight_value = 0.0;
    double fixed_mask_value = 0.0;
    double mask_vol = 0.0;
    typename GreedyAPI::VectorImageType::PixelType f1, f2, df_analytic, df_numeric;
    vnl_vector_fixed<double, VDim> grad_W, grad_WM;
    typename InterpType::InOut status;
  };
  std::vector<SampleData> samples(n_samples);

  // Sample random vectors
  vnl_random rnd(12345);
  unsigned int kind = 0;
  for(unsigned int is = 0; is < n_samples; is++)
    {
    SampleData &s = samples[is];

    // Alternate inside, border, and outside samples
    typename InterpType::InOut wanted_status = (typename InterpType::InOut) (kind++ % 3);

    // Iterate until we get a sample with the right status
    for(unsigned int attempt = 0; attempt < 1000; attempt++)
      {
      // Initialize the sample to default values
      s = SampleData();

      // Create a random sample
      for(unsigned int k = 0; k < VDim; k++)
        s.pos[k] = rnd.lrand32(0, refspace->GetBufferedRegion().GetSize(k)-1);

      // Check the fixed mask
      s.fixed_mask_value = of_helper.GetFixedMask(0, 0) ? of_helper.GetFixedMask(0, 0)->GetPixel(s.pos) : 1.0;

      // Look up phi at this location
      typename GreedyAPI::VectorImageType::PixelType s_phi = phi->GetPixel(s.pos);
      vnl_vector<double> cix(VDim, 0.0);
      for(unsigned int k = 0; k < VDim; k++)
        cix[k] = s.pos[k] + s_phi[k];

      s.fixed_value = of_helper.GetFixedComposite(0, 0)->GetPixel(s.pos)[0];

      double *p_grad_wm = s.grad_WM.data_block();
      s.status = interp.InterpolateWithGradient(cix.data_block(), &s.moving_value, &p_grad_wm);
      switch(s.status)
        {
        case InterpType::INSIDE:
          s.weight_value = 1.0; break;
        case InterpType::OUTSIDE:
          s.weight_value = 0.0; break;
        case InterpType::BORDER:
          s.weight_value = interp.GetMaskAndGradient(s.grad_W.data_block()); break;
        }

      if(s.status == wanted_status && s.fixed_mask_value > 0)
        break;
      }

    s.mask_vol = metric_report.MaskVolume;

    // Some scaling is unaccounted for
    s.df_analytic = grad_metric->GetPixel(s.pos);

    // if(gp.metric == GreedyParameters::SSD)
    //  s.df_analytic *= -2.0;

    }

  // Compute numerical derivative approximation
  int retval = 0;
  for(auto &s : samples)
    {
    auto orig = phi->GetPixel(s.pos);
    for(unsigned int k = 0; k < VDim; k++)
      {
      auto def1 = orig, def2 = orig;
      def1[k] -= epsilon;
      phi->SetPixel(s.pos, def1);
      api.EvaluateMetricForDeformableRegistration(gp, of_helper, 0, phi, metric_report, img_metric_1, grad_metric, 1.0, minimization_mode);
      double v1 = metric_report.TotalPerPixelMetric;

      def2[k] += epsilon;
      phi->SetPixel(s.pos, def2);
      api.EvaluateMetricForDeformableRegistration(gp, of_helper, 0, phi, metric_report, img_metric_2, grad_metric, 1.0, minimization_mode);
      double v2 = metric_report.TotalPerPixelMetric;

      // We scale by the central mask volume (so that we are actually testing the TotalPerPixelMetric
      // but reporting in units of the whole metric
      if(minimization_mode)
        s.df_numeric[k] = (v2 - v1) / (2 * epsilon);
      else
        s.df_numeric[k] = s.mask_vol * ((v2-v1) / (2 * epsilon));

      phi->SetPixel(s.pos, orig);
      }

    // Compute the relative error
    vnl_vector<double> rel_err_comp(VDim);
    for(unsigned int d = 0; d < VDim; d++)
      rel_err_comp[d] = fabs(s.df_analytic[d] - s.df_numeric[d]) / (0.5 * (fabs(s.df_analytic[d]) + fabs(s.df_numeric[d])) + 1e-6);
    double rel_err_sup = rel_err_comp.inf_norm();

    // Print the comparison
    const char *status_names[] = { "INSIDE", "OUTSIDE", "BORDER" };
    printf("Sample [%s] (%7s)  Num: %s  Anl: %s   Err: %12.9f\n",
           printf_index("%03ld", s.pos).c_str(),
           status_names[s.status],
           printf_vec<VDim,double>("%12.9f", s.df_numeric.GetVnlVector().data_block()).c_str(),
           printf_vec<VDim,double>("%12.9f", s.df_analytic.GetVnlVector().data_block()).c_str(),
           rel_err_sup);

    /*
    std::cout << "W      : " << s.weight_value << std::endl;
    std::cout << "WM     : " << s.moving_value << std::endl;
    std::cout << "Grad-W : " << s.grad_W << std::endl;
    std::cout << "Grad-WM: " << s.grad_WM << std::endl;
    */

    if(rel_err_sup > tol)
      retval = -1;
    }

  // Check using variational derivatives
  vnl_random randy;
  for(unsigned int i = 0; i < 5; i++)
    {
    // Compute the gradient - previous calls have corrupted it
    grad_metric->FillBuffer(typename LDDMMType::Vec(0.0));
    api.EvaluateMetricForDeformableRegistration(gp, of_helper, 0, phi, metric_report, img_metric_1, grad_metric, 1.0, minimization_mode);

    // Create a variation
    typename GreedyAPI::VectorImagePointer variation = GreedyAPI::LDDMMType::new_vimg(refspace);
    LDDMMType::vimg_add_gaussian_noise_in_place(variation, 6.0);
    LDDMMType::vimg_smooth(variation, variation, 8.0);
    typename LDDMMType::ImagePointer idot = LDDMMType::new_img(phi, 0.0);
    LDDMMType::vimg_euclidean_inner_product(idot, grad_metric, variation);
    double ana_deriv = LDDMMType::img_voxel_sum(idot);

    char buffer[256];
    sprintf(buffer, "/tmp/variation%d.nii.gz", i);
    LDDMMType::vimg_write(variation, buffer);

    // Compute numeric derivatives
    LDDMMType::vimg_add_scaled_in_place(phi, variation, epsilon);
    api.EvaluateMetricForDeformableRegistration(gp, of_helper, 0, phi, metric_report, img_metric_1, grad_metric, 1.0, minimization_mode);
    double v2 = metric_report.TotalPerPixelMetric;
    LDDMMType::vimg_add_scaled_in_place(phi, variation, -2.0 * epsilon);
    api.EvaluateMetricForDeformableRegistration(gp, of_helper, 0, phi, metric_report, img_metric_1, grad_metric, 1.0, minimization_mode);
    double v1 = metric_report.TotalPerPixelMetric;
    LDDMMType::vimg_add_scaled_in_place(phi, variation, epsilon);

    // We scale by the central mask volume (so that we are actually testing the TotalPerPixelMetric
    // but reporting in units of the whole metric
    double num_deriv = (minimization_mode)
      ? (v2 - v1) / (2 * epsilon)
      : metric_report.MaskVolume * ((v2-v1) / (2 * epsilon));

    // Compute relative difference
    double rel_diff = 2.0 * std::fabs(ana_deriv - num_deriv) / (std::fabs(ana_deriv) + std::fabs(num_deriv));

    // Compute the difference between the two derivatives
    printf("Variation %d  ANA: %12.8g  NUM: %12.8g  RELDIF: %12.8f\n", i, ana_deriv, num_deriv, rel_diff);

    if(rel_diff > tol)
      retval = -1;
    }

  if(retval == 0)
    std::cout << "Success" << std::endl;

  return retval;
}

template <unsigned int VDim>
int RunAffineGradientTest(CommandLineHelper &cl)
{
  // Set up greedy parameters for this test
  GreedyParameters gp;
  gp.dim = VDim;
  gp.mode = GreedyParameters::GREEDY;

  // Read required parameters
  gp.deriv_epsilon = cl.read_double();
  double tol = cl.read_double();

  // List of greedy commands that are recognized by this test command
  std::set<std::string> greedy_cmd {
    "-m", "-threads", "-i", "-ia", "-gm", "-mm", "-bg"
  };

  // Parse the parameters
  std::string arg;
  while(cl.read_command(arg))
    {
    if(greedy_cmd.find(arg) != greedy_cmd.end())
      gp.ParseCommandLine(arg, cl);
    else
      throw GreedyException("Unknown test parameter: %s", arg.c_str());
    }

  // Create a helper
  typedef GreedyApproach<VDim> GreedyAPI;
  GreedyAPI api;
  typename GreedyAPI::OFHelperType of_helper;

  // Configure threading
  api.ConfigThreads(gp);

  // Initialize for one level
  of_helper.SetDefaultPyramidFactors(1);

  // Add random sampling jitter for affine stability at voxel edges
  of_helper.SetJitterSigma(gp.affine_jitter);

  // Read the data
  api.ReadImages(gp, of_helper, false);

  // Create a cost function
  typedef AbstractAffineCostFunction<VDim, double> AbstractAffineCostFunction;
  AbstractAffineCostFunction *acf = api.CreateAffineCostFunction(gp, of_helper, 0);

  // Initialize the transform
  typename GreedyAPI::LinearTransformType::Pointer tLevel = GreedyAPI::LinearTransformType::New();
  api.InitializeAffineTransform(gp, of_helper, acf, tLevel);

  // Debug the derivatives
  int retval = api.CheckAffineDerivatives(gp, of_helper, acf, tLevel, 0, tol);

  if(retval == 0)
    std::cout << "Success" << std::endl;

  return retval;
}

int RunReg2D3D(CommandLineHelper &cl)
{
  // Set up greedy parameters for this test
  GreedyParameters gp;
  gp.dim = 3;

  // Read mode, etc
  auto mode = cl.read_string();
  if(mode == "def")
    gp.mode = GreedyParameters::GREEDY;
  else if(mode == "aff")
    gp.mode = GreedyParameters::AFFINE;
  else
    throw GreedyException("Unknown mode parameter: %s", mode.c_str());

  // Read required parameters
  double target_value = cl.read_double();
  double tol = cl.read_double();

  // List of greedy commands that are recognized by this test command
  std::set<std::string> greedy_cmd {
    "-m", "-threads", "-i", "-ia", "-m", "-n"
  };

  // Parse the parameters
  std::string arg;
  while(cl.read_command(arg))
    {
    if(greedy_cmd.find(arg) != greedy_cmd.end())
      gp.ParseCommandLine(arg, cl);
    else
      throw GreedyException("Unknown test parameter: %s", arg.c_str());
    }

  // Create a helper
  typedef GreedyApproach<3> GreedyAPI;
  GreedyAPI api;

  // Configure threading
  api.ConfigThreads(gp);

  // Set values specific to 2D/3D
  gp.flag_zero_last_dim = true;
  if(gp.mode == GreedyParameters::GREEDY)
    {
    gp.reference_space_padding = std::vector<int>(3, 0);
    gp.reference_space_padding[2] = 4;
    gp.background = atof("NaN");
    api.RunDeformable(gp);
    }
  else
    {
    api.RunAffine(gp);
    }

  double final_metric = api.GetLastMetricReport().TotalPerPixelMetric;
  std::cout << "Final metric: " << final_metric << " vs. target value: " << target_value << std::endl;
  int rc = std::fabs(final_metric - target_value) < tol;
  std::cout << "Test " << (rc ? "Succeeded" : "Failed") << std::endl;
  return rc;
}

template <unsigned int VDim>
int
RunTetraJacobianRegularizationTest(std::string fn_refspace, std::string fn_mesh)
{
  typedef TetraMeshConstraints<double, VDim> TMC;
  vtkSmartPointer<vtkUnstructuredGrid> tetra;
  typename TMC::ImageBaseType::Pointer refspace;
  if(fn_refspace.size())
    refspace = LDDMMData<double, VDim>::cimg_read(fn_refspace.c_str());
  if(fn_mesh.size())
    {
    vtkSmartPointer<vtkPointSet> point_set = ReadMesh(fn_mesh.c_str());
    tetra = dynamic_cast<vtkUnstructuredGrid *>(point_set.GetPointer());
    }

  return TMC::TestDerivatives(refspace, tetra) ? 0 : -1;
}

template <unsigned int VDim>
int
RunDifferentiableScalingAndSquaringTest(double noise_amplitude = 8.0, double noise_sigma = 1.0)
{
  typedef ScalingAndSquaringLayer<VDim, double> SSQLayer;
  return SSQLayer::TestDerivatives(noise_amplitude, noise_sigma) ? 0 : -1;
}

template <unsigned int VDim>
int
RunDifferentiableSelfCompositionTest()
{
  typedef DisplacementSelfCompositionLayer<VDim, double> CompLayer;
  return CompLayer::TestDerivatives() ? 0 : -1;
}

template <unsigned int VDim>
int
RunSVFSmoothnessRegularizerTest()
{
  typedef DisplacementFieldSmoothnessLoss<VDim, double> Layer;
  return Layer::TestDerivatives() ? 0 : -1;
}

#include <itkTimeProbe.h>

template <unsigned int VDim>
int
RunFastGaussianSmoothingTest(std::string fn_source, std::string fn_target, double sigma_vox)
{
  typedef LDDMMData<double, VDim> LDDMMType;
  typename LDDMMType::Vec sigma; sigma.Fill(sigma_vox);
  itk::TimeProbe probe_fast, probe_baseline;

  // Create a sigma specification
  typename LDDMMType::SmoothingSigmas sigma_spec(sigma_vox, false);

  // Run the baseline filter
  auto src_bl = LDDMMType::cimg_read(fn_source.c_str());
  probe_baseline.Start();
  LDDMMType::cimg_smooth(src_bl, src_bl, sigma_spec, LDDMMType::ITK_RECURSIVE);
  probe_baseline.Stop();

  // Run the experimental code
  auto src = LDDMMType::cimg_read(fn_source.c_str());
  probe_fast.Start();
  LDDMMType::cimg_smooth(src, src, sigma_spec, LDDMMType::FAST_ZEROPAD);
  probe_fast.Stop();
  LDDMMType::cimg_write(src, fn_target.c_str());

  // Report times
  std::cout << "Baseline time: " << probe_baseline.GetTotal() << std::endl;
  std::cout << "Fast code time: " << probe_fast.GetTotal() << std::endl;

  return 0;
}

int main(int argc, char *argv[])
{
  // Check for the environment variable of test data
  if(!itksys::SystemTools::GetEnv("GREEDY_TEST_DATA_DIR", data_root))
    data_root = itksys::SystemTools::GetCurrentWorkingDirectory();

  if(argc < 2)
    return usage();

  CommandLineHelper cl(argc, argv);
  cl.set_data_root(data_root.c_str());

  std::string cmd = cl.read_arg();
  if(cmd == "phantom")
    {
    return RunPhantomTest(cl);
    }
  else if(cmd == "grad_metric_phi")
    {
    int dim = cl.read_integer();
    if(dim == 2)
      return RunMetricVoxelwiseGradientTest<2>(cl);
    else if (dim == 3)
      return RunMetricVoxelwiseGradientTest<3>(cl);
    }
  else if(cmd == "grad_metric_aff")
    {
    int dim = cl.read_integer();
    if(dim == 2)
      return RunAffineGradientTest<2>(cl);
    else if (dim == 3)
      return RunAffineGradientTest<3>(cl);
    }
  else if(cmd == "reg_2d_3d")
    {
    return RunReg2D3D(cl);
    }
  else if(cmd == "ncc_gradient_vs_matlab")
    {
    int wgt = cl.read_integer();
    return BasicWeightedNCCGradientTest(wgt != 0);
    }
  else if(cmd  == "masked_interpolation_test")
    {
    int dim = cl.read_integer();
    if(dim == 2)
      return RunMaskedInterpolationTest<2>();
    else if(dim == 3)
      return RunMaskedInterpolationTest<3>();
    else return -1;
    }
  else if(cmd == "tet_jac_reg")
    {
    int dim = cl.read_integer();
    std::string refspace = cl.is_at_end() ? std::string() : cl.read_existing_filename();
    std::string mesh = cl.is_at_end() ? std::string() : cl.read_existing_filename();
    if(dim == 2)
      return RunTetraJacobianRegularizationTest<2>(refspace, mesh);
    else if(dim == 3)
      return RunTetraJacobianRegularizationTest<3>(refspace, mesh);
    else return -1;
    }
  else if(cmd == "comp_layer")
    {
    int dim = cl.read_integer();
    if(dim == 2)
      return RunDifferentiableSelfCompositionTest<2>();
    else if(dim == 3)
      return RunDifferentiableSelfCompositionTest<3>();
    else return -1;
    }
  else if(cmd == "ssq_layer")
    {
    int dim = cl.read_integer();
    double noise_ampl = cl.is_at_end() ? 8.0 : cl.read_double();
    double noise_sigma = cl.is_at_end() ? 1.0 : cl.read_double();
    if(dim == 2)
      return RunDifferentiableScalingAndSquaringTest<2>(noise_ampl, noise_sigma);
    else if(dim == 3)
      return RunDifferentiableScalingAndSquaringTest<3>(noise_ampl, noise_sigma);
    else return -1;
    }
  else if(cmd == "svf_smoothness_reg")
    {
    int dim = cl.read_integer();
    if(dim == 2)
      return RunSVFSmoothnessRegularizerTest<2>();
    else if(dim == 3)
      return RunSVFSmoothnessRegularizerTest<3>();
    else return -1;
    }
  else if(cmd == "fast_smoothing")
    {
    int dim = cl.read_integer();
    std::string fn_src = cl.read_existing_filename();
    std::string fn_trg = cl.read_output_filename();
    double sigma = cl.read_double();
    if(dim == 2)
      return RunFastGaussianSmoothingTest<2>(fn_src, fn_trg, sigma);
    else if(dim == 3)
      return RunFastGaussianSmoothingTest<3>(fn_src, fn_trg, sigma);
    else return -1;
    }
  else return usage();
};
