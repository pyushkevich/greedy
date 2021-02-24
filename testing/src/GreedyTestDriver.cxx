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
  else if(metric == "SSD")
    gp.metric = GreedyParameters::SSD;
  else if(metric == "NMI")
    gp.metric = GreedyParameters::NMI;

  // Set up the input filenames
  std::string fn_fix = GetFileName("phantom%02d_fixed.nii.gz", phantom_fixed_idx);
  std::string fn_mov = GetFileName("phantom%02d_moving.nii.gz", phantom_moving_idx);
  std::string fn_mask = GetFileName("phantom01_mask.nii.gz");

  gp.inputs.push_back(ImagePairSpec(fn_fix, fn_mov));
  if(use_mask)
    gp.gradient_mask = fn_mask;

  gp.affine_dof = dof == 6 ? GreedyParameters::DOF_RIGID : (
                               dof == 7 ? GreedyParameters::DOF_SIMILARITY :
                                          GreedyParameters::DOF_AFFINE);

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

#include <MultiComponentUnweightedNCCImageMetric.h>
#include <itkMultiThreaderBase.h>

int BasicUnweightedNCCGradientTest()
{
  itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(1);
  itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1);

  // Fixed and moving values from MATLAB
  double F[] = {
    9.296161e-01, 3.163756e-01, 1.839188e-01, 2.045603e-01, 5.677250e-01,
    5.955447e-01, 9.645145e-01, 6.531771e-01, 7.489066e-01, 6.535699e-01,
    7.477148e-01, 9.613067e-01, 8.388298e-03, 1.064444e-01, 2.987037e-01,
    6.564112e-01, 8.098126e-01, 8.721759e-01, 9.646476e-01, 7.236853e-01,
    6.424753e-01, 7.174536e-01, 4.675990e-01, 3.255847e-01, 4.396446e-01,
    7.296891e-01, 9.940146e-01, 6.768737e-01, 7.908225e-01, 1.709143e-01,
    2.684928e-02, 8.003702e-01, 9.037225e-01, 2.467621e-02, 4.917473e-01,
    5.262552e-01, 5.963660e-01, 5.195755e-02, 8.950895e-01, 7.282662e-01,
    8.183500e-01, 5.002228e-01, 8.101894e-01, 9.596853e-02, 2.189500e-01,
    2.587191e-01, 4.681058e-01, 4.593732e-01, 7.095098e-01, 1.780530e-01};

  double M[] = {
    5.314499e-01, 1.677422e-01, 7.688139e-01, 9.281705e-01, 6.094937e-01,
    1.501835e-01, 4.896267e-01, 3.773450e-01, 8.486014e-01, 9.110972e-01,
    3.838487e-01, 3.154959e-01, 5.683942e-01, 1.878180e-01, 1.258415e-01,
    6.875958e-01, 7.996067e-01, 5.735366e-01, 9.732300e-01, 6.340544e-01,
    8.884217e-01, 4.954148e-01, 3.516165e-01, 7.142304e-01, 5.039291e-01,
    2.256376e-01, 2.449744e-01, 7.928007e-01, 4.951724e-01, 9.150937e-01,
    9.453718e-01, 5.332322e-01, 2.524926e-01, 7.208621e-01, 3.674388e-01,
    4.986484e-01, 2.265750e-01, 3.535656e-01, 6.508518e-01, 3.129329e-01,
    7.687354e-01, 7.818371e-01, 8.524095e-01, 9.499057e-01, 1.073229e-01,
    9.107254e-01, 3.360552e-01, 8.263804e-01, 8.981006e-01, 4.271530e-02};

  // Expected NCC and grad
  double expected_NCC[] = {
    2.583758e-01, 2.887292e-02, -7.598351e-02, -3.301155e-01, -4.029192e-01,
    -3.337573e-01, 6.243161e-02, 2.681470e-02, -5.688283e-02, -2.121289e-01,
    -2.607225e-02, 6.500015e-03, -4.144126e-03, 1.739461e-02, 4.146321e-01,
    7.681433e-01, 7.973615e-01, 2.968104e-01, 1.687726e-02, 6.934884e-02,
    5.308758e-01, 2.892673e-03, 9.323315e-03, -4.469545e-01, -6.587454e-01,
    -3.937374e-01, -2.057817e-01, -6.310519e-01, -8.129745e-01, -7.990482e-01,
    -9.159599e-01, -7.866231e-01, -7.059908e-01, -5.876293e-01, -7.680782e-01,
    -3.205142e-01, 2.816491e-01, 1.511195e-01, 2.684958e-01, 1.820792e-01,
    -1.567554e-03, -2.007631e-01, 9.088352e-02, 3.537281e-02, 4.532429e-03,
    -4.510806e-02, 2.006805e-01, 2.598261e-01, 7.862010e-01, 8.984511e-01};

  double expected_Gradient[] = {
    -7.472937e-01, -5.277118e-01, -3.579529e-01, 4.176036e-01, -4.569258e-01,
    -4.568537e-01, -5.054241e-01, -7.442020e-01, 3.376598e-02, 2.415960e-01,
    5.414401e-03, 5.056736e-01, 9.494791e-01, 8.323091e-02, 3.091146e-01,
    -2.553842e-01, 2.765792e-02, 1.874931e+00, -9.152049e-01, -1.451540e-01,
    9.669073e-01, -4.455490e-01, -1.247729e+00, 2.585849e-01, 9.120287e-01,
    -2.909071e-02, 1.986637e+00, -1.455216e+00, 3.117488e-01, -2.868933e-02,
    6.609938e-01, -1.281233e+00, 4.438519e-01, 1.324770e+00, -2.532062e-01,
    -4.327096e-01, 2.455944e-01, -2.508391e+00, -8.856205e-01, 9.593683e-01,
    1.791241e-02, -6.279737e-02, 1.738582e-01, 2.053536e+00, -5.635263e-01,
    9.575991e-01, 7.792790e-01, -4.770163e-02, -1.638606e+00, -8.836335e-03};

  // Generate images from this data
  typedef LDDMMData<double, 2> LDDMMType;

  // Create a reference space
  typedef itk::Image<unsigned char, 2> RefImageType;
  RefImageType::Pointer ref = RefImageType::New();
  RefImageType::RegionType region;
  region.SetSize(0, 50);
  region.SetSize(1, 1);
  ref->SetRegions(region);

  // Load test data
  LDDMMType::CompositeImagePointer fix = LDDMMType::new_cimg(ref.GetPointer(), 1);
  LDDMMType::CompositeImagePointer mov = LDDMMType::new_cimg(ref.GetPointer(), 1);
  for(unsigned int i = 0; i < 50; i++)
    {
    fix->GetBufferPointer()[i] = F[i];
    mov->GetBufferPointer()[i] = M[i];
    }

  // Create zero deformation
  LDDMMType::VectorImagePointer phi = LDDMMType::new_vimg(ref.GetPointer());

  // Create outputs
  LDDMMType::ImagePointer nccmap = LDDMMType::new_img(ref.GetPointer());
  LDDMMType::VectorImagePointer grad = LDDMMType::new_vimg(ref.GetPointer());

  // Create the filter
  typedef DefaultMultiComponentImageMetricTraits<double, 2> TraitsType;
  typedef MultiComponentUnweightedNCCImageMetric<TraitsType> MetricType;

  // Create the working image
  LDDMMType::CompositeImagePointer work = LDDMMType::new_cimg(ref.GetPointer(), 10);

  // Run the filter
  MetricType::Pointer metric = MetricType::New();
  metric->SetFixedImage(fix);
  metric->SetMovingImage(mov);
  metric->SetDeformationField(phi);
  metric->SetWeights(vnl_vector<float>(1, 1.0));
  metric->SetComputeGradient(true);
  metric->GetMetricOutput()->Graft(nccmap);
  metric->GetDeformationGradientOutput()->Graft(grad);
  metric->SetRadius(itk::Size<2>({{2,0}}));
  metric->SetWorkingImage(work);
  metric->Update();

  // Check the results
  double test_eps = 1e-6;
  int status = 0;
  for(unsigned int i = 0; i < 50; i++)
    {
    itk::Index<2> pos({{i,0}});
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

int RunWeightedNCCGradientTest(CommandLineHelper &cl)
{
  // Set up greedy parameters for this test
  GreedyParameters gp;
  gp.dim = 3;
  gp.mode = GreedyParameters::GREEDY;

  // TODO: remove later
  gp.threads = 1;

  // Which phantom to use
  int phantom_fixed_idx = cl.read_integer();
  int phantom_moving_idx = cl.read_integer();

  // Read the metric - this determines which image pair to use
  std::string metric = cl.read_string();
  double epsilon = cl.read_double();

  // Configure the degrees of freedom
  if(metric == "NCC")
    {
    gp.metric = GreedyParameters::NCC;
    gp.metric_radius = std::vector<int>(3, 2);
    }
  else if(metric == "SSD")
    gp.metric = GreedyParameters::SSD;
  else if(metric == "NMI")
    gp.metric = GreedyParameters::NMI;

  // Set up the input filenames
  std::string fn_fix = GetFileName("phantom%02d_fixed.nii.gz", phantom_fixed_idx);
  std::string fn_mov = GetFileName("phantom%02d_moving.nii.gz", phantom_moving_idx);
  std::string fn_init_rigid = GetFileName("phantom01_rigid.mat");

  gp.inputs.push_back(ImagePairSpec(fn_fix, fn_mov));

  // Create a helper
  typedef GreedyApproach<3> GreedyAPI;
  GreedyAPI api;
  GreedyAPI::OFHelperType of_helper;

  // Configure threading
  api.ConfigThreads(gp);

  // Initialize for one level
  of_helper.SetDefaultPyramidFactors(1);

  // Read the data
  api.ReadImages(gp, of_helper);

  // Generate the initial deformation field
  GreedyAPI::ImageBaseType *refspace = of_helper.GetReferenceSpace(0);
  GreedyAPI::VectorImagePointer phi = GreedyAPI::LDDMMType::new_vimg(refspace);
  GreedyAPI::VectorImagePointer grad_metric = GreedyAPI::LDDMMType::new_vimg(refspace);
  GreedyAPI::ImagePointer img_metric_1 = GreedyAPI::LDDMMType::new_img(refspace);
  GreedyAPI::ImagePointer img_metric_2 = GreedyAPI::LDDMMType::new_img(refspace);

  // Initialize phi to some dummy value
  gp.affine_init_mode = RAS_FILENAME;
  gp.affine_init_transform = fn_init_rigid;
  api.LoadInitialTransform(gp, of_helper, 0, phi);

  // Compute the metric and gradient
  MultiComponentMetricReport metric_report;
  api.EvaluateMetricForDeformableRegistration(gp, of_helper, 0, phi, metric_report, img_metric_1, grad_metric, 1.0);

  // Choose a set of locations to evaluate the metric
  unsigned int n_samples = 10;
  struct SampleData {
    itk::Index<3> pos;
    double fixed_value, moving_value;
    GreedyAPI::VectorImageType::PixelType f1, f2, df_analytic, df_numeric;
  };
  std::vector<SampleData> samples(n_samples);

  // Sample random vectors
  vnl_random rnd(12345);
  for(auto &s : samples)
    {
    // We are ok with 1/5 of the samples being outside
    bool ok_with_zero = rnd.lrand32(1,5) == 5;
    do
      {
      for(unsigned int k = 0; k < 3; k++)
        s.pos[k] = rnd.lrand32(0, refspace->GetBufferedRegion().GetSize(k)-1);
      s.fixed_value = of_helper.GetFixedComposite(0)->GetPixel(s.pos)[0];
      s.moving_value = of_helper.GetMovingComposite(0)->GetPixel(s.pos)[0];
      }
    while(s.fixed_value == 0 && !ok_with_zero);

    // Some scaling is unaccounted for
    s.df_analytic = grad_metric->GetPixel(s.pos);
    if(gp.metric == GreedyParameters::SSD)
      s.df_analytic *= -2.0;
    }

  // Compute numerical derivative approximation
  for(auto &s : samples)
    {
    auto orig = phi->GetPixel(s.pos);
    for(unsigned int k = 0; k < 3; k++)
      {
      auto def1 = orig, def2 = orig;
      def1[k] -= epsilon;
      phi->SetPixel(s.pos, def1);
      api.EvaluateMetricForDeformableRegistration(gp, of_helper, 0, phi, metric_report, img_metric_1, grad_metric, 1.0);

      def2[k] += epsilon;
      phi->SetPixel(s.pos, def2);
      api.EvaluateMetricForDeformableRegistration(gp, of_helper, 0, phi, metric_report, img_metric_2, grad_metric, 1.0);

      // We cannot rely on the metric report because of numerical errors (huge number of additions
      // across all pixels in the image) so instead we should integrate the difference between the
      // two metric images
      GreedyAPI::LDDMMType::img_subtract_in_place(img_metric_2, img_metric_1);
      double del = GreedyAPI::LDDMMType::img_voxel_sum(img_metric_2);
      s.df_numeric[k] = del / (2 * epsilon);
      // s.df_numeric[k] = (img_metric_2->GetPixel(s.pos) - img_metric_1->GetPixel(s.pos)) / (2 * epsilon);

      GreedyAPI::LDDMMType::img_write(img_metric_2, "delta.nii.gz");

      phi->SetPixel(s.pos, orig);
      }

    // Scale both vectors by 10000
    printf("Sample [%03ld,%03ld,%03ld]  Int: %6.2f %6.2f   Num: %10.4f %10.4f %10.4f   Anl: %10.4f %10.4f %10.4f\n",
           s.pos[0], s.pos[1], s.pos[2], s.fixed_value, s.moving_value,
           s.df_numeric[0], s.df_numeric[1], s.df_numeric[2],
           s.df_analytic[0], s.df_analytic[1], s.df_analytic[2]);
    }

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
  std::string cmd = cl.read_arg();
  if(cmd == "phantom")
    {
    return RunPhantomTest(cl);
    }
  else if(cmd == "grad_ncc_def")
    {
    return RunWeightedNCCGradientTest(cl);
    }
  else if(cmd == "grad_ncc_unw_simple")
    {
    return BasicUnweightedNCCGradientTest();
    }
  else return usage();
};
