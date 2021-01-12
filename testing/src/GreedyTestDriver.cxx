#include <GreedyAPI.h>
#include <CommandLineHelper.h>
#include <itksys/SystemTools.hxx>
#include <itkMatrixOffsetTransformBase.h>
#include <GreedyException.h>
#include <lddmm_data.h>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>

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
  else return usage();
};
