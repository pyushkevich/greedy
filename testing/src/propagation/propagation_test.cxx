#include <iostream>
#include <string>
#include <memory>
#include <exception>
#include "CommandLineHelper.h"
#include "PropagationAPI.h"
#include "PropagationIO.h"
#include "PropagationCommon.h"
#include "PropagationTools.h"
#include "itkLabelOverlapMeasuresImageFilter.h"

using namespace propagation;

// Global variable storing the test data root
std::string data_root;

int usage()
{

}

template<typename TReal>
int run_basic(CommandLineHelper &)
{
  PROPAGATION_DATA_TYPEDEFS
  using PTools = PropagationTools<TReal>;
  PropagationInputBuilder<TReal> ib;

  auto img4d = PTools::template ReadImage<TImage4D>("./propagation/img4d.nii.gz");
  ib.SetImage4D(img4d);

  auto seg3d = PTools::template ReadImage<TLabelImage3D>("./propagation/seg05.nii.gz");
  ib.SetReferenceSegmentationIn3D(seg3d);

  ib.SetReferenceTimePoint(5);
  ib.SetTargetTimePoints({1,2,3,4,6,7});
  ib.SetAffineDOF(GreedyParameters::AffineDOF::DOF_RIGID);
  ib.SetMultiResolutionSchedule({10,10});
  ib.SetRegistrationMetric(GreedyParameters::SSD);
  ib.SetResliceMetricToLabel(0.2, false);
  ib.SetGreedyVerbosity(GreedyParameters::Verbosity::VERB_NONE);
  ib.SetPropagationVerbosity(PropagationParameters::Verbosity::VERB_DEFAULT);

  PropagationAPI<TReal> api(ib.BuildPropagationInput());
  api.Run();
  auto output = api.GetOutput();
  auto out_seg = output->GetSegmentation3D(4);
  auto out_seg4d = output->GetSegmentation4D();

  // Compare generated seg4d vs reference seg4d
  const double tolerance = 0.9;
  auto seg4d_ref = PTools::template ReadImage<TLabelImage4D>("./propagation/seg4d_resliced.nii.gz");
  auto fltOverlap = itk::LabelOverlapMeasuresImageFilter<TLabelImage4D>::New();
  fltOverlap->SetSourceImage(seg4d_ref);
  fltOverlap->SetTargetImage(out_seg4d);
  fltOverlap->Update();
  double dice = fltOverlap->GetDiceCoefficient();

  std::cout << "[test_propagation] dice coefficient = " << dice << std::endl;
  int rc = (dice > tolerance ? EXIT_SUCCESS : EXIT_FAILURE);

  return rc;
}

template<typename TReal>
int run_extra_mesh(CommandLineHelper &)
{
  PROPAGATION_DATA_TYPEDEFS
  using PTools = PropagationTools<TReal>;
  PropagationInputBuilder<TReal> ib;

  auto img4d = PTools::template ReadImage<TImage4D>("./propagation/img4d.nii.gz");
  ib.SetImage4D(img4d);

  auto seg3d = PTools::template ReadImage<TLabelImage3D>("./propagation/seg05.nii.gz");
  ib.SetReferenceSegmentationIn3D(seg3d);

  ib.SetReferenceTimePoint(5);
  ib.SetTargetTimePoints({1,2,3,4,6,7});
  ib.SetAffineDOF(GreedyParameters::AffineDOF::DOF_RIGID);
  ib.SetMultiResolutionSchedule({10,10});
  ib.SetRegistrationMetric(GreedyParameters::SSD);
  ib.SetResliceMetricToLabel(0.2, false);
  ib.SetGreedyVerbosity(GreedyParameters::Verbosity::VERB_NONE);
  ib.SetPropagationVerbosity(PropagationParameters::Verbosity::VERB_DEFAULT);

  return EXIT_SUCCESS;
}

int main (int argc, char *argv[])
{
  std::cout << "========================================" << std::endl;
  std::cout << "-- Propagation Test Driver " << std::endl;
  std::cout << "========================================" << std::endl;

  // Check for the environment variable of test data
  if(!itksys::SystemTools::GetEnv("GREEDY_TEST_DATA_DIR", data_root))
    data_root = itksys::SystemTools::GetCurrentWorkingDirectory();

  std::cout << "-- data_root: " << data_root << std::endl;

  CommandLineHelper cl(argc, argv);
  cl.set_data_root(data_root.c_str());

  int rc = EXIT_SUCCESS;

  std::string cmd = cl.read_arg();

  if(cmd == "basic")
    {
    return run_basic<double>(cl);
    }
  else if(cmd == "extra_mesh")
    {
    return run_extra_mesh<double>(cl);
    }
  else return usage();

  return rc;
}
