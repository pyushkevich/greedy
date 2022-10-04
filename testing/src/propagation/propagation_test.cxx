#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <exception>
#include "PropagationAPI.h"
#include "PropagationIO.h"
#include "PropagationCommon.h"
#include "PropagationTools.h"
#include "itkImageFileReader.h"

using namespace propagation;

template<typename TReal>
int run_test(int argc, char *argv[])
{
  PROPAGATION_DATA_TYPEDEFS
  using PTools = PropagationTools<TReal>;

  std::cout << "-- [run_test] Start running" << std::endl;

  PropagationInputBuilder<TReal> ib;

  auto img4d =
      PTools::template ReadImage<TImage4D>("/Users/jileihao/data/data_propagation/bav07/bav07.nii.gz");
  ib.SetImage4D(img4d);

  auto seg3d =
      PTools::template ReadImage<TLabelImage3D>(
        "/Users/jileihao/data/data_propagation/bav07/seg05_bav07_root_labeled_zflip.nii.gz");
  ib.SetReferenceSegmentationIn3D(seg3d);

  ib.SetReferenceTimePoint(5);
  ib.SetTargetTimePoints({3,4,6,7});
  ib.SetAffineDOF(GreedyParameters::AffineDOF::DOF_RIGID);
  ib.SetMultiResolutionSchedule({50, 50});
  ib.SetRegistrationMetric(GreedyParameters::SSD);
  ib.SetResliceMetricToLabel(0.2, false);
  ib.SetGreedyVerbosity(GreedyParameters::Verbosity::VERB_NONE);
  ib.SetPropagationVerbosity(PropagationParameters::Verbosity::VERB_DEFAULT);

  PropagationAPI<TReal> api(ib.BuildPropagationInput());
  api.Run();
  auto output = api.GetOutput();

  auto out_seg = output->GetSegmentation3D(4);

  out_seg->Print(std::cout);

  return EXIT_SUCCESS;

}

int main (int argc, char *argv[])
{
  std::cout << "========================================" << std::endl;
  std::cout << "-- Propagation Test Driver " << std::endl;
  std::cout << "========================================" << std::endl;

  try
  {
    run_test<double>(argc, argv);
  }
  catch (std::exception &ex)
  {
    std::cerr << "[PROPAGATION TEST EXCEPTION CAUGHT]: " << ex.what() << std::endl;
  }

  return EXIT_SUCCESS;
}
