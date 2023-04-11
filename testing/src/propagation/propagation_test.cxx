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
#include <vtkDecimatePro.h>

using namespace propagation;

typedef double TReal;
PROPAGATION_DATA_TYPEDEFS
typedef PropagationAPI<TReal> PropagationAPIType;
using PTools = PropagationTools<TReal>;

// Global variable storing the test data root
std::string data_root;

int usage()
{

}

template<typename TReal>
int run_basic(CommandLineHelper &)
{
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
  PropagationInputBuilder<TReal> ib;

  auto img4d = PTools::template ReadImage<TImage4D>("./propagation/img4d.nii.gz");
  ib.SetImage4D(img4d);

  auto seg3d = PTools::template ReadImage<TLabelImage3D>("./propagation/seg05.nii.gz");
  ib.SetReferenceSegmentationIn3D(seg3d);

  auto mesh_dc = PTools::GetMeshFromLabelImage(seg3d);
  vtkNew<vtkDecimatePro> fltDecimate;
  fltDecimate->AddInputData(mesh_dc);
  fltDecimate->SetTargetReduction(0.7);
  fltDecimate->Update();
  mesh_dc = fltDecimate->GetOutput();

  auto dc_count = mesh_dc->GetNumberOfPoints();

  std::string meshTag = "mesh_dc";

  ib.SetReferenceTimePoint(5);
  ib.SetTargetTimePoints({6});
  ib.SetAffineDOF(GreedyParameters::AffineDOF::DOF_RIGID);
  ib.SetMultiResolutionSchedule({10,10});
  ib.SetRegistrationMetric(GreedyParameters::SSD);
  ib.SetResliceMetricToLabel(0.2, false);
  ib.SetGreedyVerbosity(GreedyParameters::Verbosity::VERB_NONE);
  ib.SetPropagationVerbosity(PropagationParameters::Verbosity::VERB_DEFAULT);
  ib.AddExtraMeshToWarp(mesh_dc, meshTag);

  // build input and run
  PropagationAPIType api(ib.BuildPropagationInput());
  api.Run();

  // validate results
  auto propaOut = api.GetOutput();
  auto extraMeshSeries = propaOut->GetAllExtraMeshSeries();
  for (auto tagKv : extraMeshSeries)
    {
    std::cout << "-- tag: " << tagKv.first << std::endl;
    for (auto tpKv : tagKv.second)
      {
      std::cout << "---- tp: " << tpKv.first << "; pointCnt: "
                << tpKv.second->GetNumberOfPoints()
                << std::endl;

      if (tpKv.second->GetNumberOfPoints() != dc_count)
        {
        std::cout << "[Extra Mesh Test] tag: " << tagKv.first
                  << " ; tp: " << tpKv.first
                  << " ; pointCnt: " << tpKv.second->GetNumberOfPoints()
                  << " ; expected: " << dc_count << std::endl;

        return EXIT_FAILURE;
        }
      }
    }



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
