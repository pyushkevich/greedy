#include "PropagationAPI.h"
#include "CommandLineHelper.h"
#include "PropagationParameters.h"
#include "PropagationIO.h"
#include <iostream>
#include <memory>

using namespace propagation;

// List of greedy commands that are recognized by this mode
const std::set<std::string> greedy_cmd {
  "-threads", "-d", "-m", "-n", "-dump-pyramid", "-dump-metric",
  "-float", "-V"
};

int usage()
{
  printf("greedy_propagation: Segmentation Propagation with Greedy Registration\n");
  printf("usage: \n");
  printf("  greedy_propagation <options> <greedy_options> \n");
  printf("options for propagation: \n");
  printf("  -spi img4d.nii         : 4D image that is the base of the segmentation \n");
  printf("  -sps seg.nii outdir    : 3D segmentation for the reference time point of the 4D base image. \n");
  printf("                           Specify an outdir to save the propagated output segmentations \n");
  printf("                           The output file will have same filename of the reference file, with \n");
  printf("                           target time point number as suffix. \n");
  printf("  -spm mesh.vtk outdir   : Segmentation mesh for the reference time point of the 4D base image \n");
  printf("                           Specify an outdir to save the propagated output meshes \n");
  printf("  -spr timepoint         : The reference time point of the given segmentation image \n");
  printf("  -spt <target tp str>   : A comma separated string of target time points for the propagation \n");
  printf("  -sp-debug <outdir>     : Enable debugging mode for propagation: Dump intermediary files to outdir\n");
  printf("main greedy options accepted: \n");
  printf("  ");
  for (auto cit = greedy_cmd.cbegin(); cit != greedy_cmd.cend(); ++cit)
    printf("%s ", cit->c_str());
  printf("\n");
  return -1;
}

void parse_command_line(int argc, char *argv[],
                        PropagationParameters &pParam, GreedyParameters &gParam)
{
  CommandLineHelper cl(argc, argv);

  std::string arg;

  while (cl.read_command(arg))
    {
    if(arg == "-spi")
      {
      // propagation mode: read input 4D image
      pParam.img4d = cl.read_existing_filename();
      }
    else if(arg == "-sps")
      {
      // propagation mode: add a segmentation pair
      SegmentationSpec segspec;
      pParam.segspec.refseg = cl.read_existing_filename();
      pParam.segspec.outsegdir = cl.read_output_dir();
      }
    else if(arg == "-spm")
      {
      // propagation mode: add mesh pair
      MeshSpec meshspec;
      meshspec.refmesh = cl.read_existing_filename();
      meshspec.outmeshdir = cl.read_output_dir();
      pParam.extra_mesh_list.push_back(meshspec);
      }
    else if(arg == "-spr")
      {
      // propagation mode: read reference time point number
      pParam.refTP = cl.read_integer();
      }
    else if(arg == "-spt")
      {
      // propagation mode: read target timepoint number string
      std::vector<int> result = cl.read_int_vector(',');
      // eliminate any duplicate input
      std::set<int> unique(result.begin(), result.end());
      // validate and push to the parameter
      for (int n : unique)
        {
        if (n <= 0)
          throw GreedyException("%d is not a valid time point value!", n);

        pParam.targetTPs.push_back(n);
        }
      }
    else if (arg == "-sp-interp-spec")
      {
      // propagation mode: read sigmas for label reslicing interpolation mode
      std::string mode = cl.read_string();
      if(mode == "nn" || mode == "NN" || mode == "0")
        {
        pParam.reslice_spec.mode = InterpSpec::NEAREST;
        }
      else if(mode == "linear" || mode == "LINEAR" || mode == "1")
        {
        pParam.reslice_spec.mode = InterpSpec::LINEAR;
        }
      else if(mode == "label" || mode == "LABEL")
        {
        pParam.reslice_spec.mode = InterpSpec::LABELWISE;
        pParam.reslice_spec.sigma.sigma = cl.read_scalar_with_units(
              pParam.reslice_spec.sigma.physical_units);
        }
      else
        {
        std::cerr << "Propagation interpolation spec: Unknown interpolation mode" << std::endl;
        }
      }
    else if (arg == "-sp-debug")
      {
      pParam.debug = true;
      pParam.debug_dir = cl.read_output_dir();
      }
    else if(greedy_cmd.find(arg) != greedy_cmd.end())
      {
      gParam.ParseCommandLine(arg, cl);
      }
    else
      throw GreedyException("Unknown parameter: %s", arg.c_str());
    }
}

template<typename TReal>
int run(PropagationParameters &pParam, GreedyParameters &gParam)
{
  std::cout << "-- [Propagation] Run started" << std::endl;
  auto pInput = PropagationInputBuilder<TReal>::CreateInputForCommandLineRun(pParam, gParam);
  PropagationAPI<TReal> api(pInput);
  return api.Run();
}

int main (int argc, char *argv[])
{
  if (argc < 2)
    return usage();

  PropagationParameters pParam;
  GreedyParameters gParam;
  parse_command_line(argc, argv, pParam, gParam);

  if (gParam.dim != 3)
    throw GreedyException("Invalid dimension %d. Propagation currently only support -d 3", gParam.dim);

  // Run the propagation based on real type
  if (gParam.flag_float_math)
    return run<float>(pParam, gParam);
  else
    return run<double>(pParam, gParam);

  return EXIT_SUCCESS;
}
