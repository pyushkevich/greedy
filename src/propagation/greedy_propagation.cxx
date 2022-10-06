#include "PropagationAPI.h"
#include "CommandLineHelper.h"
#include "PropagationParameters.h"
#include "PropagationIO.h"
#include <iostream>
#include <memory>

using namespace propagation;

// List of greedy commands that are recognized by this mode
const std::set<std::string> greedy_cmd {
  "-threads", "-m", "-n", "-s", "-dof", "-dump-pyramid", "-dump-metric", "-float", "-V"
};

int usage()
{
  printf("greedy_propagation: Segmentation Propagation with Greedy Registration\n");
  printf("description: \n");
  printf("  greedy_propagation warps a 3d segmentation from one time point to the specified target \n");
  printf("  timepoints in a 4d image by using greedy registrations. \n");
  printf("usage: \n");
  printf("  greedy_propagation <options> <greedy_options> \n");
  printf("options for propagation: \n");
  printf("  -spi img4d.nii        : 4D image that is the base of the segmentation \n");
  printf("  -sps seg.nii          : 3D segmentation for the reference time point of the 4D base image. \n");
  printf("                          Only one reference segmentation input is allowed per run.\n");
  printf("                          This option will override all previous specified -sps or -sps-4d. \n");
  printf("  -sps-4d seg4d.nii     : An alternative way to provide the reference segmentation. \n");
  printf("                          Only segementation from the reference time point will be used. \n");
  printf("                          This option will override all previous specified -sps or -sps-4d. \n");
  printf("  -spo outdir           : Output directory to store propagated segmentations and meshes\n");
  printf("  -sps-op pattern       : Segmentation output filename pattern \n");
  printf("                          The output filenames can be configured using a c-style pattern string, \n");
  printf("                          with a timepoint number embedded.\n");
  printf("                          For example: \"Seg_%%02d_resliced.nii.gz\" will generate \"Seg_05_resliced.nii.gz\" \n");
  printf("                          for timepoint 5. \n");
  printf("                          Filename pattern without %%format will have time point appended\n");
  printf("  -sps-mop pattern      : Segmentation Mesh output filename pattern. \n");
  printf("  -spm mesh.vtk pattern : Segmentation mesh for the reference time point of the 4D base image \n");
  printf("  -spr timepoint        : The reference time point of the given segmentation image \n");
  printf("  -spt <target tp str>  : A comma separated string of target time points for the propagation \n");
  printf("  -sp-debug <outdir>    : Enable debugging mode for propagation: Dump intermediary files to outdir \n");
  printf("  -sp-verbose <value>   : Set propagation verbosity level (0: none, 1: default, 2: verbose) \n");
  printf("main greedy options accepted: \n");
  printf("  ");
  for (auto cit = greedy_cmd.crbegin(); cit != greedy_cmd.crend(); ++cit)
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
    if (arg == "-spi")
      {
      pParam.fn_img4d = cl.read_existing_filename();
      }
    else if (arg == "-spo")
      {
      pParam.outdir = cl.read_output_dir();
      }
    else if (arg == "-sps")
      {
      pParam.fn_seg3d = cl.read_existing_filename();
      pParam.use4DSegInput = false;
      }
    else if (arg == "-sps-4d")
      {
      pParam.fn_seg4d = cl.read_existing_filename();
      pParam.use4DSegInput = true;
      }
    else if (arg == "-sps-op")
      {
      pParam.fnsegout_pattern = cl.read_string();
      }
    else if (arg == "-sps-mop")
      {
      pParam.fnmeshout_pattern = cl.read_string();
      }
    else if (arg == "-spm")
      {
      MeshSpec meshspec;
      meshspec.fn_mesh = cl.read_existing_filename();
      meshspec.fnout_pattern = cl.read_string();
      pParam.extra_mesh_list.push_back(meshspec);
      }
    else if (arg == "-spr")
      {
      pParam.refTP = cl.read_integer();
      }
    else if (arg == "-spt")
      {
      std::vector<int> result = cl.read_int_vector(',');
      std::set<int> unique(result.begin(), result.end()); // remove duplicates
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
      if (mode == "nn" || mode == "NN" || mode == "0")
        {
        pParam.reslice_spec.mode = InterpSpec::NEAREST;
        }
      else if (mode == "linear" || mode == "LINEAR" || mode == "1")
        {
        pParam.reslice_spec.mode = InterpSpec::LINEAR;
        }
      else if (mode == "label" || mode == "LABEL")
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
    else if (arg == "-sp-verbose")
      {
      int level = cl.read_integer();
      if(level < 0 || level >= PropagationParameters::VERB_INVALID)
        throw GreedyException("Invalid propagation verbosity level %d", level);

      pParam.verbosity = (PropagationParameters::Verbosity)level;
      }
    else if (greedy_cmd.count(arg))
      {
      gParam.ParseCommandLine(arg, cl);
      }
    else
      throw GreedyException("Unknown parameter: %s", arg.c_str());
    }
}

template<typename TReal>
int run (PropagationParameters &pParam, GreedyParameters &gParam)
{
  std::cout << "-- [Propagation] Run started" << std::endl;
  auto pInput = PropagationInputBuilder<TReal>::BuildInputForCommandLineRun(pParam, gParam);
  PropagationAPI<TReal> api(pInput);
  return api.Run();
}

int main (int argc, char *argv[])
{
  if (argc < 2)
    return usage();

  PropagationParameters pParam;
  GreedyParameters gParam;

  try
    {
    parse_command_line(argc, argv, pParam, gParam);

    // Run the propagation based on real type
    if (gParam.flag_float_math)
      return run<float>(pParam, gParam);
    else
      return run<double>(pParam, gParam);
    }
  catch (std::exception &ex)
    {
      std::cerr << "ABORTING PROGRAM DUE TO RUNTIME EXCEPTION -- " << ex.what() << std::endl;
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
