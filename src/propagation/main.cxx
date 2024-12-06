#include "PropagationAPI.h"
#include "CommandLineHelper.h"
#include "PropagationParameters.hxx"
#include "PropagationInputBuilder.h"
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
  printf("  -i <img4d.nii>                : 4D image that is the base of the segmentation \n");
  printf("  -sr3 <seg.nii>                : 3D segmentation for the reference time point of the 4D base image. \n");
  printf("                                  Only one reference segmentation input is allowed per run.\n");
  printf("                                  This option will override all previous specified -sr or -sr4. \n");
  printf("  -sr4 <seg4d.nii>              : 4D segmentation providing the segmentation for the reference time point. \n");
  printf("                                  Only segementation from the reference time point will be used. \n");
  printf("                                  This option will override all previous specified -sr3 or -sr4. \n");
  printf("  -tpr <timepoint>              : The reference time point of the given segmentation image \n");
  printf("  -tpt <target tp str>          : A comma separated string of target time points for the propagation \n");
  printf("  -o <outdir>                   : Output directory to store propagated segmentations and meshes\n");
  printf("  -sr-op <pattern>              : (optional) Segmentation output filename pattern \n");
  printf("                                  The output filenames can be configured using a c-style pattern string, \n");
  printf("                                  with a timepoint number embedded.\n");
  printf("                                  For example: \"Seg_%%02d_resliced.nii.gz\" will generate \"Seg_05_resliced.nii.gz\" \n");
  printf("                                  for timepoint 5. \n");
  printf("  -sr-mop <pattern>             : (optional) Set the filename pattern for the mesh generated from the provided reference segmentation image. \n");
  printf("                                  Filename pattern without %%format will have time point appended\n");
  printf("  -emr <mesh.vtk> <pattern>     : (optional) Add an extra segmentation mesh for the referenfe tp. Repeat to add multiple meshes. \n");
  printf("  -debug <outdir>               : (optional) Enable debugging mode for propagation: Dump intermediary files to outdir \n");
  printf("  -verbose <value>              : (optional) Set propagation verbosity level (0: none, 1: default, 2: verbose) \n");
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
    if (arg == "-i")
      {
      pParam.fn_img4d = cl.read_existing_filename();
      }
    else if (arg == "-o")
      {
      pParam.outdir = cl.read_output_dir();
      }
    else if (arg == "-sr3")
      {
      pParam.fn_seg3d = cl.read_existing_filename();
      }
    else if (arg == "-sr4")
      {
      pParam.fn_seg4d = cl.read_existing_filename();
      pParam.use4DSegInput = true;
      }
    else if (arg == "-sr-op")
      {
      pParam.fnsegout_pattern = cl.read_string();
      }
    else if (arg == "-sr-mop")
      {
      pParam.fnmeshout_pattern = cl.read_string();
      }
    else if (arg == "-emr")
      {
      MeshSpec meshspec;
      meshspec.cached = false; // mesh is from a file
      meshspec.fn_mesh = cl.read_existing_filename();
      meshspec.fnout_pattern = cl.read_string();
      pParam.extra_mesh_list.push_back(meshspec);
      }
    else if (arg == "-tpr")
      {
      pParam.refTP = cl.read_integer();
      }
    else if (arg == "-tpt")
      {
      std::vector<int> result = cl.read_int_vector(',');
      std::set<int> unique(result.begin(), result.end()); // remove duplicates
      if (unique.size() == 0)
        throw GreedyException("Propagation: Target timepoints list cannot be empty!");

      for (int n : unique)
        {
        if (n <= 0)
          throw GreedyException("%d is not a valid time point value!", n);
        pParam.targetTPs.push_back(n);
        }
      }
    else if (arg == "-debug")
      {
      pParam.debug = true;
      pParam.debug_dir = cl.read_output_dir();
      }
    else if (arg == "-verbose")
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
int run (const PropagationParameters &pParam, const GreedyParameters &gParam)
{
  std::cout << "-- [Propagation] Run started" << std::endl;

  PropagationInputBuilder<TReal> builder;
  builder.ConfigForCLI(pParam, gParam);
  auto pInput = builder.BuildPropagationInput();
  
  PropagationAPI<TReal> api(pInput);
  return api.Run();
}

int main (int argc, char *argv[])
{
  if (argc < 2)
    return usage();

  PropagationParameters pParam; 
  GreedyParameters gParam; // for general greedy parameters

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
