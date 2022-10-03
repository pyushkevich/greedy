#ifndef PROPAGATIONPARAMETERS_H
#define PROPAGATIONPARAMETERS_H

#include <iostream>
#include <string>
#include <vector>
#include "GreedyParameters.h"

namespace propagation
{


struct MeshSpec
{
  std::string fn_mesh;
  std::string fnout_pattern;
};

// Parameters for the segmentation propagation
struct PropagationParameters
{
  std::string fn_img4d;
  std::string fn_seg3d;
  std::string fn_seg4d;
  std::vector<MeshSpec> extra_mesh_list;
  std::string fnsegout_pattern;
  std::string fnmeshout_pattern;
  std::string outdir;
  unsigned int refTP;
  std::vector<unsigned int> targetTPs;

  InterpSpec reslice_spec;
  bool debug = false;
  std::string debug_dir;
  bool writeOutputToDisk = true; // whether to write final output data to disk
  bool use4DSegInput = false; // whether to use 4d segmentation input as reference
};

} // End of namespace propagation

#endif // PROPAGATIONPARAMETERS_H
