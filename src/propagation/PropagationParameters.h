#ifndef PROPAGATIONPARAMETERS_H
#define PROPAGATIONPARAMETERS_H

#include <iostream>
#include <string>
#include <vector>
#include "GreedyParameters.h"

namespace propagation
{

struct SegmentationSpec
{
  // Input reference segmentation
  std::string refseg;

  // Output directory for the propagated segmentation images
  std::string outsegdir;
};

struct MeshSpec
{
  // Input or generated reference mesh
  std::string refmesh;

  // Output directory for the propagated meshes
  std::string outmeshdir;
};

// Parameters for the segmentation propagation
struct PropagationParameters
{
  std::string img4d;
  SegmentationSpec segspec;
  std::vector<MeshSpec> extra_mesh_list;

  unsigned int refTP;
  std::vector<unsigned int> targetTPs;

  InterpSpec reslice_spec;

  bool debug = false;
  std::string debug_dir;
  bool writeOutputToDisk = true; // whether to write final output data to disk
};

} // End of namespace propagation

#endif // PROPAGATIONPARAMETERS_H
