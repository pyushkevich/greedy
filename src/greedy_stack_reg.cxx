/*=========================================================================

  Program:   ALFABIS fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  This program is part of ALFABIS: Adaptive Large-Scale Framework for
  Automatic Biomedical Image Segmentation.

  ALFABIS development is funded by the NIH grant R01 EB017255.

  ALFABIS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ALFABIS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ALFABIS.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================*/
#include "CommandLineHelper.h"

#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cerrno>

#include "lddmm_common.h"
#include "lddmm_data.h"

struct StackParameters
{
  std::string manifest, volume_image, output_dir;
  double z_range;

  StackParameters()
    : z_range(0.0) {}
};

struct SliceData
{
  std::string raw_filename;
  std::string unique_id;
  double z_pos;
};

// How to specify how many neighbors a slice will be registered to?
// - minimum of one neighbor
// - maximum of <user_specified> neighbors
// - maximum offset

int usage()
{
  printf("stack_greedy: Paul's histology stack to MRI registration implementation\n");
  printf("Usage: \n");
  printf("  stack_greedy [options]\n");
  printf("Required options: \n");
  printf("  -m <manifest>      : Manifest of slices to be reconstructed in 3D. Each line of\n");
  printf("                       the manifest file contains the following fields:\n");
  printf("                         unique_id         : a unique identifier, for output dirs\n");
  printf("                         z_position        : a float indicating slice z-position\n");
  printf("                         filename          : path to image filename\n");
  printf("  -o <directory>     : Output directory for matrices/warps\n");
  printf("Additional options: \n");
  printf("  -i <image>         : 3D image to use as target of registration\n");
  printf("  -z <offset>        : Maximum range in z when slices are considered neighbors\n");
  printf("Options shared with Greedy: \n");
  printf("  -m metric              : metric (see Greedy docs)\n");
  printf("  -n NxNxN               : number of iterations per level of multi-res (100x100) \n");
  printf("  -threads N             : set the number of allowed concurrent threads\n");
  printf("  -gm-trim <radius>      : generate mask for gradient computation (see Greedy docs)");
  printf("  -search N s_ang s_xyz  : Random search over rigid transforms (see Greedy docs)\n");
}

int main(int argc, char *argv[])
{
  // Parameters specifically for this application
  StackParameters param;

  // Parameters for running Greedy in general
  GreedyParameters gparam;
  if(argc < 2)
    return usage();

  // List of greedy commands that are recognized by this code
  std::set<std::string> greedy_cmd {
    "-m", "-n", "-threads", "-gm-trim", "-search", "-dof"
  };

  try
  {
  CommandLineHelper cl(argc, argv);
  while(!cl.is_at_end())
    {
    // Read the next command
    std::string arg = cl.read_command();

    if(arg == "-m")
      {
      param.manifest = cl.read_existing_filename();
      }
    else if(arg == "-o")
      {
      param.output_dir = cl.read_string();
      }
    else if(arg == "-i")
      {
      param.volume_image = cl.read_existing_filename();
      }
    else if(arg == "-z")
      {
      param.z_range = cl.read_double();
      }
    else if(greedy_cmd.find(arg) != greedy_cmd.end())
      {
      gparam.ParseCommandLine(arg, cl);
      }
    else
      {
      std::cerr << "Unknown parameter " << arg << std::endl;
      return -1;
      }
    }
  }
  catch(std::exception &exc)
  {
    std::cerr << "ABORTING PROGRAM DUE TO RUNTIME EXCEPTION -- "
              << exc.what() << std::endl;
    return -1;
  }

  // Run the main portion of the code

  // Maintain a flat list of slices (in manifest order)
  std::vector<SliceData> slices;

  // Maintain a list of slices sorted by the z-position
  typedef std::pair<double, unsigned int> slice_ref;
  typedef std::set<slice_ref> slice_ref_set;
  slice_ref_set z_sort;

  // Read the manifest file
  std::ifstream fin(param.manifest);
  std::string f_line;
  while(std::getline(fin, f_line))
    {
    // Read the values from the manifest
    std::istringstream iss(f_line);
    SliceData slice;
    if(!(iss >> slice.unique_id >> slice.z_pos >> slice.raw_filename))
      throw GreedyException("Error reading manifest file, line %s", f_line.c_str());

    // Add to sorted list
    z_sort.insert(std::make_pair(slice.z_pos, slices.size()));
    slices.push_back(slice);
    }

  // Set up the graph of all registrations. Each slice is a node and edges are between each
  // slice and its closest slices, as well as between each slice and slices in the z-range
  // typedef std::pair<unsigned int, unsigned int> GraphEdge;
  // typedef std::set<GraphEdge> Graph;
  // Graph slice_graph;

  // We keep for each slice the list of z-sorted neigbors
  std::vector<slice_ref_set> slice_nbr(slices.size());

  // Forward pass
  for(auto it = z_sort.begin(); it != z_sort.end(); ++it)
    {
    // Add at least the following slice
    auto it_next = it; ++it;
    unsigned int n_added = 0;

    // Now add all the slices in the range
    while(it_next != z_sort.end()
          && (n_added < 1
              || fabs(it->first - it_next->first) < param.z_range))
      {
      slice_nbr[it->second].insert(*it_next);
      n_added++;
      }
    }

  // Forward pass
  for(auto it = z_sort.rbegin(); it != z_sort.rend(); ++it)
    {
    // Add at least the following slice
    auto it_next = it; ++it;
    unsigned int n_added = 0;

    // Now add all the slices in the range
    while(it_next != z_sort.rend()
          && (n_added < 1
              || fabs(it->first - it_next->first) < param.z_range))
      {
      slice_nbr[it->second].insert(*it_next);
      n_added++;
      }
    }

  // Keep a list of loaded images
  typedef LDDMMData<float, 2> LDDMMType;
  std::map<slice_ref, LDDMMType::ImagePointer> loaded_slices;

  // Perform rigid registration between pairs of images. We should do this in a way that
  // the number of images loaded and unloaded is kept to a minimum, without filling memory.
  // The best way to do so would be to progress in z order and release images that are too
  // far behind in z to be included for the current 'reference' image
  for(auto it = z_sort.begin(); it != z_sort.end(); ++it)
    {
    const auto &nbr = slice_nbr[it->second];

    // Prune images no longer required from the loaded slices list
    for(auto it_l = loaded_slices.begin(); it_l != loaded_slices.end(); )
      {
      if(nbr.find(it_l->first) == nbr.end() && it_l->first != *it)
        loaded_slices.erase(it_l++);
      else
        ++it_l;
      }

    // Make sure the reference image itself is loaded
    LDDMMType::ImagePointer i_ref;
    if(loaded_slices.find(*it) == loaded_slices.end())
      {
      LDDMMType::img_read(slices[it->second].raw_filename.c_str(), i_ref);
      loaded_slices[*it] = i_ref;
      }
    else
      {
      i_ref = loaded_slices[*it];
      }

    // Iterate over the neighbor slices
    for(auto it_n = nbr.begin(); it_n != nbr.end(); ++it_n)
      {
      // Load or retrieve the corresponding image
      LDDMMType::ImagePointer i_mov;
      if(loaded_slices.find(*it_n) == loaded_slices.end())
        {
        LDDMMType::img_read(slices[it_n->second].raw_filename.c_str(), i_mov);
        loaded_slices[*it_n] = i_mov;
        }
      else
        {
        i_mov = loaded_slices[*it_n];
        }

      // Perform the registration between i_ref and i_mov
      GreedyApproach<2, float> greedy_api;

      // Make a copy of the template parameters
      GreedyParameters my_param = gparam;

      // Set up the image pair for registration
      ImagePairSpec img_pair;
      img_pair.weight = 1.0;
      img_pair.fixed = slices[it->second].raw_filename;
      img_pair.moving = slices[it_n->second].raw_filename;
      greedy_api.AddCachedInputObject(slices[it->second].raw_filename, i_ref.GetPointer());
      greedy_api.AddCachedInputObject(slices[it_n->second].raw_filename, i_mov.GetPointer());
      my_param.inputs.push_back(img_pair);

      // Set up the output of the affine
      char fn_matrix[1024];
      sprintf(fn_matrix, "%s/affine_ref_%s_mov_%s.mat",
              param.output_dir.c_str(), slices[it->second].unique_id.c_str());
      my_param.output = fn_matrix;

      // Perform affine/rigid
      greedy_api.RunAffine(my_param);
      }
    }
}


