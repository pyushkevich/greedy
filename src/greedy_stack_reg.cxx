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
#include "ShortestPath.h"

#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cerrno>

#include "itkMatrixOffsetTransformBase.h"

#include "lddmm_common.h"
#include "lddmm_data.h"

struct StackParameters
{
  std::string manifest, volume_image, output_dir, image_extension;
  double z_range;
  double z_epsilon;
  bool reuse;

  StackParameters()
    : image_extension("nii.gz"),
      z_range(0.0), z_epsilon(0.1),
      reuse(false) {}

};

struct SliceData
{
  std::string raw_filename;
  std::string unique_id;
  double z_pos;
};

enum FileIntent {
  AFFINE_MATRIX = 0, METRIC_VALUE, ACCUM_MATRIX, ACCUM_RESLICE
};

std::string GetFilenameForSlicePair(
    const StackParameters &param, const SliceData &ref, const SliceData &mov,
    FileIntent intent)
{
  char filename[1024];
  static std::vector<std::string> patterns {
    "%s/affine_ref_%s_mov_%s.mat",
    "%s/affine_ref_%s_mov_%s_metric.txt",
    "%s/accum_affine_root_%s_mov_%s.mat",
    "%s/accum_affine_root_%s_mov_%s_reslice.%s"
  };

  if(intent == ACCUM_RESLICE)
    {
    sprintf(filename, patterns[intent].c_str(),
            param.output_dir.c_str(), ref.unique_id.c_str(), mov.unique_id.c_str(),
            param.image_extension.c_str());
    }
  else
    {
    sprintf(filename, patterns[intent].c_str(),
            param.output_dir.c_str(), ref.unique_id.c_str(), mov.unique_id.c_str());
    }

  return filename;
}



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
  printf("  -M <manifest>      : Manifest of slices to be reconstructed in 3D. Each line of\n");
  printf("                       the manifest file contains the following fields:\n");
  printf("                         unique_id         : a unique identifier, for output dirs\n");
  printf("                         z_position        : a float indicating slice z-position\n");
  printf("                         filename          : path to image filename\n");
  printf("  -o <directory>     : Output directory for matrices/warps\n");
  printf("Additional options: \n");
  printf("  -i <image>             : 3D image to use as target of registration\n");
  printf("  -z <offset> <eps>      : Parameters for the graph-theoretic algorithm. Offset is the\n");
  printf("                           maximum distance in z when slices are considered neighbors.\n");
  printf("                           Epsilon is the constant in eq.(2) from Alder et al. 2014, used\n");
  printf("                           to control how likely slices are to be skipped\n");
  printf("  -N                     : Reuse results from previous runs if found.\n");
  printf("  -ext <extension>       : Extension to use for output image files (without trailing period).)\n");
  printf("                           Default extension is nii.gz.\n");
  printf("Options shared with Greedy: \n");
  printf("  -m metric              : metric (see Greedy docs)\n");
  printf("  -n NxNxN               : number of iterations per level of multi-res (100x100) \n");
  printf("  -threads N             : set the number of allowed concurrent threads\n");
  printf("  -gm-trim <radius>      : generate mask for gradient computation (see Greedy docs)");
  printf("  -search N s_ang s_xyz  : Random search over rigid transforms (see Greedy docs)\n");
  return -1;
}

int main(int argc, char *argv[])
{
  // Parameters specifically for this application
  StackParameters param;

  // Parameters for running Greedy in general
  GreedyParameters gparam;
  GreedyParameters::SetToDefaults(gparam);
  if(argc < 2)
    return usage();

  // Some typedefs
  typedef LDDMMData<double, 2> LDDMMType;
  typedef GreedyApproach<2, double> GreedyAPI;

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

    if(arg == "-M")
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
    else if(arg == "-ext")
      {
      param.image_extension = cl.read_string();
      }
    else if(arg == "-z")
      {
      param.z_range = cl.read_double();
      param.z_epsilon = cl.read_double();
      }
    else if(arg == "-N")
      {
      param.reuse = true;
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

  // Configure the threads
  GreedyAPI::ConfigThreads(gparam);

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

  // We keep for each slice the list of z-sorted neigbors
  std::vector<slice_ref_set> slice_nbr(slices.size());
  unsigned int n_edges = 0;

  // Forward pass
  for(auto it = z_sort.begin(); it != z_sort.end(); ++it)
    {
    // Add at least the following slice
    auto it_next = it; ++it_next;
    unsigned int n_added = 0;

    // Now add all the slices in the range
    while(it_next != z_sort.end()
          && (n_added < 1
              || fabs(it->first - it_next->first) < param.z_range))
      {
      slice_nbr[it->second].insert(*it_next);
      n_added++;
      ++it_next;
      n_edges++;
      }
    }

  // Forward pass
  for(auto it = z_sort.rbegin(); it != z_sort.rend(); ++it)
    {
    // Add at least the following slice
    auto it_next = it; ++it_next;
    unsigned int n_added = 0;

    // Now add all the slices in the range
    while(it_next != z_sort.rend()
          && (n_added < 1
              || fabs(it->first - it_next->first) < param.z_range))
      {
      slice_nbr[it->second].insert(*it_next);
      n_added++;
      ++it_next;
      n_edges++;
      }
    }

  // Keep a list of loaded images
  std::map<slice_ref, LDDMMType::CompositeImagePointer> loaded_slices;

  // At this point we can create a rigid adjacency structure for the graph-theoretic algorithm,
  vnl_vector<unsigned int> G_adjidx(z_sort.size()+1, 0u);
  vnl_vector<unsigned int> G_adj(n_edges, 0u);
  vnl_vector<double> G_edge_weight(n_edges, DijkstraShortestPath<double>::INFINITE_WEIGHT);

  for(unsigned int k = 0, p = 0; k < slices.size(); k++)
    {
    G_adjidx[k+1] = G_adjidx[k] + (unsigned int) slice_nbr[k].size();
    for(auto it : slice_nbr[k])
      G_adj[p++] = it.second;
    }


  // Set up the graph of all registrations. Each slice is a node and edges are between each
  // slice and its closest slices, as well as between each slice and slices in the z-range
  typedef std::tuple<unsigned int, unsigned int, double> GraphEdge;
  std::set<GraphEdge> slice_graph;

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
    LDDMMType::CompositeImagePointer i_ref;
    if(loaded_slices.find(*it) == loaded_slices.end())
      {
      LDDMMType::cimg_read(slices[it->second].raw_filename.c_str(), i_ref);
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
      LDDMMType::CompositeImagePointer i_mov;
      if(loaded_slices.find(*it_n) == loaded_slices.end())
        {
        LDDMMType::cimg_read(slices[it_n->second].raw_filename.c_str(), i_mov);
        loaded_slices[*it_n] = i_mov;
        }
      else
        {
        i_mov = loaded_slices[*it_n];
        }

      // Get the filenames that will be generated by registration
      std::string fn_matrix = GetFilenameForSlicePair(param, slices[it->second], slices[it_n->second], AFFINE_MATRIX);
      std::string fn_metric = GetFilenameForSlicePair(param, slices[it->second], slices[it_n->second], METRIC_VALUE);
      double pair_metric = 1e100;

      // Perform registration or reuse existing registration results
      if(param.reuse && itksys::SystemTools::FileExists(fn_matrix) && itksys::SystemTools::FileExists(fn_metric))
        {
        std::ifstream fin(fn_metric);
        fin >> pair_metric;
        }
      else
        {
        // Perform the registration between i_ref and i_mov
        GreedyAPI greedy_api;

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

        // Set other parameters
        my_param.affine_init_mode = IMG_CENTERS;

        // Set up the output of the affine
        my_param.output = fn_matrix;

        // Perform affine/rigid
        printf("#############################\n");
        printf("### Fixed :%s   Moving %s ###\n", slices[it->second].unique_id.c_str(),slices[it_n->second].unique_id.c_str());
        printf("#############################\n");
        greedy_api.RunAffine(my_param);

        // Get the metric for the affine registration
        pair_metric = greedy_api.GetLastMetricValue();
        std::cout << "Last metric value: " << pair_metric << std::endl;

        // Normalize the metric to give the actual mean NCC
        pair_metric /= -10000.0 * i_ref->GetNumberOfComponentsPerPixel();
        std::ofstream f_metric(fn_metric);
        f_metric << pair_metric << std::endl;
        }

      // Map the metric value into a weight
      double weight = (1.0 - pair_metric) * pow(1 + param.z_epsilon, fabs(it_n->first - it->first));

      // Regardless of whether we did registration or not, record the edge in the graph
      G_edge_weight[G_adjidx[it->second] + (unsigned int) std::distance(nbr.begin(), it_n)] = weight;
      }
    }

  // Run the shortest path computations
  DijkstraShortestPath<double> dijkstra((unsigned int) slices.size(),
                                        G_adjidx.data_block(), G_adj.data_block(), G_edge_weight.data_block());

  // Compute the shortest paths from every slice to the rest and record the total distance. This will
  // help generate the root of the tree
  unsigned int i_root = 0;
  double best_root_dist = 0.0;
  for(unsigned int i = 0; i < slices.size(); i++)
    {
    dijkstra.ComputePathsFromSource(i);
    double root_dist = 0.0;
    for(unsigned int j = 0; j < slices.size(); j++)
      root_dist += dijkstra.GetDistanceArray()[j];
    std::cout << "Root distance " << i << " : " << root_dist << std::endl;
    if(i == 0 || best_root_dist > root_dist)
      {
      i_root = i;
      best_root_dist = root_dist;
      }
    }

  // Compute the composed transformations between the root and each of the inputs
  dijkstra.ComputePathsFromSource(i_root);

  // Load the root image into memory
  LDDMMType::CompositeImagePointer img_root;
  LDDMMType::cimg_read(slices[i_root].raw_filename.c_str(), img_root);

  // Compute transformation for each slice
  for(unsigned int i = 0; i < slices.size(); i++)
    {
    // Initialize the total transform matrix
    vnl_matrix<double> t_accum(3, 3, 0.0);
    t_accum.set_identity();

    // Traverse the path
    unsigned int i_curr = i, i_prev = dijkstra.GetPredecessorArray()[i];
    std::cout << "Chain for " << i << " : ";
    while(i_prev != DijkstraShortestPath<double>::NO_PATH && (i_prev != i_curr))
      {
      // Load the matrix
      std::string fn_matrix =
          GetFilenameForSlicePair(param, slices[i_prev], slices[i_curr], AFFINE_MATRIX);
      vnl_matrix<double> t_step = GreedyAPI::ReadAffineMatrix(TransformSpec(fn_matrix));

      // Accumulate the total transformation
      t_accum = t_accum * t_step;

      std::cout << i_prev << " ";

      // Go to the next edge
      i_curr = i_prev;
      i_prev = dijkstra.GetPredecessorArray()[i_curr];
      }

    std::cout << std::endl;

    // Store the accumulated transform
    std::string fn_accum_matrix =
        GetFilenameForSlicePair(param, slices[i_root], slices[i], ACCUM_MATRIX);

    GreedyAPI::WriteAffineMatrix(fn_accum_matrix, t_accum);

    // Write a resliced image
    std::string fn_accum_reslice =
        GetFilenameForSlicePair(param, slices[i_root], slices[i], ACCUM_RESLICE);

    // Perform the registration between i_ref and i_mov
    GreedyAPI greedy_api;

    // Make a copy of the template parameters
    GreedyParameters my_param = gparam;

    // Set up the image pair for registration
    my_param.reslice_param.ref_image = slices[i_root].raw_filename;
    my_param.reslice_param.images.push_back(ResliceSpec(slices[i].raw_filename, fn_accum_reslice));
    my_param.reslice_param.transforms.push_back(TransformSpec(fn_accum_matrix));
    greedy_api.AddCachedInputObject(slices[i_root].raw_filename, img_root.GetPointer());
    greedy_api.RunReslice(my_param);
    }
}


