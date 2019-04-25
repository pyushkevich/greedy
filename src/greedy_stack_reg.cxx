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
#include <numeric>
#include <cerrno>

#include "itkMatrixOffsetTransformBase.h"
#include "itkImageAlgorithm.h"
#include "itkZeroFluxNeumannPadImageFilter.h"
#include "itkImageFileReader.h"

#include "lddmm_common.h"
#include "lddmm_data.h"

struct StackParameters
{
  std::string manifest, volume_image, output_dir, image_extension;
  double z_range;
  double z_epsilon;
  bool reuse;

  unsigned int iter_affine, iter_deformable;

  StackParameters()
    : image_extension("nii.gz"),
      z_range(0.0), z_epsilon(0.1),
      reuse(false), iter_affine(5), iter_deformable(5) {}

};

struct SliceData
{
  std::string raw_filename;
  std::string unique_id;
  double z_pos;
};

enum FileIntent {
  AFFINE_MATRIX = 0, METRIC_VALUE, ACCUM_MATRIX, ACCUM_RESLICE,
  VOL_INIT_MATRIX, VOL_SLIDE, VOL_MEDIAN_INIT_MATRIX,
  VOL_ITER_MATRIX, VOL_ITER_WARP, ITER_METRIC_DUMP
};

std::string GetFilenameForSlicePair(
    const StackParameters &param, const SliceData &ref, const SliceData &mov,
    FileIntent intent)
{
  char filename[1024];
  const char *dir = param.output_dir.c_str(), *ext = param.image_extension.c_str();
  const char *rid = ref.unique_id.c_str(), *mid = mov.unique_id.c_str();

  switch(intent)
    {
    case AFFINE_MATRIX:
      sprintf(filename, "%s/affine_ref_%s_mov_%s.mat", dir, rid, mid);
      break;
    case METRIC_VALUE:
      sprintf(filename, "%s/affine_ref_%s_mov_%s_metric.txt", dir, rid, mid);
      break;
    case ACCUM_MATRIX:
      sprintf(filename, "%s/accum_affine_root_%s_mov_%s.mat", dir, rid, mid);
      break;
    case ACCUM_RESLICE:
      sprintf(filename, "%s/accum_affine_root_%s_mov_%s_reslice.%s", dir, rid, mid, ext);
      break;
    default:
      throw GreedyException("Wrong intent in GetFilenameForSlicePair");
    }

  return filename;
}

std::string GetFilenameForSlice(
    const StackParameters &param, const SliceData &slice, int intent, ...)
{
  char filename[1024];
  const char *dir = param.output_dir.c_str(), *ext = param.image_extension.c_str();
  const char *sid = slice.unique_id.c_str();

  va_list args;
  va_start(args, intent);

  switch(intent)
    {
    case VOL_INIT_MATRIX:
      sprintf(filename, "%s/affine_refvol_mov_%s.mat", dir, sid);
      break;
    case VOL_SLIDE:
      sprintf(filename, "%s/vol_slide_%s.%s", dir, sid, ext);
      break;
    case VOL_ITER_MATRIX:
      sprintf(filename, "%s/affine_refvol_mov_%s_iter%02d.mat", dir, sid, va_arg(args, int));
      break;
    case VOL_ITER_WARP:
      sprintf(filename, "%s/warp_refvol_mov_%s_iter%02d.%s", dir, sid, va_arg(args, int), ext);
      break;
    case ITER_METRIC_DUMP:
      sprintf(filename, "%s/metric_refvol_mov_%s_iter%02d.txt", dir, sid, va_arg(args, int));
      break;
    default:
      throw GreedyException("Wrong intent in GetFilenameForSlice");
    }

  va_end(args);

  return filename;
}

std::string GetFilenameForGlobal(
    const StackParameters &param, FileIntent intent)
{
  char filename[1024];
  const char *dir = param.output_dir.c_str(), *ext = param.image_extension.c_str();

  switch(intent)
    {
    case VOL_MEDIAN_INIT_MATRIX:
      sprintf(filename, "%s/affine_refvol_median.mat", dir);
      break;
    default:
      throw GreedyException("Wrong intent in GetFilenameForGlobal");
    }

  return filename;
}

/**
 * This class represents a reference to an image that may exist on disk, or may be
 * stored in memory. There is a limit on the amount of memory that can be used by
 * all the image refs, and images are rotated in and out of memory based on when
 * they were last accessed
 */
class ImageCache
{
public:

  ImageCache(unsigned long max_memory = 0l, unsigned int max_images = 0)
    : m_MaxMemory(max_memory), m_MaxImages(max_images), m_Counter(0l) {}

  template <typename TImage> typename TImage::Pointer GetImage(const std::string &filename)
  {
    // Check the cache for the image
    auto it = m_Cache.find(filename);
    if(it != m_Cache.end())
      {
      TImage *image = dynamic_cast<TImage *>(std::get<2>(it->second).GetPointer());
      if(!image)
        throw GreedyException("Type mismatch in image cache");
      typename TImage::Pointer image_ptr = image;
      return image_ptr;
      }

    // Image does not exist in cache, load it
    typedef itk::ImageFileReader<TImage> ReaderType;
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(filename.c_str());
    reader->Update();
    typename TImage::Pointer image_ptr = reader->GetOutput();

    // Get the size of the image in bytes
    unsigned long img_size = image_ptr->GetPixelContainer()->Size()
                             * sizeof (typename TImage::PixelContainer::Element);

    // If the size of the image is too large, we need to reduce the size of the cache
    this->ShrinkCache(img_size, 1);

    // Add the new image
    m_Cache[filename] = std::make_tuple(m_Counter++, img_size, image_ptr);
    m_UsedMemory += img_size;

    // Return the image
    return image_ptr;
  }

  void ShrinkCache(unsigned long new_bytes, unsigned int new_images)
  {
    // Remove something from the cache until it's not empty and the constraints of the
    // cache are satisfied
    while(IsCacheFull(new_bytes, new_images) && m_Cache.size() > 0)
      {
      // Find the oldest entry in the cache
      std::map<unsigned long, std::string> sort_map;
      for(auto it : m_Cache)
        sort_map[std::get<0>(it.second)] = it.first;

      // Remove the first (oldest) entry
      auto it_erase = m_Cache.find(sort_map.begin()->second);
      m_UsedMemory -= std::get<1>(it_erase->second);
      m_Cache.erase(it_erase);
      }
  }

  bool IsCacheFull(unsigned long new_bytes, unsigned int new_images)
  {
    if(m_MaxMemory > 0 && m_UsedMemory + new_bytes > m_MaxMemory)
      return true;

    if(m_MaxImages > 0 && m_Cache.size() + new_images > m_MaxImages)
      return true;

    return false;
  }

  void PurgeCache()
  {
    m_Cache.clear();
    m_UsedMemory = 0;
  }

protected:

  // Cache entry (age, size, pointer)
  typedef std::tuple<unsigned long, unsigned long, itk::Object::Pointer> CacheEntry;
  typedef std::map<std::string, CacheEntry> CacheType;

  // Cache for images
  CacheType m_Cache;

  unsigned long m_MaxMemory, m_UsedMemory;
  unsigned int m_MaxImages;
  unsigned long m_Counter;
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

/** Helper methods for stack registration */
class GreedyStackHelper
{
public:
  typedef LDDMMData<double, 2> LDDMMType;
  typedef GreedyApproach<2, double> GreedyAPI;
  typedef LDDMMType::CompositeImageType SlideImageType;
  typedef LDDMMType::CompositeImagePointer SlideImagePointer;

  typedef LDDMMData<double, 3> LDDMMType3D;
  typedef LDDMMType3D::CompositeImageType VolumeImage;
  typedef LDDMMType3D::CompositeImagePointer VolumePointer;

  static SlideImagePointer ExtractSliceFromVolume(VolumePointer vol, double z_pos)
  {
    VolumePointer vol_slice = LDDMMType3D::CompositeImageType::New();
    typename LDDMMType3D::RegionType reg_slice = vol->GetBufferedRegion();
    reg_slice.GetModifiableSize()[2] = 1;
    vol_slice->CopyInformation(vol);
    vol_slice->SetRegions(reg_slice);
    vol_slice->Allocate();

    // Adjust the origin of the slice
    auto origin_slice = vol_slice->GetOrigin();
    origin_slice[2] = z_pos;
    vol_slice->SetOrigin(origin_slice);

    // Generate a blank deformation field
    LDDMMType3D::VectorImagePointer zero_warp = LDDMMType3D::new_vimg(vol_slice);

    // Sample the slice from the volume
    LDDMMType3D::interp_cimg(vol, zero_warp, vol_slice, false, true, 0.0);

    // Now drop the dimension of the slice to 2D
    LDDMMType::RegionType reg_slice_2d;
    LDDMMType::CompositeImageType::PointType origin_2d;
    LDDMMType::CompositeImageType::SpacingType spacing_2d;
    LDDMMType::CompositeImageType::DirectionType dir_2d;

    for(unsigned int a = 0; a < 2; a++)
      {
      reg_slice_2d.SetIndex(a, reg_slice.GetIndex(a));
      reg_slice_2d.SetSize(a, reg_slice.GetSize(a));
      origin_2d[a] = vol_slice->GetOrigin()[a];
      spacing_2d[a] = vol_slice->GetSpacing()[a];
      dir_2d(a,0) = vol_slice->GetDirection()(a,0);
      dir_2d(a,1) = vol_slice->GetDirection()(a,1);
      }

    SlideImagePointer vol_slice_2d = SlideImageType::New();
    vol_slice_2d->SetRegions(reg_slice_2d);
    vol_slice_2d->SetOrigin(origin_2d);
    vol_slice_2d->SetDirection(dir_2d);
    vol_slice_2d->SetSpacing(spacing_2d);
    vol_slice_2d->SetNumberOfComponentsPerPixel(vol_slice->GetNumberOfComponentsPerPixel());
    vol_slice_2d->Allocate();

    // Copy data between the pixel containers
    itk::ImageAlgorithm::Copy(vol_slice.GetPointer(), vol_slice_2d.GetPointer(),
                              vol_slice->GetBufferedRegion(), vol_slice_2d->GetBufferedRegion());

    return vol_slice_2d;
  }
};

/**
 * This method extracts a slice from a volumetric image for specified z-index
 */



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
  typedef LDDMMType::CompositeImageType SlideImageType;
  typedef LDDMMType::CompositeImagePointer SlideImagePointer;

  // List of greedy commands that are recognized by this code
  std::set<std::string> greedy_cmd {
    "-m", "-n", "-threads", "-gm-trim", "-search"
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

  // Set up a cache for loaded images. These images can be cycled in and out of memory
  // depending on need. TODO: let user configure cache sizes
  ImageCache slice_cache(0, 20);

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

    // Read the reference slide from the cache
    SlideImagePointer i_ref =
        slice_cache.GetImage<SlideImageType>(slices[it->second].raw_filename);

    // Iterate over the neighbor slices
    for(auto it_n = nbr.begin(); it_n != nbr.end(); ++it_n)
      {
      // Load or retrieve the corresponding image
      SlideImagePointer i_mov =
          slice_cache.GetImage<SlideImageType>(slices[it_n->second].raw_filename);

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
        ImagePairSpec img_pair(slices[it->second].raw_filename, slices[it_n->second].raw_filename);
        greedy_api.AddCachedInputObject(slices[it->second].raw_filename, i_ref.GetPointer());
        greedy_api.AddCachedInputObject(slices[it_n->second].raw_filename, i_mov.GetPointer());
        my_param.inputs.push_back(img_pair);

        // Set other parameters
        my_param.affine_dof = GreedyParameters::DOF_RIGID;
        my_param.affine_init_mode = IMG_CENTERS;

        // Set up the output of the affine
        my_param.output = fn_matrix;

        // Perform affine/rigid
        printf("#############################\n");
        printf("### Fixed :%s   Moving %s ###\n", slices[it->second].unique_id.c_str(),slices[it_n->second].unique_id.c_str());
        printf("#############################\n");
        greedy_api.RunAffine(my_param);

        // Get the metric for the affine registration
        pair_metric = greedy_api.GetLastMetricReport().TotalMetric;
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
  LDDMMType::ImagePointer img_root;
  LDDMMType::img_read(slices[i_root].raw_filename.c_str(), img_root);

  // Apply some padding to the root image.
  typedef itk::ZeroFluxNeumannPadImageFilter<LDDMMType::ImageType, LDDMMType::ImageType> PadFilter;
  PadFilter::Pointer fltPad = PadFilter::New();
  fltPad->SetInput(img_root);

  // Determine the amount of padding to add
  unsigned int max_dim =
      std::max(img_root->GetBufferedRegion().GetSize()[0],img_root->GetBufferedRegion().GetSize()[1]);
  itk::Size<2> pad_size;
  pad_size.Fill(max_dim / 4);
  fltPad->SetPadBound(pad_size);
  fltPad->Update();

  // Store the result
  LDDMMType::ImagePointer img_root_padded = fltPad->GetOutput();

  // The padded image has a non-zero index, which causes problems downstream for GreedyAPI.
  // To account for this, we save and load the image
  // TODO: handle this internally using a filter!
  LDDMMType::img_write(img_root_padded, "/tmp/padded.nii.gz");
  img_root_padded = LDDMMType::img_read("/tmp/padded.nii.gz");

  // Also read the 3D volume into memory if we have it
  typedef LDDMMData<double, 3> LDDMMType3D;
  LDDMMType3D::CompositeImagePointer vol;
  if(param.volume_image.length())
    {
    LDDMMType3D::cimg_read(param.volume_image.c_str(), vol);
    }

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

    // Hold the resliced image in memory
    LDDMMType::CompositeImagePointer img_reslice = LDDMMType::CompositeImageType::New();

    // Only do reslice if necessary
    if(!param.reuse || !itksys::SystemTools::FileExists(fn_accum_reslice))
      {
      // Perform the registration between i_ref and i_mov
      GreedyAPI greedy_api;

      // Make a copy of the template parameters
      GreedyParameters my_param = gparam;

      // Set up the image pair for registration
      my_param.reslice_param.ref_image = "root_slice_padded";
      my_param.reslice_param.images.push_back(ResliceSpec(slices[i].raw_filename, fn_accum_reslice));
      my_param.reslice_param.transforms.push_back(TransformSpec(fn_accum_matrix));
      greedy_api.AddCachedInputObject("root_slice_padded", img_root_padded.GetPointer());
      greedy_api.AddCachedInputObject(slices[i].raw_filename,
                                      slice_cache.GetImage<SlideImageType>(slices[i].raw_filename));
      greedy_api.AddCachedOutputObject(fn_accum_reslice, img_reslice.GetPointer(), true);
      greedy_api.RunReslice(my_param);
      }
    else
      {
      // Just read the resliced image
      img_reslice = LDDMMType::cimg_read(fn_accum_reslice.c_str());
      }

    // If there is a 3D image volume, do that registration
    if(vol)
      {
      // Filename for the volume slice corresponding to current slide
      std::string fn_vol_slide = GetFilenameForSlice(param, slices[i], VOL_SLIDE);

      // Output matrix for this registration
      std::string fn_vol_init_matrix = GetFilenameForSlice(param, slices[i], VOL_INIT_MATRIX);

      if(!param.reuse || !itksys::SystemTools::FileExists(fn_vol_slide)
          || !itksys::SystemTools::FileExists(fn_vol_init_matrix))
        {
        // Extract the slice from the 3D image
        SlideImagePointer vol_slice_2d =
            GreedyStackHelper::ExtractSliceFromVolume(vol, slices[i].z_pos);

        // Write the 2d slice
        LDDMMType::cimg_write(vol_slice_2d, fn_vol_slide.c_str());

        // Try registration between resliced slide and corresponding volume slice with
        // a brute force search. This will be used to create a median transformation
        // between slide space and volume space. Since the volume may come with a mask,
        // we use volume slice as fixed, and the slide image as moving
        GreedyAPI greedy_api;

        // TODO: we need separate parameters for the multi-modality registrations!
        GreedyParameters my_param = gparam;

        // Set up the image pair for registration
        // TODO: have moving image in memory
        ImagePairSpec img_pair("vol_slice", "resliced_slide", 1.0);
        greedy_api.AddCachedInputObject("vol_slice", vol_slice_2d.GetPointer());
        greedy_api.AddCachedInputObject("resliced_slide", img_reslice.GetPointer());
        my_param.inputs.push_back(img_pair);

        // Set other parameters
        my_param.affine_dof = GreedyParameters::DOF_AFFINE;
        my_param.affine_init_mode = IMG_CENTERS;

        // Set up the output of the affine
        my_param.output = fn_vol_init_matrix;

        // Run the affine registration
        greedy_api.RunAffine(my_param);
        }
      }
    }

  // If volume supplied, we can not compute the median affine transformation to the
  // volume, which we will be able to use to initialize all iterative registrations
  if(vol)
    {
    // List of affine matrices to the volume slice
    typedef vnl_matrix_fixed<double, 3, 3> Mat3;
    std::vector<Mat3> vol_affine(slices.size());
    for(unsigned int i = 0; i < slices.size(); i++)
      {
      std::string fn_vol_init_matrix = GetFilenameForSlice(param, slices[i], VOL_INIT_MATRIX);
      vol_affine[i] = GreedyAPI::ReadAffineMatrix(TransformSpec(fn_vol_init_matrix));
      }

    // Compute distances between all pairs of affine matrices
    vnl_matrix<double> aff_dist(slices.size(), slices.size()); aff_dist.fill(0.0);
    for(unsigned int i = 0; i < slices.size(); i++)
      {
      for(unsigned int j = 0; j < i; j++)
        {
        aff_dist(i,j) = (vol_affine[i] - vol_affine[j]).array_one_norm();
        aff_dist(j,i) = aff_dist(i,j);
        }
      }

    // Compute the sum of distances from each matrix to the rest
    vnl_vector<double> row_sums = aff_dist * vnl_vector<double>(slices.size(), 1.0);

    // Find the index of the smallest element
    unsigned int idx_best =
        std::find(row_sums.begin(), row_sums.end(), row_sums.min_value()) -
        row_sums.begin();

    // The median affine
    Mat3 median_affine = vol_affine[idx_best];

    // Write the median affine to a file
    GreedyAPI::WriteAffineMatrix(GetFilenameForGlobal(param, VOL_MEDIAN_INIT_MATRIX), median_affine);

    // Now write the complete initial to-volume transform for each slide
    for(unsigned int i = 0; i < slices.size(); i++)
      {
      vnl_matrix<double> M_root =
          GreedyAPI::ReadAffineMatrix(
            GetFilenameForSlicePair(param, slices[i_root], slices[i], ACCUM_MATRIX));

      vnl_matrix<double> M_vol = M_root * median_affine;

      GreedyAPI::WriteAffineMatrix(
            GetFilenameForSlice(param, slices[i], VOL_ITER_MATRIX, 0), M_vol);
      }
    }


  // Now that we have the affine initialization from the histology space to the volume space, we can
  // perform iterative optimization, where each slice is matched to its neighbors and to the
  // corresponding MRI slice. The only issue here is how do we want to use the graph in this
  // process: we don't want the bad neighbors to pull the registration away from the good
  // solution. On the other hand, we can expect the bad slices to eventually auto-correct. It seems
  // that the proper approach would be to down-weigh certain slices by their metric, but then
  // again, do we want to do this based on initial metric or current metric. For now, we can start
  // by just using same weights.

  // This is an iterative process, so we need to keep track of the iteration.
  for(unsigned int iter = 1; iter <= param.iter_affine + param.iter_deformable; iter++)
    {
    // Randomly shuffle the order in which slices are considered
    std::vector<unsigned int> ordering(slices.size());
    std::iota(ordering.begin(), ordering.end(), 0);
    std::random_shuffle(ordering.begin(), ordering.end());

    // Keep track of the total neighbor metric and total volume metric
    double total_to_nbr_metric = 0.0;
    double total_to_vol_metric = 0.0;

    // Iterate over the ordering
    for(unsigned int k : ordering)
      {
      // The output filename for this affine registration
      std::string fn_result =
          iter <= param.iter_affine
          ? GetFilenameForSlice(param, slices[k], VOL_ITER_MATRIX, iter)
          : GetFilenameForSlice(param, slices[k], VOL_ITER_WARP, iter);

      // Has this already been done? Then on to the next!
      if(param.reuse && itksys::SystemTools::FileExists(fn_result))
        continue;

      // Get the pointer to the current slide (used as moving image)
      SlideImagePointer img_slide = slice_cache.GetImage<SlideImageType>(slices[k].raw_filename);

      // Get the corresponding slice from the 3D volume
      SlideImagePointer vol_slice_2d =
          GreedyStackHelper::ExtractSliceFromVolume(vol, slices[k].z_pos);

      // Set up the registration. We are registering to the volume and to the transformed
      // adjacent slices. We should do everything in the space of the MRI volume because
      // (a) it should be large enough to cover the histology and (b) there might be a mask
      // in this space, while we cannot expect there to be a mask in the other space.

      // Find the adjacent slices. TODO: there is all kinds of stuff that could be done here,
      // like allowing a z-range for adjacent slices registration, modulating weight by the
      // distance, and detecting and down-weighting 'bad' slices. For now just pick the slices
      // immediately below and above the current slice
      auto itf = z_sort.find(std::make_pair(slices[k].z_pos, k));
      if(itf == z_sort.end())
        throw GreedyException("Slice not found in sorted list (%d, z = %f)", k, slices[k].z_pos);

      // Go backward and forward one slice
      slice_ref_set k_nbr;
      auto itf_back = itf, itf_fore = itf;
      if(itf != z_sort.begin())
        k_nbr.insert(*(--itf_back));
      if((++itf_fore) != z_sort.end())
        k_nbr.insert(*itf_fore);

      // Create the greedy API for the main registration task
      GreedyAPI api_reg;
      api_reg.AddCachedInputObject("moving", img_slide);
      api_reg.AddCachedInputObject("volume_slice", vol_slice_2d);

      // We need to hold on to the resliced image pointers, because otherwise they will be deallocated
      std::vector<SlideImagePointer> resliced_neighbors(slices.size());

      // TODO: erase!
      LDDMMType::cimg_write(vol_slice_2d, "/tmp/fix.nii.gz");
      LDDMMType::cimg_write(img_slide, "/tmp/mov.nii.gz");

      // Set up the main registration pair
      GreedyParameters param_reg = gparam;
      // param_reg.inputs.push_back(ImagePairSpec("volume_slice", "moving", 1.0));
      param_reg.inputs.push_back(ImagePairSpec("/tmp/fix.nii.gz", "/tmp/mov.nii.gz", 4.0));



      // Handle each of the neighbors
      for(auto nbr : k_nbr)
        {
        unsigned int j = nbr.second;

        // Create an image pointer for the reslicing output
        resliced_neighbors[j] = SlideImageType::New();

        // Each of the neighbor slices needs to be resliced using last iteration's transform. We
        // could cache these images, but then again, it does not take so much to do this on the
        // fly. For now we will do this on the fly.
        GreedyAPI api_reslice;
        api_reslice.AddCachedInputObject("vol_slice", vol_slice_2d);
        api_reslice.AddCachedOutputObject("output", resliced_neighbors[j], false);

        GreedyParameters param_reslice = gparam;
        param_reslice.reslice_param.ref_image = "vol_slice";
        param_reslice.reslice_param.images.push_back(ResliceSpec(slices[j].raw_filename, "output"));

        // Was the previous iteration a deformable iteration? If so, apply the warp
        if(iter - 1 <= param.iter_affine)
          {
          param_reslice.reslice_param.transforms.push_back(
                TransformSpec(GetFilenameForSlice(param, slices[j], VOL_ITER_MATRIX, iter-1)));
          }
        else
          {
          param_reslice.reslice_param.transforms.push_back(
                TransformSpec(GetFilenameForSlice(param, slices[j], VOL_ITER_WARP, iter-1)));
          param_reslice.reslice_param.transforms.push_back(
                TransformSpec(GetFilenameForSlice(param, slices[j], VOL_ITER_MATRIX, param.iter_affine)));
          }

        // Perform the reslicing
        api_reslice.RunReslice(param_reslice);

        // Add the image pair to the registration
        char fixed_fn[64];
        sprintf(fixed_fn, "neighbor_%03d", j);
        api_reg.AddCachedInputObject(fixed_fn, resliced_neighbors[j]);


        char fff[256];
        sprintf(fff, "/tmp/nr%03d.nii.gz", j);
        LDDMMType::cimg_write(resliced_neighbors[j], fff);

        // param_reg.inputs.push_back(ImagePairSpec(fixed_fn, "moving", 1.0));
        param_reg.inputs.push_back(ImagePairSpec(fff, "/tmp/mov.nii.gz", 1.0));

        }

      printf("#############################\n");
      printf("### Iter :%d   Slide %s ###\n", iter, slices[k].unique_id.c_str());
      printf("#############################\n");

      // What kind of registration are we doing at this iteration?
      if(iter <= param.iter_affine)
        {
        // Specify the DOF, etc
        param_reg.affine_dof = GreedyParameters::DOF_AFFINE;
        param_reg.affine_init_mode = RAS_FILENAME;
        param_reg.affine_init_transform =
            TransformSpec(GetFilenameForSlice(param, slices[k], VOL_ITER_MATRIX, iter-1));
        param_reg.rigid_search = RigidSearchSpec();

        // Specify the output
        std::string fn_result = GetFilenameForSlice(param, slices[k], VOL_ITER_MATRIX, iter);
        param_reg.output = fn_result;

        // Run this registration!
        api_reg.RunAffine(param_reg);
        }
      else
        {
        // Apply the last affine transformation
        param_reg.moving_pre_transforms.push_back(
              TransformSpec(GetFilenameForSlice(param, slices[k], VOL_ITER_MATRIX,param.iter_affine)));

        // Specify the output
        param_reg.output = fn_result;
        param_reg.affine_init_mode = VOX_IDENTITY;


        param_reg.sigma_pre.sigma = 10.0;
        param_reg.sigma_pre.physical_units = false;

        param_reg.sigma_post.sigma = 2.0;
        param_reg.sigma_post.physical_units = false;

        // Run the registration
        api_reg.RunDeformable(param_reg);
        }

      MultiComponentMetricReport last_metric_report = api_reg.GetLastMetricReport();
      total_to_vol_metric += last_metric_report.ComponentMetrics[0];
      for(unsigned int a = 1; a < last_metric_report.ComponentMetrics.size(); a++)
        total_to_nbr_metric += last_metric_report.ComponentMetrics[a];

      // Write the metric for this slide to file
      std::string fn_metric = GetFilenameForSlice(param, slices[k], ITER_METRIC_DUMP, iter);
      std::ofstream fout(fn_metric);
      fout << api_reg.PrintIter(-1, -1, last_metric_report) << std::endl;
      }

    printf("ITER %3d  TOTAL_VOL_METRIC = %8.4f  TOTAL_NBR_METRIC = %8.4f\n",
           iter, total_to_vol_metric, total_to_nbr_metric);
    }

  // The following step is to perform rigid/affine registration of the 3D volume
  // relative to the 2D slices. The unknown here is a 3D matrix, but it must be applied
  // at each 2D slice. There is also a possibility to repeat this alignment from
  // iteration to iteration.

  // There are multiple ways to match a 3D volume to the 2D slices. These include:
  //   1. When we know the slice-to-slice matching (e.g. based on blockface images),
  //      we only need to perform 2D registrations. Specifically, we need to find
  //      the initial alignment of the volume to 2D slices before running more local
  //      per-slice alignments
  //   2. When we need to allow the 3D volume to rotate in 3D, we can either build a
  //      3D volume from the histology slices (assuming regular spacing) and perform
  //      regular 3D registration; or we can code up a hybrid 2D/3D method where each
  //      slice is registered to the 3D volume.

  // For now, we will focus on (1). The first thing we need to do is to figure out how
  // to sample slices from the 3D volume that correspond to individual slices.

  // As a first step, we can attempt a brute force search between each histology slice
  // and each MRI slice. Then we can find the median of these transformations, or some
  // other robust combination.

}


