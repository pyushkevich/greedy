
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
#ifndef GREEDYAPI_H
#define GREEDYAPI_H

#include "GreedyParameters.h"
#include "GreedyException.h"
#include "MultiComponentMetricReport.h"
#include "lddmm_data.h"
#include "AffineCostFunctions.h"
#include <vnl/vnl_random.h>
#include <map>
#include "itkCommand.h"
#include <vtkSmartPointer.h>
#include <vtkObject.h>

template <typename T, unsigned int V> class MultiImageOpticalFlowHelper;

namespace itk {
  template <typename T, unsigned int D1, unsigned int D2> class MatrixOffsetTransformBase;

}

class vtkPointSet;

/**
 * This is the top level class for the greedy software. It contains methods
 * for deformable and affine registration.
 */
template <unsigned int VDim, typename TReal = double>
class GreedyApproach
{
public:

  typedef GreedyApproach<VDim, TReal> Self;

  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef typename LDDMMType::ImageBaseType ImageBaseType;
  typedef typename LDDMMType::ImageType ImageType;
  typedef typename LDDMMType::ImagePointer ImagePointer;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::VectorImagePointer VectorImagePointer;
  typedef typename LDDMMType::CompositeImageType CompositeImageType;
  typedef typename LDDMMType::CompositeImagePointer CompositeImagePointer;

  typedef vnl_vector_fixed<TReal, VDim> VecFx;
  typedef vnl_matrix_fixed<TReal, VDim, VDim> MatFx;

  typedef std::vector< std::vector<MultiComponentMetricReport> > MetricLogType;
  typedef std::vector< std::vector<GreedyRegularizationReport> > RegularizationLogType;

  typedef MultiImageOpticalFlowHelper<TReal, VDim> OFHelperType;

  typedef itk::MatrixOffsetTransformBase<TReal, VDim, VDim> LinearTransformType;
  typedef itk::MatrixOffsetTransformBase<double, VDim, VDim> LinearTransformIOType;

  struct ImagePair {
    ImagePointer fixed, moving;
    VectorImagePointer grad_moving;
    double weight;
  };

  // Mesh data structures
  typedef vtkPointSet MeshType;
  typedef vtkSmartPointer<MeshType> MeshPointer;
  typedef std::vector<MeshPointer> MeshArray;

  static void ConfigThreads(const GreedyParameters &param);

  int Run(GreedyParameters &param);

  int RunCommandLine(const char *cmd);

  int RunDeformable(GreedyParameters &param);

  int RunDeformableOptimization(GreedyParameters &param);

  int RunAffine(GreedyParameters &param);

  int RunBrute(GreedyParameters &param);

  int RunReslice(GreedyParameters &param);

  int RunInvertWarp(GreedyParameters &param);

  int RunRootWarp(GreedyParameters &param);

  int RunAlignMoments(GreedyParameters &param);

  int RunJacobian(GreedyParameters &param);
  
  int RunMetric(GreedyParameters &param);

  int ComputeMetric(GreedyParameters &param, MultiComponentMetricReport &metric_report);

  /**
   * Add an image that is already in memory to the internal cache, and
   * associate it with a filename. This provides a way for images already
   * loaded in memory to be passed in to the Greedy API while using the
   * standard parameter structures.
   *
   * Normally, images such as the fixed image are passed as part of the
   * GreedyParameters object as filenames. For example, we might set
   *
   *   param.inputs[0].fixed = "/tmp/goo.nii.gz";
   *
   * However, if we are linking to the greedy API from another program and
   * already have the fixed image in memory, we can use the cache mechanism
   * instead.
   *
   *   greedyapi.AddCachedInputObject("FIXED-0", myimage);
   *   param.inputs[0].fixed = "FIXED-0";
   *
   * The API will check the cache before loading the image. The type of the
   * object in the cache must match the type of the object expected internally,
   * which is VectorImage for most images. If not, an exception will be
   * thrown.
   *
   */
  void AddCachedInputObject(std::string key, itk::Object *object);
  void AddCachedInputObject(std::string key, vtkObject *object);

  /**
   * Add an image/matrix to the output cache. This has the same behavior as
   * the input cache, but there is an additional flag as to whether you want
   * to save the output object to the specified filename in addition to writing
   * it to the cached image/matrix. This allows you to both store the result in
   * the cache and write it to a filename specified in the key
   *
   * The cached output object can be a null pointer, in which case the object
   * will be allocated. It can then me accessed using GetCachedObject()
   */
  void AddCachedOutputObject(std::string key, itk::Object *object, bool force_write = false);
  void AddCachedOutputObject(std::string key, vtkObject *object, bool force_write = false);

  /**
   * Get a cached object by name
   */
  itk::Object *GetCachedObject(std::string key);

  /**
   * Get the list of all cached objects (input and output)
   */
  std::vector<std::string> GetCachedObjectNames() const;

  /**
   * Get the metric log - values of metric per level. Can be called from
   * callback functions and observers
   */
  const MetricLogType &GetMetricLog() const;

  /** Get the last value of the metric recorded */
  MultiComponentMetricReport GetLastMetricReport() const;

  vnl_matrix<double> ReadAffineMatrixViaCache(const TransformSpec &ts);

  void WriteAffineMatrixViaCache(const std::string &filename, const vnl_matrix<double> &Qp);

  static vnl_matrix<double> ReadAffineMatrix(const TransformSpec &ts);

  /**
   * Helper method to read an affine matrix from file into an ITK transform type
   */
  static void ReadAffineTransform(const TransformSpec &ts, LinearTransformType *tran);

  static void WriteAffineMatrix(const std::string &filename, const vnl_matrix<double> &Qp);

  /**
   * Helper method to write an affine ITK transform type to a matrix file
   */
  static void WriteAffineTransform(const std::string &filename, LinearTransformType *tran);

  static vnl_matrix<double> MapAffineToPhysicalRASSpace(
      OFHelperType &of_helper, unsigned int group, unsigned int level,
      LinearTransformType *tran);

  static void MapPhysicalRASSpaceToAffine(
      OFHelperType &of_helper, unsigned int group, unsigned int level,
      vnl_matrix<double> &Qp,
      LinearTransformType *tran);

  static void MapRASAffineToPhysicalWarp(const vnl_matrix<double> &mat,
                                         VectorImagePointer &out_warp);

  template <class TAffineTransform>
  static vnl_matrix<double> MapITKTransformToRASMatrix(
    const TAffineTransform *tran);

  template <class TAffineTransform>
  static void MapRASMatrixToITKTransform(
    const vnl_matrix<double> &mat, TAffineTransform *tran);

  void RecordMetricValue(const MultiComponentMetricReport &metric);

  // Helper method to print iteration reports
  std::string PrintIter(int level, int iter,
                        const MultiComponentMetricReport &metric,
                        const GreedyRegularizationReport &reg) const;

  /**
   * Read images specified in parameters into a helper data structure and initialize
   * the multi-resolution pyramid
   */
  void ReadImages(GreedyParameters &param, OFHelperType &ofhelper,
                  bool force_resample_to_fixed_space);

  /**
   * Compute one of the metrics (specified in the parameters). This code is called by
   * RunDeformable and is provided as a separate public method for testing purposes
   *
   * The minimization mode makes sure that that regardless of the metric, the objective
   * value returned should be minimized (i.e., NCC is mapped to 1-NCC, MI to 2 - MI, etc)
   * and that the metric gradient is scaled appropriately as well, so that for any variation
   * v, the dot product <grad, v> is equal to the directional derivative of TotalPerPixelMetric
   * with respect to v. The value eps is ignored in minimization mode.
   */
  void EvaluateMetricForDeformableRegistration(
      GreedyParameters &param, OFHelperType &of_helper, unsigned int level,
      VectorImageType *phi, MultiComponentMetricReport &metric_report,
      ImageType *out_metric_image, VectorImageType *out_metric_gradient,
      double eps, bool minimization_mode = false);

  /**
   * Load initial transform (affine or deformable) into a deformation field
   */
  void LoadInitialTransform(
      GreedyParameters &param, OFHelperType &of_helper,
      unsigned int level, VectorImageType *phi);


  /**
   * Generate an affine cost function for given level based on parameter values
   */
  AbstractAffineCostFunction<VDim, TReal> *CreateAffineCostFunction(
      GreedyParameters &param, OFHelperType &of_helper, int level);

  /**
   * Initialize affine transform (to identity, filename, etc.) based on the
   * parameter values; resulting transform is placed into tLevel.
   */
  void InitializeAffineTransform(
      GreedyParameters &param, OFHelperType &of_helper,
      AbstractAffineCostFunction<VDim, TReal> *acf,
      LinearTransformType *tLevel);

  /**
   * Check the derivatives of affine transform
   */
  int CheckAffineDerivatives(GreedyParameters &param, OFHelperType &of_helper,
                             AbstractAffineCostFunction<VDim, TReal> *acf,
                             LinearTransformType *tLevel, int level, double tol);

  /** Apply affine transformation to a mesh */
  static void TransformMeshAffine(vtkPointSet *mesh, vnl_matrix<double> mat);

  /** Apply warp to a mesh */
  static void TransformMeshWarp(vtkPointSet *mesh, VectorImageType *warp);

  // Read a chain of transforms into a single warp, optionally applying to a set of meshes
  void ReadTransformChain(const std::vector<TransformSpec> &tran_chain,
                          ImageBaseType *ref_space,
                          VectorImagePointer &out_warp,
                          MeshArray *meshes = nullptr);


protected:

  struct CacheEntry {
    typename itk::Object::Pointer target;
    bool force_write;
  };

  struct VTKCacheEntry {
    vtkObject *target;
    bool force_write;
  };

  typedef std::map<std::string, CacheEntry> ImageCache;
  ImageCache m_ImageCache;

  typedef std::map<std::string, VTKCacheEntry> MeshCache;
  MeshCache m_MeshCache;

  // A log of metric values used during registration - so metric can be looked up
  // in the callbacks to RunAffine, etc.
  MetricLogType m_MetricLog;

  // Also a log of regularization values
  RegularizationLogType m_RegularizationLog;

  // This function reads the image from disk, or from a memory location mapped to a
  // string. The first approach is used by the command-line interface, and the second
  // approach is used by the API, allowing images to be passed from other software.
  // An optional second argument is used to store the component type, but only if
  // the image is actually loaded from disk. For cached images, the component type
  // will be unknown.
  template <class TImage>
  itk::SmartPointer<TImage> ReadImageViaCache(const std::string &filename,
                                              itk::IOComponentEnum *comp_type = NULL);

  MeshPointer ReadMeshViaCache(const std::string &filename);

  template<class TObject> TObject *CheckCache(const std::string &filename) const;

  // Get a filename for dumping intermediate outputs
  std::string GetDumpFile(const GreedyParameters &param, const char *pattern, ...);

  // This function reads an image base object via cache. It is more permissive than using
  // ReadImageViaCache.
  typename ImageBaseType::Pointer ReadImageBaseViaCache(const std::string &filename);


  // Write an image using the cache
  template <class TImage>
  void WriteImageViaCache(TImage *img, const std::string &filename,
                          itk::IOComponentEnum comp = itk::IOComponentEnum::UNKNOWNCOMPONENTTYPE);

  // Write a compressed warp via cache (in float format)
  void WriteCompressedWarpInPhysicalSpaceViaCache(
    ImageBaseType *moving_ref_space, VectorImageType *warp, const char *filename, double precision);

  void WriteMeshViaCache(MeshType *mesh, const std::string &filename);

  // Compute the moments of a composite image (mean and covariance matrix of coordinate weighted by intensity)
  void ComputeImageMoments(CompositeImageType *image, const vnl_vector<float> &weights, VecFx &m1, MatFx &m2);

  // Resample an image to reference space if the spaces do not match or if an explicit warp is provided
  CompositeImagePointer ResampleImageToReferenceSpaceIfNeeded(
      CompositeImageType *img, ImageBaseType *ref_space, VectorImageType *resample_warp, TReal fill_value);

  ImagePointer ResampleMaskToReferenceSpaceIfNeeded(
      ImageType *mask, ImageBaseType *ref_space, VectorImageType *resample_warp);

  // friend class PureAffineCostFunction<VDim, TReal>;

};

// Print greedy usage
int greedy_usage(bool print_template_params = true);

// Parse greedy command line
GreedyParameters greedy_parse_commandline(CommandLineHelper &cl, bool parse_template_params);

#endif // GREEDYAPI_H
