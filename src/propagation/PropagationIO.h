#ifndef PROPAGATIONIO_H
#define PROPAGATIONIO_H

#include <iostream>
#include <memory>
#include "PropagationCommon.hxx"
#include "PropagationAPI.h"
#include "PropagationData.hxx"
#include "PropagationParameters.hxx"
#include "GreedyParameters.h"


namespace propagation
{
  template<typename TReal>
  class PropagationInputBuilder;
}

namespace propagation
{

template<typename TReal>
class PropagationInput
{
public:
  PropagationInput();
  ~PropagationInput();
  PropagationInput(const PropagationInput &other) = default;
  PropagationInput &operator=(const PropagationInput &other) = default;

  /** Set Greedy Parameters for the input. The setter will also validate input
   *  and populate default values to fields where value are not provided by the user */
  void SetGreedyParameters(const GreedyParameters &gParam);
  const GreedyParameters &GetConstGreedyParameters() const
  { return m_GreedyParam; }

  /** Set Propagation Parameters for the input. The setter will also validate input
   *  and populate default values to fields where value are not provided by the user */
  void SetPropagationParameters(const PropagationParameters &pParam);
  const PropagationParameters &GetConstPropagationParameters() const
  { return m_PropagationParam; }

  void SetOutputDirectory(std::string outdir) { m_Data->outdir = outdir; }

  friend class PropagationInputBuilder<TReal>;
  friend class PropagationAPI<TReal>;

private:
  void SetDefaultGreedyParameters();
  void SetDefaultPropagationParameters();

  void ValidateGreedyParameters() const;
  void ValidatePropagationParameters() const;

  /** For API Runs, image data is directly passed into the Input using the
   *  PropagationInputBuilder*/
  void SetDataForAPIRun(std::shared_ptr<PropagationData<TReal>> data);

  std::shared_ptr<PropagationData<TReal>> m_Data;
  GreedyParameters m_GreedyParam;
  PropagationParameters m_PropagationParam;
};

template<typename TReal>
class PropagationOutput
{
public:
  PROPAGATION_DATA_TYPEDEFS

  PropagationOutput();
  ~PropagationOutput();
  PropagationOutput(const PropagationOutput &other) = default;
  PropagationOutput &operator=(const PropagationOutput &other) = default;

  void Initialize(std::shared_ptr<PropagationData<TReal>> data);
  bool IsInitialized() const
  { return m_Data != nullptr; }

  typedef std::map<unsigned int, TPropagationMeshPointer> TMeshSeries;
  typedef std::map<std::string, TMeshSeries> TMeshSeriesMap;
  typedef std::map<unsigned int, typename TLabelImage3D::Pointer> TSegmentation3DSeries;

  /** Image Getter */
  typename TImage3D::Pointer GetImage3D(unsigned int tp);

  /** Segmentation Getters */
  typename TLabelImage4D::Pointer GetSegmentation4D();
  typename TLabelImage3D::Pointer GetSegmentation3D(unsigned int tp);
  TSegmentation3DSeries GetSegmentation3DSeries();

  /** Segmentation Mesh Getters*/
  TMeshSeries GetMeshSeries();
  TPropagationMeshPointer GetMesh(unsigned int tp);

  /** Extra Mesh Getters */
  TMeshSeries GetExtraMeshSeries(std::string tag);
  TPropagationMeshPointer GetExtraMesh(std::string tag, unsigned int tp);
  TMeshSeriesMap GetAllExtraMeshSeries();

  size_t GetNumberOfTimePoints();
  std::vector<unsigned int> GetTimePointList();

private:
  std::shared_ptr<PropagationData<TReal>> m_Data;
};

template<typename TReal>
class PropagationInputBuilder
{
public:
  PROPAGATION_DATA_TYPEDEFS

  PropagationInputBuilder();
  ~PropagationInputBuilder();
  PropagationInputBuilder(const PropagationInputBuilder &other) = delete;
  PropagationInputBuilder &operator=(const PropagationInputBuilder &other) = delete;

  static std::shared_ptr<PropagationInput<TReal>> BuildInputForCommandLineRun(
      const PropagationParameters &pParam, const GreedyParameters &gParam);

  void Reset();

  //--------------------------------------------------
  // Propagation Parameters Configuration

  /** Set Reference 4D Image */
  void SetImage4D(TImage4D *img4d);

  /** Set 3D Segmentation Image for the reference time point.*/
  void SetReferenceSegmentationIn3D(TLabelImage3D *seg3d);

  /** Set 4D Segmentation Image with reference time segmentation. This will override
   *  3D segmentation image input */
  void SetReferenceSegmentationIn4D(TLabelImage4D *seg4d);

  /** Set reference time point */
  void SetReferenceTimePoint(unsigned int refTP);
  unsigned int GetReferenceTimePoint() const;

  /** Set target time point list. Reference time point is ignored */
  void SetTargetTimePoints(const std::vector<unsigned int> &targetTPs);

  /** Set Reslice Metric */
  void SetResliceMetric(const InterpSpec metric);
  void SetResliceMetricToLinear();
  void SetResliceMetricToNearestNeighbor();
  void SetResliceMetricToLabel(double sigma, bool is_physical_unit);
  InterpSpec GetResliceMetric() const;

  /** Add Extra Mesh to Warp */
  void AddExtraMeshToWarp(std::string fnmesh, std::string outpattern);
  void AddExtraMeshToWarp(TPropagationMeshPointer mesh, std::string tag);
  std::vector<MeshSpec> GetExtraMeshesToWarp() const;

  /** Set Propagation Verbosity */
  void SetPropagationVerbosity(PropagationParameters::Verbosity v);
  PropagationParameters::Verbosity GetPropagationVerbosity() const;

  /** Turn on the debug mode (-sp-debug) for propagation. A debugging output directory is needed for
   *  dumping out intermediary files */
  void SetDebugOn(std::string &debug_dir);

  /** Turn off the debug mode for propagation */
  void SetDebugOff();

  //--------------------------------------------------
  // Greedy Parameters Configuration

  /** Set the metric (-m) for greedy run */
  void SetRegistrationMetric(GreedyParameters::MetricType metric, std::vector<int> metric_radius = std::vector<int>());
  GreedyParameters::MetricType GetRegistrationMetric() const;
  std::vector<int> GetRegistrationMetricRadius() const;

  /** Set multi-resolution schedule (-n) default is 100x100 */
  void SetMultiResolutionSchedule(std::vector<int> iter_per_level);
  std::vector<int> GetMultiResolutionSchedule() const;

  /** Set Affine degree of freedom */
  void SetAffineDOF(GreedyParameters::AffineDOF dof);
  GreedyParameters::AffineDOF GetAffineDOF() const;

  /** Set Verbose Level */
  void SetGreedyVerbosity(GreedyParameters::Verbosity v);
  GreedyParameters::Verbosity GetGreedyVerbosity() const;

  /** Build a propagation input object to pass to the Propagation API */
  std::shared_ptr<PropagationInput<TReal>> BuildPropagationInput();


  void SetGreedyParameters(const GreedyParameters &gParam) { m_GParam = gParam; }
  const GreedyParameters &GetConstGreedyParameters() const
  { return m_GParam; }

  void SetPropagationParameters(const PropagationParameters &pParam) { m_PParam = pParam; }
  const PropagationParameters &GetConstPropagationParameters() const
  { return m_PParam; }


private:
  //std::shared_ptr<PropagationInput<TReal>> m_Input;
  GreedyParameters m_GParam;
  PropagationParameters m_PParam;
  std::shared_ptr<PropagationData<TReal>> m_Data;
};


}


#endif // PROPAGATIONIO_H
