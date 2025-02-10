
#include <iostream>
#include <memory>
#include "PropagationCommon.hxx"
#include "PropagationIO.h"

namespace propagation
{

template<typename TReal>
class PropagationInputBuilder
{
public:
  PROPAGATION_DATA_TYPEDEFS

  PropagationInputBuilder();
  ~PropagationInputBuilder();
  PropagationInputBuilder(const PropagationInputBuilder &other) = delete;
  PropagationInputBuilder &operator=(const PropagationInputBuilder &other) = delete;

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
  unsigned int GetReferenceTimePoint() const { return m_PParam.refTP; };

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


  void SetGreedyParameters(const GreedyParameters &gParam) { m_GParam = gParam; }
  const GreedyParameters &GetConstGreedyParameters() const
  { return m_GParam; }

  void SetPropagationParameters(const PropagationParameters &pParam) { m_PParam = pParam; }
  const PropagationParameters &GetConstPropagationParameters() const
  { return m_PParam; }

  // Parse the parameters, read data, configure this builder and build an input
  void ConfigForCLI(const PropagationParameters &pParam, const GreedyParameters &gParam);

  // Actual method to build the input
  std::shared_ptr<PropagationInput<TReal>> BuildPropagationInput();


private:
  //std::shared_ptr<PropagationInput<TReal>> m_Input;
  GreedyParameters m_GParam;
  PropagationParameters m_PParam;
  std::shared_ptr<PropagationData<TReal>> m_Data;
};

} // namespace propagation

