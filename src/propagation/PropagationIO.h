#ifndef PROPAGATIONIO_H
#define PROPAGATIONIO_H

#include <iostream>
#include <memory>
#include "PropagationCommon.h"
#include "PropagationAPI.h"
#include "PropagationData.h"
#include "PropagationParameters.h"
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
  PROPAGATION_DATA_TYPEDEFS

  PropagationInput();
  ~PropagationInput();
  PropagationInput(const PropagationInput &other) = default;
  PropagationInput &operator=(const PropagationInput &other) = default;

  void SetGreedyParameters(const GreedyParameters &gParam);
  const GreedyParameters &GetConstGreedyParameters() const
  { return m_GreedyParam; }

  void SetPropagationParameters(const PropagationParameters &pParam);
  const PropagationParameters &GetConstPropagationParameters() const
  { return m_PropagationParam; }

  friend class PropagationInputBuilder<TReal>;
  friend class PropagationAPI<TReal>;

private:
  void SetDefaultGreedyParameters();
  void SetDefaultPropagationParameters();

  void ValidateGreedyParameters() const;
  void ValidatePropagationParameters() const;

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

  typename TLabelImage4D::Pointer GetSegmentation4D();
  typename TLabelImage3D::Pointer GetSegmentation3D(unsigned int tp);
  size_t GetNumberOfTimePoints();
  std::vector<unsigned int> GetTimePointList();

private:
  typename TLabelImage4D::Pointer m_Segmentation4D;
  std::map<unsigned int, typename TLabelImage3D::Pointer> m_Segmentation3DMap;
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

  static std::shared_ptr<PropagationInput<TReal>> CreateInputForCommandLineRun(
      const PropagationParameters &pParam, const GreedyParameters &gParam);

  void Reset();

  std::shared_ptr<PropagationInput<TReal>> Create();

  /** Set Reference 4D Image */
  void SetImage4D(TImage4D *img4d);

  /** Set 3D Segmentation Image for the reference time point.*/
  void SetReferenceSegmentationIn3D(TLabelImage3D *seg3d);

  /** Set 4D Segmentation Image with reference time segmentation. This will override
   *  3D segmentation image input */
  void SetReferenceSegmentationIn4D(TLabelImage4D *seg4d);

  /** Set reference time point */
  void SetReferenceTimePoint(unsigned int refTP);

  /** Set target time point list. Reference time point is ignored */
  void SetTargetTimePoints(const std::vector<unsigned int> &targetTPs);

  // General greedy setting
  /** Get current greedy parameters for advanced configuration */
  GreedyParameters GetGreedyParameters()
  { return m_Input->m_GreedyParam; }

  /** Set the metric for greedy run */
  void SetMetric(GreedyParameters::MetricType metric);

  /** Set metric radius if needed */
  void SetMetricRadius(std::vector<int> metric_radius);

  /** Set multi-resolution schedule (-n) default is 100x100 */
  void SetMultiResolutionSchedule(std::vector<int> iter_per_level);

  /** Turn on the debug mode for propagation. A debugging output directory is needed for
   *  dumping out intermediary files */
  void SetDebugOn(std::string &debug_dir);

  /** Turn off the debug mode for propagation */
  void SetDebugOff();

  /** Get propagation input object to pass to the Propagation API */
  std::shared_ptr<PropagationInput<TReal>> GetPropagationInput()
  { return m_Input; }


private:
  std::shared_ptr<PropagationInput<TReal>> m_Input;
};


}


#endif // PROPAGATIONIO_H
