#ifndef PROPAGATIONIO_H
#define PROPAGATIONIO_H

#include <iostream>
#include <memory>
#include "PropagationCommon.h"
#include "PropagationAPI.h"
#include "PropagationData.h"
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

private:
  std::shared_ptr<PropagationData<TReal>> m_Data;
  GreedyParameters m_Param;

  friend class PropagationInputBuilder<TReal>;
  friend class PropagationAPI<TReal>;
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

  static std::shared_ptr<PropagationInput<TReal>> CreateInputFromParameter(const GreedyParameters &param);

  void Reset();

  std::shared_ptr<PropagationInput<TReal>> Create();

  void SetImage4D(TImage4D *img4d);

  void SetReferenceSegmentationIn3D(TLabelImage3D *seg3d);

  void SetReferenceSegmentationIn4D(TLabelImage4D *seg4d);

  void SetReferenceTimePoint(unsigned int refTP);

  void SetTargetTimePoints(const std::vector<unsigned int> &targetTPs);

  void SetDebugOn(std::string &debug_dir);

  void SetDebugOff();


private:
  std::shared_ptr<GreedyParameters> m_CurrentParam;
};


}


#endif // PROPAGATIONIO_H
