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

}


#endif // PROPAGATIONIO_H
