#ifndef PROPAGATIONDATA_HXX
#define PROPAGATIONDATA_HXX

#include <map>
#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <vtkSmartPointer.h>
#include <itkMatrixOffsetTransformBase.h>
#include <vtkPolyData.h>
#include "lddmm_data.h"
#include "PropagationAPI.h"
#include "PropagationCommon.hxx"

namespace propagation
{

template<typename TReal>
class PropagationMeshGroup
{
  PROPAGATION_DATA_TYPEDEFS
};

template<typename TReal>
class TimePointTransformSpec
{
public:
  PROPAGATION_DATA_TYPEDEFS
  TimePointTransformSpec(typename TTransform::Pointer _affine,
                         typename TVectorImage3D::Pointer _deform, unsigned int _crntTP)
    : affine(_affine), deform (_deform), currentTP(_crntTP) {}

  TimePointTransformSpec(const TimePointTransformSpec &other) = default;
  TimePointTransformSpec & operator=(const TimePointTransformSpec &other) = default;

  unsigned int currentTP;
  typename TTransform::Pointer affine;
  typename TVectorImage3D::Pointer deform;
};

template<typename TReal>
class TimePointData
{
public:
  PROPAGATION_DATA_TYPEDEFS
  TimePointData();
  ~TimePointData();
  TimePointData(const TimePointData &other) = default;
  TimePointData &operator=(const TimePointData &other) = default;

  typename TImage3D::Pointer img;
  typename TImage3D::Pointer img_srs;
  typename TLabelImage3D::Pointer seg;
  typename TLabelImage3D::Pointer seg_srs;
  typename TLabelImage3D::Pointer full_res_mask;
  typename TTransform::Pointer affine_to_prev;
  typename TVectorImage3D::Pointer deform_to_prev;
  typename TVectorImage3D::Pointer deform_to_ref;
  typename TVectorImage3D::Pointer deform_from_prev;
  typename TVectorImage3D::Pointer deform_from_ref;
  TPropagationMeshPointer seg_mesh; // mesh warped from reference tp

  std::vector<TimePointTransformSpec<TReal>> transform_specs;
  std::vector<TimePointTransformSpec<TReal>> full_res_label_trans_specs;

  void AddExtraMesh(std::string tag, TPropagationMeshPointer mesh)
  {
    m_ExtraMeshes.insert({tag, mesh});
  }

  TPropagationMeshPointer GetExtraMesh(std::string &tag)
  {
    TPropagationMeshPointer ret = nullptr;
    if (m_ExtraMeshes.count(tag))
      ret = m_ExtraMeshes[tag];

    return ret;
  }

  std::vector<std::string> GetExtraMeshTags()
  {
    std::vector<std::string> ret;

    for (auto kv : m_ExtraMeshes)
      ret.push_back(kv.first);

    return ret;
  }

  size_t GetExtraMeshSize() { return m_ExtraMeshes.size(); }

protected:
  // warped extra meshes, indexed by tags
  std::map<std::string, TPropagationMeshPointer> m_ExtraMeshes;
};

template<typename TReal>
class PropagationData
{
public:
  PROPAGATION_DATA_TYPEDEFS
  PropagationData();

  size_t GetNumberOfTimePoints();

  std::map<unsigned int, TimePointData<TReal>> tp_data;
  typename TImage4D::Pointer img4d;
  typename TLabelImage3D::Pointer seg_ref;
  typename TLabelImage4D::Pointer seg4d_in;
  typename TLabelImage4D::Pointer seg4d_out;
  std::string outdir;
  typename TImage3D::Pointer full_res_ref_space;

  // extra meshes for warping, indexed by tags
  std::map<std::string, TPropagationMeshPointer> extra_mesh_cache;
};

template<typename TReal>
PropagationData<TReal>
::PropagationData()
{

}

template <typename TReal>
size_t
PropagationData<TReal>
::GetNumberOfTimePoints()
{
  return this->img4d->GetBufferedRegion().GetSize()[3];
}

template<typename TReal>
TimePointData<TReal>
::TimePointData()
{
  affine_to_prev = TTransform::New();
}

template<typename TReal>
TimePointData<TReal>
::~TimePointData()
{

}

} // end of namespace propagation

#endif // PROPAGATIONDATA_HXX
