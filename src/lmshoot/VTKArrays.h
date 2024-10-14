#ifndef VTKARRAYS_H
#define VTKARRAYS_H

#include <vtkFieldData.h>
#include <vtkIntArray.h>
#include <vtkDoubleArray.h>
#include <vtkDataSet.h>

template <typename TVal, typename = void>
struct VTKArrayTypeSelector
{
    using type = void; // Default case
};

template <typename TVal>
struct VTKArrayTypeSelector<TVal, std::enable_if_t<std::is_integral_v<TVal>>>
{
    using type = vtkIntArray; // Default case
};

template <typename TVal>
struct VTKArrayTypeSelector<TVal, std::enable_if_t<std::is_floating_point_v<TVal>>>
{
    using type = vtkDoubleArray; // Default case
};

template<class TVal>
void vtk_set_scalar_field_data(vtkDataSet *pd, const char *name, TVal value)
{
  using ArrayType = typename VTKArrayTypeSelector<TVal>::type;
  vtkNew<ArrayType> array;
  array->SetNumberOfComponents(1);
  array->SetName(name);
  array->InsertNextValue((int) value);
  pd->GetFieldData()->AddArray(array);
}

template<class TVal>
TVal vtk_get_scalar_field_data(vtkDataSet *pd, const char *name)
{
  using ArrayType = typename VTKArrayTypeSelector<TVal>::type;
  auto *arr = pd->GetFieldData()->GetArray(name);
  if(arr && arr->GetNumberOfComponents() == 1 && arr->GetNumberOfTuples() == 1)
    return (TVal) arr->GetTuple1(0);
  else
    throw std::out_of_range(name);
}

template<class TVal>
TVal vtk_get_scalar_field_data(vtkDataSet *pd, const char *name, TVal deflt_value)
{
  using ArrayType = typename VTKArrayTypeSelector<TVal>::type;
  auto *arr = pd->GetFieldData()->GetArray(name);
  if(arr && arr->GetNumberOfComponents() == 1 && arr->GetNumberOfTuples() == 1)
    return (TVal) arr->GetTuple1(0);
  else
    return deflt_value;
}


#endif // VTKARRAYS_H
