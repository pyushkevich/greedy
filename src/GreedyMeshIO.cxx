#include "GreedyMeshIO.h"
#include "GreedyException.h"

#include <vtkGenericDataObjectReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPLYReader.h>
#include <vtkPLYWriter.h>
#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>
#include <vtkBYUReader.h>
#include <vtkBYUWriter.h>
#include <vtkOBJReader.h>
#include <vtkTetra.h>
#include <vtkTriangle.h>
#include <vtkDoubleArray.h>
#include <vtkCellData.h>

namespace greedy_mesh_io {

template <class TReader, class TMesh>
vtkSmartPointer<TMesh> ReadMesh(const char *fname)
{
  vtkSmartPointer<TReader> reader = TReader::New();
  reader->SetFileName(fname);
  reader->Update();
  return reader->GetOutput();
};

template <class TWriter>
void ConfigureWriter(TWriter *w, const char *fname)
{
  w->SetFileName(fname);
}

template <>
void ConfigureWriter(vtkBYUWriter *w, const char *fname)
{
  w->SetGeometryFileName(fname);
}

template <class TWriter, class TMesh>
void WriteMesh(TMesh *mesh, const char *fname)
{
  vtkSmartPointer<TWriter> writer = TWriter::New();
  ConfigureWriter<TWriter>(writer, fname);
  writer->SetInputData(mesh);
  writer->Update();
};

template<class TMesh>
vtkSmartPointer<TMesh> ReadMeshByExtension(const char *fname)
{
  std::string fn_str = fname;
  if(fn_str.rfind(".byu") == fn_str.length() - 4)
    return ReadMesh<vtkBYUReader, TMesh>(fname);
  else if(fn_str.rfind(".obj") == fn_str.length() - 4)
    return ReadMesh<vtkOBJReader, TMesh>(fname);
  else if(fn_str.rfind(".stl") == fn_str.length() - 4)
    return ReadMesh<vtkSTLReader, TMesh>(fname);
  else if(fn_str.rfind(".ply") == fn_str.length() - 4)
    return ReadMesh<vtkPLYReader, TMesh>(fname);
  else if(fn_str.rfind(".vtk") == fn_str.length() - 4)
    {
    vtkSmartPointer<vtkGenericDataObjectReader> reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
    reader->SetFileName(fname);
    reader->Update();
    if(reader->IsFilePolyData())
      return reader->GetPolyDataOutput();
    else if (reader->IsFileUnstructuredGrid())
      return reader->GetUnstructuredGridOutput();
    else
      throw GreedyException("No mesh reader for file %s", fname);
    }
  else
    throw GreedyException("No mesh reader for file %s", fname);
}

template<class TMesh>
void WriteMeshByExtension(TMesh *mesh, const char *fname)
{
  std::string fn_str = fname;
  if(fn_str.rfind(".byu") == fn_str.length() - 4)
    WriteMesh<vtkBYUWriter, TMesh>(mesh, fname);
  else if(fn_str.rfind(".stl") == fn_str.length() - 4)
    WriteMesh<vtkSTLWriter, TMesh>(mesh, fname);
  else if(fn_str.rfind(".ply") == fn_str.length() - 4)
    WriteMesh<vtkPLYWriter, TMesh>(mesh, fname);
  else if(fn_str.rfind(".vtk") == fn_str.length() - 4)
    {
    vtkPolyData *pd = dynamic_cast<vtkPolyData *>(mesh);
    vtkUnstructuredGrid *usg = dynamic_cast<vtkUnstructuredGrid *>(mesh);
    if(pd)
      WriteMesh<vtkPolyDataWriter, vtkPolyData>(pd, fname);
    else if (usg)
      WriteMesh<vtkUnstructuredGridWriter, vtkUnstructuredGrid>(usg, fname);
    }
  else
    throw GreedyException("No mesh writer for file %s", fname);
}

} // namespace

vtkSmartPointer<vtkPointSet> ReadMesh(const char *fname)
{
  return greedy_mesh_io::ReadMeshByExtension<vtkPointSet>(fname);
}

void WriteMesh(vtkPointSet *mesh, const char *fname)
{
  greedy_mesh_io::WriteMeshByExtension<vtkPointSet>(mesh, fname);
}

double GetCellVolumeOrArea(vtkPointSet *mesh, vtkCell *cell)
{
  if(cell->GetCellType() == VTK_TETRA)
    {
    double p[4][3];
    mesh->GetPoint(cell->GetPointId(0), p[0]);
    mesh->GetPoint(cell->GetPointId(1), p[1]);
    mesh->GetPoint(cell->GetPointId(2), p[2]);
    mesh->GetPoint(cell->GetPointId(3), p[3]);
    return vtkTetra::ComputeVolume(p[0], p[1], p[2], p[3]);
    }
  else if(cell->GetCellType() == VTK_TRIANGLE)
    {
    return dynamic_cast<vtkTriangle *>(cell)->ComputeArea();
    }
  else
    return 0.0;
}

void WriteJacobianMesh(vtkPointSet *fixed_mesh, vtkPointSet *moving_mesh, const char *fname)
{
  // Create a point array to store the Jacobian
  vtkSmartPointer<vtkPointSet> jac_mesh = DeepCopyMesh(fixed_mesh);

  vtkSmartPointer<vtkDoubleArray> jac_array = vtkNew<vtkDoubleArray>();
  jac_array->SetNumberOfComponents(1);
  jac_array->SetNumberOfTuples(fixed_mesh->GetNumberOfCells());
  jac_array->SetName("jacobian");

  for(vtkIdType i = 0; i < fixed_mesh->GetNumberOfCells(); i++)
    {
    vtkCell *c_fixed = fixed_mesh->GetCell(i);
    double v_fixed = GetCellVolumeOrArea(fixed_mesh, c_fixed);
    double v_moved = GetCellVolumeOrArea(moving_mesh, c_fixed);
    double jacobian = v_moved / v_fixed;
    jac_array->SetTuple1(i, jacobian);
    }

  jac_mesh->GetCellData()->AddArray(jac_array);
  WriteMesh(jac_mesh, fname);
}

vtkSmartPointer<vtkPointSet> DeepCopyMesh(vtkPointSet *mesh)
{
  vtkPolyData *pd = dynamic_cast<vtkPolyData *>(mesh);
  if(pd)
    {
    vtkSmartPointer<vtkPolyData> pd_copy = vtkNew<vtkPolyData>();
    pd_copy->DeepCopy(pd);
    return pd_copy;
    }
  vtkUnstructuredGrid *usg = dynamic_cast<vtkUnstructuredGrid *>(mesh);
  if(usg)
    {
    vtkSmartPointer<vtkUnstructuredGrid> usg_copy = vtkNew<vtkUnstructuredGrid>();
    usg_copy->DeepCopy(usg);
    return usg_copy;
    }

  return nullptr;
}
