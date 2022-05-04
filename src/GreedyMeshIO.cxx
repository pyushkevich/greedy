#include "GreedyMeshIO.h"
#include "GreedyException.h"

#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkPLYReader.h>
#include <vtkPLYWriter.h>
#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>
#include <vtkBYUReader.h>
#include <vtkBYUWriter.h>
#include <vtkOBJReader.h>

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
    return ReadMesh<vtkPolyDataReader, TMesh>(fname);
  else
    throw GreedyException("No mesh reader for file %s", fname);
}

template<class TMesh>
void WriteMeshByExtension(vtkPolyData *mesh, const char *fname)
{
  std::string fn_str = fname;
  if(fn_str.rfind(".byu") == fn_str.length() - 4)
    WriteMesh<vtkBYUWriter, TMesh>(mesh, fname);
  else if(fn_str.rfind(".stl") == fn_str.length() - 4)
    WriteMesh<vtkSTLWriter, TMesh>(mesh, fname);
  else if(fn_str.rfind(".ply") == fn_str.length() - 4)
    WriteMesh<vtkPLYWriter, TMesh>(mesh, fname);
  else if(fn_str.rfind(".vtk") == fn_str.length() - 4)
    WriteMesh<vtkPolyDataWriter, TMesh>(mesh, fname);
  else
    throw GreedyException("No mesh writer for file %s", fname);
}

} // namespace

vtkSmartPointer<vtkPolyData> ReadPolyData(const char *fname)
{
  return greedy_mesh_io::ReadMeshByExtension<vtkPolyData>(fname);
}

void WritePolyData(vtkPolyData *mesh, const char *fname)
{
  greedy_mesh_io::WriteMeshByExtension<vtkPolyData>(mesh, fname);
}
