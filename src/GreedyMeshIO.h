#ifndef GREEDYMESHIO_H
#define GREEDYMESHIO_H

#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>

class vtkPolyDataReader;
class vtkPolyDataWriter;
class vtkPLYReader;
class vtkPLYWriter;
class vtkSTLReader;
class vtkSTLWriter;
class vtkBYUReader;
class vtkBYUWriter;
class vtkOBJReader;
class vtkOBJWriter;

vtkSmartPointer<vtkPolyData> ReadPolyData(const char *fname);
void WritePolyData(vtkPolyData *mesh, const char *fname);

#endif // GREEDYMESHIO_H
