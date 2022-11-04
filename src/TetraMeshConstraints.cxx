#include "TetraMeshConstraints.h"
#include <vtkUnstructuredGrid.h>
#include "GreedyException.h"

TetraMeshConstraints::TetraMeshConstraints()
{

}

template<class TFloat, unsigned int VDim>
void TetraMeshConstraints<TFloat, VDim>
::SetMesh(vtkUnstructuredGrid *mesh)
{
  m_MeshVTK = mesh;

  // Allocate and assign the vertex coordinates
  m_TetraX.set_size(mesh->GetNumberOfPoints(), VDim);
  for(unsigned int i = 0; i < mesh->GetNumberOfPoints(); i++)
    {
    double *x = mesh->GetPoint(i)
    for(unsigned int j = 0; j < VDim; j++)
      m_TetraX(i,j) = x[j];
    }

  // Allocate and assign the tetrahedral vertex indices
  m_TetraVI.set_size(mesh->GetNumberOfCells(), VDim + 1);
  for(unsigned int i = 0; i < m_TetraVI->GetNumberOfCells(); i++)
    {
    auto *cell = mesh->GetCell(i);
    if(cell->GetNumberOfPoints() != VDim + 1)
      throw GreedyException("Mesh has cells of incorrect dimension");

    for(unsigned int j = 0; j < VDim+1; j++)
      m_TetraVI(i,j) = cell->GetPointId(j);

    // Find neighbors across each edge
    vtkIdType edge[VDim];
    for(unsigned int j = 0; j < VDim+1; j++)
      {
      vtkIdType *p_edge = edge;
      vtkIdList nbr;
      for(unsigned int k = 0; k < VDim+1; k++)
        if(k != j)
          *p_edge++ = m_TetraVI(i,k)
      mesh->GetCellNeighbors(i, VDim, edge, &nbr);
      }


}
