#ifndef __itkSLICSuperVoxelImageFilter_hxx_
#define __itkSLICSuperVoxelImageFilter_hxx_

#include "itkSLICSuperVoxelImageFilter.h"
#include "itkImageToImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include <algorithm>

namespace itk {


template <typename TInputImage, typename TLabelImage, typename TRealImage>
SLICSuperVoxelImageFilter<TInputImage, TLabelImage, TRealImage>
::SLICSuperVoxelImageFilter()
{
  m_MParameter = 20;
  m_SeedsPerDimension = 20;

  this->SetNumberOfRequiredInputs(2);
}

template <typename TInputImage, typename TLabelImage, typename TRealImage>
void
SLICSuperVoxelImageFilter<TInputImage, TLabelImage, TRealImage>
::SetGradientImage(TRealImage *image)
{
  m_GradientImage = image;
  this->SetNthInput(1, image);
}

template <typename TInputImage, typename TLabelImage, typename TRealImage>
void
SLICSuperVoxelImageFilter<TInputImage, TLabelImage, TRealImage>
::GenerateData()
{
  typedef ImageRegionConstIteratorWithIndex<InputImageType> InputIter;
  typedef ImageRegionIterator<OutputImageType> OutputIter;
  typedef ImageRegionIterator<TRealImage> DistIter;

  // Input and output
  typename OutputImageType::Pointer output = this->GetOutput();
  typename InputImageType::ConstPointer input = this->GetInput();

  // Call AllocateOutputs is required
  this->AllocateOutputs();

  // Allocate the distance image
  m_DistanceImage = TRealImage::New();
  m_DistanceImage->SetRegions(output->GetBufferedRegion());
  m_DistanceImage->CopyInformation(output);
  m_DistanceImage->Allocate();
  m_DistanceImage->FillBuffer(1.0e100);

  // Initialize the label images
  output->FillBuffer(-1);

  // To sample the initial cluster centers, we create a dummy image of cluster
  // centers
  typename TLabelImage::Pointer imgCluster = TLabelImage::New();
  RegionType rgnOutput;
  for(int i = 0; i < ImageDimension; i++)
    {
    rgnOutput.SetSize(i, m_SeedsPerDimension);
    rgnOutput.SetIndex(i, 0);
    }

  imgCluster->SetRegions(rgnOutput);
  imgCluster->SetOrigin(input->GetOrigin());
  imgCluster->SetDirection(input->GetDirection());

  Vector<double,ImageDimension> newspc;

  m_SearchRegionMaxWidth = 0;
  for(int i = 0; i < ImageDimension; i++)
    {
    int dim = input->GetBufferedRegion().GetSize(i);
    double spc = input->GetSpacing()[i];
    double width = dim * spc;
    newspc[i] = width / m_SeedsPerDimension;

    int pixWidth = dim / m_SeedsPerDimension;
    m_SearchRegionOffset[i] = - pixWidth;
    m_SearchRegionSize[i] = 1 + 2 * pixWidth;

    if(m_SearchRegionMaxWidth < pixWidth)
      m_SearchRegionMaxWidth = pixWidth;
    }

  imgCluster->SetSpacing(newspc);
  imgCluster->SetPixelContainer(output->GetPixelContainer());

  // Create a neighborhood iterator for the gradient image search
  typedef itk::ConstNeighborhoodIterator<TRealImage> HoodIter;
  SizeType radius; radius.Fill(1);
  HoodIter itGrad(radius, m_GradientImage, m_GradientImage->GetBufferedRegion());

  // Create the cluster array
  m_Clusters.resize(imgCluster->GetBufferedRegion().GetNumberOfPixels());
  typedef ImageRegionIteratorWithIndex<TLabelImage> OutIter;
  int k = 0;
  for(OutIter it(imgCluster, imgCluster->GetBufferedRegion()); !it.IsAtEnd(); ++it, ++k)
    {
    // Map the cluster center into a voxel position
    itk::ContinuousIndex<double, ImageDimension> cidx;
    for(int i = 0; i < ImageDimension; i++)
      cidx[i] = it.GetIndex()[i] + 0.5;
    IndexType idxInput;
    PointType ptx;
    imgCluster->TransformContinuousIndexToPhysicalPoint(cidx, ptx);
    input->TransformPhysicalPointToIndex(ptx, idxInput);

    // Seach the gradient magnitude image for the
    itGrad.SetLocation(idxInput);

    bool inbounds;
    int jBest = 0;
    RealPixelType gBest = 1e100;

    for(int j = 0; j < itGrad.Size(); j++)
      {
      RealPixelType gval = itGrad.GetPixel(j, inbounds);
      if(inbounds && gval < gBest)
        {
        gBest = gval; jBest = j;
        }
      }

    // Now we have finally an index
    IndexType idxCenter = itGrad.GetIndex(jBest);
    InputPixelType pxCenter = input->GetPixel(idxCenter);
    m_Clusters[k].Index = idxCenter;
    m_Clusters[k].Pixel = pxCenter;
    m_Clusters[k].Count = 1;
    m_Clusters[k].MaxPixelDist = m_MParameter;
    }

  // A mutex for accelerating cluster info
  std::mutex cluster_mutex;

  // Run for some number of iterations
  for(int iter = 0; iter < 10; iter++)
    {
      // Update voxel memberships
      for(int i = 0; i < m_Clusters.size(); i++)
        {
        // The cluster
        Cluster &cluster = m_Clusters[i];

        // Get the search region for this cluster
        RegionType region;
        region.SetSize(this->m_SearchRegionSize);
        region.SetIndex(cluster.Index + this->m_SearchRegionOffset);

        // Check if the region overlaps with the output region
        if(region.Crop(this->GetOutput()->GetBufferedRegion()))
          {
          // Create an iterator for the cropped region
          InputIter itInput(input, region);
          OutputIter itOutput(output, region);
          DistIter itDist(m_DistanceImage, region);

          while(!itInput.IsAtEnd())
            {
            double dist = this->ComputeDistance(itInput.GetIndex(), itInput.Get(), cluster);
            if(itDist.Value() > dist)
              {
              itDist.Set(dist);
              itOutput.Set(i);
              }
            ++itInput; ++itOutput; ++itDist;
            }
          }

        // While we are in this loop, we clear the local cluster information
        cluster.Reset();
        }

      // We have computed the membership of each voxel in our region. Now, we need to update the
      // cluster means.

      // Create an iterator for the cropped region
      InputIter itInput(input, this->GetOutput()->GetBufferedRegion());
      OutputIter itOutput(output, this->GetOutput()->GetBufferedRegion());

      while(!itInput.IsAtEnd())
        {
        int clid = itOutput.Value();
        InputPixelType pix = itInput.Get();
        Cluster &cl = m_Clusters[clid];
        for(int d = 0; d < ImageDimension; d++)
          cl.Index[d] += itInput.GetIndex()[d];

        if(pix > cl.MaxPixel)
          cl.MaxPixel = pix;
        if(pix < cl.MinPixel)
          cl.MinPixel = pix;

        cl.Pixel += pix;
        cl.Count++;

        ++itInput; ++itOutput;
        }

    // Normalize the cluster centers
    for(int i = 0; i < m_Clusters.size(); i++)
      {
      for(int d = 0; d < ImageDimension; d++)
        m_Clusters[i].Index[d] = m_Clusters[i].Index[d] / m_Clusters[i].Count;
      m_Clusters[i].Pixel = m_Clusters[i].Pixel / m_Clusters[i].Count;
      m_Clusters[i].MaxPixelDist = m_Clusters[i].MaxPixel - m_Clusters[i].MinPixel;
      }
    }
}

template <typename TInputImage, typename TLabelImage, typename TRealImage>
double
SLICSuperVoxelImageFilter<TInputImage, TLabelImage, TRealImage>
::ComputeDistance(const IndexType &index, const InputPixelType &pixel, const Cluster &cluster)
{
  // Compute the squared spatial distance
  double ds2 = 0.0;
  for(int d = 0; d < ImageDimension; d++)
    {
    ds2 += (index[d] - cluster.Index[d]) * (index[d] - cluster.Index[d]);
    }

  double dc2 = (pixel - cluster.Pixel) * (pixel - cluster.Pixel);

  return (dc2 * m_SearchRegionMaxWidth * m_SearchRegionMaxWidth 
    + ds2 * m_MParameter * m_MParameter);

}

}

#endif

