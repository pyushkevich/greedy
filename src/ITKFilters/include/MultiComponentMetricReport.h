#ifndef MultiComponentMetricReport_H
#define MultiComponentMetricReport_H

#include <vnl/vnl_vector.h>

/**
 * A structure used to communicate metric information. Meant to be an atomic structure
 * that can be expanded with additional bits of information in the future
 */
struct MultiComponentMetricReport
{
  /**
   * Average per-pixel metric added across all components
   */
  double TotalPerPixelMetric;

  /**
   * Average per-pixel metric for all components
   */
  vnl_vector<double> ComponentPerPixelMetrics;

  /**
   * Size of mask over which the metric is integrated and normalized
   */
  double MaskVolume;

  void Scale(double scale_factor)
  {
    TotalPerPixelMetric *= scale_factor;
    ComponentPerPixelMetrics *= scale_factor;
  }
};


#endif // MultiComponentMetricReport_H
