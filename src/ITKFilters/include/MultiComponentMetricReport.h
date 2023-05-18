#ifndef MultiComponentMetricReport_H
#define MultiComponentMetricReport_H

#include <vnl/vnl_vector.h>
#include <map>

/**
 * A structure used to communicate metric information. Meant to be an atomic structure
 * that can be expanded with additional bits of information in the future
 */
struct MultiComponentMetricReport
{
  /**
   * Average per-pixel metric added across all components
   */
  double TotalPerPixelMetric = 0.0;

  /**
   * Average per-pixel metric for all components
   */
  vnl_vector<double> ComponentPerPixelMetrics;

  /**
   * Size of mask over which the metric is integrated and normalized
   */
  double MaskVolume = 0.0;

  void Scale(double scale_factor)
  {
    TotalPerPixelMetric *= scale_factor;
    ComponentPerPixelMetrics *= scale_factor;
  }

  void Shift(double per_component_amount)
  {
    TotalPerPixelMetric += per_component_amount * ComponentPerPixelMetrics.size();
    ComponentPerPixelMetrics += per_component_amount;
  }

  void Append(MultiComponentMetricReport &other)
  {
    // Add the scalars
    TotalPerPixelMetric += other.TotalPerPixelMetric;
    MaskVolume += other.MaskVolume;

    // Merge the component vectors
    vnl_vector<double> merged(ComponentPerPixelMetrics.size() +
                              other.ComponentPerPixelMetrics.size());
    merged.update(ComponentPerPixelMetrics, 0);
    merged.update(other.ComponentPerPixelMetrics, ComponentPerPixelMetrics.size());
    ComponentPerPixelMetrics = merged;
  }
};

/**
 * A structure to store regularization information
 */
struct GreedyRegularizationReport
{
  // A pair of weight and value for a regularization term
  typedef std::pair<double, double> WeightValuePair;

  // A dictionary of strings mapped to WeightValuePair
  typedef std::map<std::string, WeightValuePair> TermsMap;

  TermsMap terms;
};


#endif // MultiComponentMetricReport_H
