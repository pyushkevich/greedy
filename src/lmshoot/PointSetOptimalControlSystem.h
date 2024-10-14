#ifndef __PointSetOptimalControlSystem_h_
#define __PointSetOptimalControlSystem_h_

#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_vector_fixed.h>
#include <vector>

namespace ctpl { class thread_pool; }

template <class TFloat, unsigned int VDim>
class PointSetOptimalControlSystem
{
public:

  typedef vnl_matrix<TFloat> Matrix;
  typedef vnl_vector<TFloat> Vector;

  typedef vnl_vector_fixed<TFloat, VDim> VecD;

  typedef std::vector<Matrix> MatrixArray;

  /**
   * Constructor - set basic parameters of the system and allocate
   * all of the necessary matrices
   * 
   * q0      : N x D vector of template landmark positions
   * sigma   : standard deviation of the Gaussian kernel 
   * N       : number of timesteps for the ODE
   */
  PointSetOptimalControlSystem(
    const Matrix &q0, 
    TFloat sigma,
    unsigned int N);

  /** 
   * Get the number of time steps
   */
  unsigned int GetN() const { return N; }

  /**
   * Compute the kinetic energy and the flow velocities at a given timepoint
   */
  TFloat ComputeEnergyAndVelocity(const Matrix &q, const Matrix &u);

  /**
   * Perform forward flow with control u, returning the total kinetic energy
   * of the flow. The endpoint and intermediate timepoints can be queried
   * using GetQt()
   */
  TFloat Flow(const MatrixArray &u);

  /**
   * Perform backward flow in order to compute the gradient of some function
   * f with respect to the control u. The input is the array of partial derivatives
   * of the function f with respect to the path q(t).
   *
   * The function also includes the partial derivatives of the kinetic energy
   * with respect to u, with weight w_kinetic 
   */
  void FlowBackward(const MatrixArray &u, const MatrixArray &d_f__d_qt, 
                    TFloat w_kinetic, MatrixArray &d_f__d_u);

  /**
   * Get the curves
   */
  const Matrix &GetQt(unsigned int t) const { return Qt[t]; }

  /** Get the velocities */
  const Matrix &GetVt(unsigned int t) const { return Vt[t]; }

  const TFloat GetDeltaT() const { return dt; }

protected:

  // Step of backpropagation
  void PropagateAlphaBackwards(
    const Matrix &q, const Matrix &u, 
    const Vector alpha[], Vector alpha_Q[], Vector alpha_U[]);

  // Initial ladnmark coordinates - fixed for duration
  Matrix q0;

  // Standard deviation of Gaussian kernel; time step
  TFloat sigma, dt;

  // Number of timesteps for integration; number of points
  unsigned int N, k;

  // The current velocity vector
  Vector d_q__d_t[VDim];

  // Streamlines - paths of the landmarks over time
  std::vector<Matrix> Qt;

  // Streamline velocities
  std::vector<Matrix> Vt;

    // Multi-threaded quantities
  struct ThreadData 
    {
    // List of rows handled by this thread
    std::vector<unsigned int> rows;
    TFloat KE;
    Vector d_q__d_t[VDim];
    Vector alpha_U[VDim], alpha_Q[VDim];
    };

  // Data associated with each thread
  std::vector<ThreadData> td;

  // A thread pool to handle execution
  ctpl::thread_pool *thread_pool;

  void SetupMultiThreaded();

  void ComputeEnergyAndVelocityThreadedWorker(
    const Matrix &q, const Matrix &u, ThreadData *tdi);

  void PropagateAlphaBackwardsThreadedWorker(
    const Matrix &q, const Matrix &u, 
    const Vector alpha[], ThreadData *tdi);
};



#endif // __PointSetOptimalControlSystem_h_
