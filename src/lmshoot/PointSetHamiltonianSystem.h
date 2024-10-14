#ifndef __PointSetHamiltionianSystem_h_
#define __PointSetHamiltionianSystem_h_

#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_vector_fixed.h>
#include <vector>

namespace ctpl { class thread_pool; }

template <class TFloat, unsigned int VDim>
class PointSetHamiltonianSystem
{
public:

  typedef vnl_matrix<TFloat> Matrix;
  typedef vnl_vector<TFloat> Vector;

  typedef vnl_vector_fixed<TFloat, VDim> VecD;

  typedef PointSetHamiltonianSystem<TFloat, VDim> Self;

  /**
   * Constructor - set basic parameters of the system and allocate
   * all of the necessary matrices
   * 
   * q0      : N x D vector of template landmark positions
   * sigma   : standard deviation of the Gaussian kernel 
   * Nt      : number of timesteps for the ODE
   * Nr      : number of 'rider' points, i.e., points that are carried by
   *           the flow but are not themselves control points. Default is 0.
   *           Rider points are at the end of the vector q0.
   */
  PointSetHamiltonianSystem(
    const Matrix &q0, 
    TFloat sigma,
    unsigned int Nt,
    unsigned int Nr,
    unsigned int n_threads);

  ~PointSetHamiltonianSystem();

  /** 
   * Get the number of time steps
   */
  unsigned int GetN() const { return N; }

  /**
   * Turn on Ralston integration, i.e., second-order Runge-Kutta method.
   * It is twice slower but more accurate.
   */
  void SetRalstonIntegration(bool flag) { flag_ralston_integration = flag; }
  bool GetRalstonIntegration() const { return flag_ralston_integration; }

  /**
   * Multi-threaded computation of the Hamiltonian and derivatives. For now it
   * does not support hessian computation
   */
  TFloat ComputeHamiltonianAndGradientThreaded(const Matrix &q, const Matrix &p);
  

  /**
   * Compute the Hamiltonian and its derivatives for given p/q. The
   * computation is stored internally in variables Hp, Hq, Hqq, Hqp, Hpp
   */
  TFloat ComputeHamiltonianJet(const Matrix &q, const Matrix &p, bool flag_hessian);

  /**
   * Compute the product of the Hessian of the Hamiltonian and a pair of
   * vectors \alpha and \beta, as follows:
   *
   * d_alpha[a] = \sum_b (alpha[b] * Hpq[b][a] - beta[b] * Hqq[b][a])
   * d_beta[a] = \sum_b (alpha[b] * Hpp[b][a] - beta[b] * Hqp[b][a])
   *
   * This allows backward flowing of the gradient without having to store
   * the Hessian matrices, and hopefully would be somewhat faster
   */
  void ApplyHamiltonianHessianToAlphaBeta(
      const Matrix &q, const Matrix &p,
      const Vector alpha[VDim], const Vector beta[VDim],
      Vector d_alpha[VDim], Vector d_beta[VDim]);

  void ApplyHamiltonianHessianToAlphaBetaThreaded(
      const Matrix &q, const Matrix &p,
      const Vector alpha[VDim], const Vector beta[VDim],
      Vector d_alpha[VDim], Vector d_beta[VDim]);

  /**
   * Flow the Hamiltonian system with initial momentum p0 without gradient 
   * computation. Returns the kinetic energy (Hamiltonian value that should 
   * be preserved over the time evolution). 
   */
  TFloat FlowHamiltonian(const Matrix &p0, Matrix &q, Matrix &p);

  /**
   * Flow the Hamiltonian system with gradient computation. The gradient is 
   * strored as a VDim x VDim array of k x k matrices. This is a pretty expensive
   * operation because of many matrix multiplications that are involved
   */
  TFloat FlowHamiltonianWithGradient(
    const Matrix &p0, Matrix &q, Matrix &p,
    Matrix grad_q[VDim][VDim], Matrix grad_p[VDim][VDim]);

  /** 
   * Computes the expression alpha' * Q1 + beta' * P1, where alpha and beta are 
   * vectors, and Q1 and P1 are D_p0(q1) and D_p0(p1), respectively.
   *
   * This can be used to efficiently compute the gradient of any function f(q1,p1)
   * with respect to the initial momentum, since 
   *
   *    D_p0(f) = Dq_f' * D_p0(q1) + Dp_f' * D_p0(p1)
   *
   * The method flows alpha and beta back in time, avoiding expensive matrix-matrix
   * multiplication. This method requires FlowHamiltonian to have been run already
   */ 
  void FlowGradientBackward(
    const Vector alpha[VDim], const Vector beta[VDim], Vector result[VDim]);

  void FlowGradientBackward(
    const Matrix &alpha, const Matrix &beta, Matrix &result);



  /**
   * This function is used when we have an objective function that involves
   * q_t for all time points, and not just the final time point. For example,
   * this may be a constraint that must hold through the entire flow. The input
   * alpha here is parameterized by time, i.e., the gradient of the objective
   * functions evaluated at each time point. 
   *
   * Here there is no beta input, which is normally zero since we rarely involve
   * the momentum in objective functions
   */
  void FlowTimeVaryingGradientsBackward(const std::vector<Matrix> d_obj__d_qt, Vector result[VDim]);

  const Vector &GetHp(unsigned int d) const { return Hp[d]; }
  const Vector &GetHq(unsigned int d) const { return Hq[d]; }
  const Matrix &GetHqq(unsigned int a, unsigned int b) const { return Hqq[a][b]; }
  const Matrix &GetHqp(unsigned int a, unsigned int b) const { return Hqp[a][b]; }
  const Matrix &GetHpp(unsigned int a, unsigned int b) const { return Hpp[a][b]; }

  const Matrix &GetQt(unsigned int t) const { return Qt[t]; }
  const Matrix &GetPt(unsigned int t) const { return Pt[t]; }

  const TFloat GetDeltaT() const { return dt; }

  /**
   * Interpolate the velocity at a given point in space at time point t. This
   * can only be called after running FlowHamiltonian()
   */
  void InterpolateVelocity(unsigned int t, const TFloat *x, TFloat *v);

protected:

  // Initial ladnmark coordinates - fixed for duration
  Matrix q0;

  // Standard deviation of Gaussian kernel; time step
  TFloat sigma, dt;

  // Number of timesteps for integration; number of control and total points
  unsigned int N, k, m;

  // Gradient of the Hamiltonian components: Hq and Hp
  Vector Hp[VDim], Hq[VDim];

  // Number of threads used
  unsigned int n_threads;

  // Type of integration to use, Euler or Ralston method
  bool flag_ralston_integration = false;

  // Multi-threaded quantities
  struct ThreadData 
    {
    // List of rows handled by this thread
    std::vector<unsigned int> rows;
    TFloat H;
    Vector Hp[VDim], Hq[VDim];
    Vector d_alpha[VDim], d_beta[VDim];
    };

  // Data associated with each thread
  std::vector<ThreadData> td;

  // A thread pool to handle execution
  ctpl::thread_pool *thread_pool;

  // Hessian of the Hamiltonian components: Hqq, Hqp, Hpp
  // matrices Hqq and Hpp are symmetric
  Matrix Hqq[VDim][VDim], Hqp[VDim][VDim], Hpp[VDim][VDim];

  // Streamlines - paths of the landmarks over time
  std::vector<Matrix> Qt, Pt;

  // Additional intermediate points used during Raston integration
  std::vector<Matrix> Qt_ralston, Pt_ralston;

  // Set up multi-threaded variables
  void SetupMultiThreaded();

  // Multi-threaded worker functions
  void ComputeHamiltonianAndGradientThreadedWorker(const Matrix *q, const Matrix *p, ThreadData *tdi);
  void ApplyHamiltonianHessianToAlphaBetaThreadedWorker(
    const Matrix *q, const Matrix *p, const Vector alpha[], const Vector beta[], ThreadData *tdi);


private:
  // Helper method used during Euler or Ralston integration
  void UpdatePQbyHamiltonianGradient(Matrix &q, Matrix &p, TFloat step);
};



#endif // __PointSetHamiltionianSystem_h_
