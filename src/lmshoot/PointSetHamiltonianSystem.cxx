#include "PointSetHamiltonianSystem.h"
#include <vnl/vnl_fastops.h>
#include <iostream>
#include "ctpl_stl.h"

template <class TFloat, unsigned int VDim>
PointSetHamiltonianSystem<TFloat, VDim>
::PointSetHamiltonianSystem(
    const Matrix &q0, TFloat sigma,
    unsigned int Nt, unsigned int Nr,
    unsigned int n_threads)
{
  // Copy parameters
  this->q0 = q0;
  this->sigma = sigma;
  this->N = Nt;
  this->m = q0.rows();
  this->k = this->m - Nr;
  this->dt = 1.0 / (N-1);

  // Set the number of threads
  this->n_threads = n_threads > 0 ? n_threads : std::thread::hardware_concurrency();

  // Allocate H derivatives
  for(unsigned int a = 0; a < VDim; a++)
    {
    this->Hq[a].set_size(k);
    this->Hp[a].set_size(m);

    for(unsigned int b = 0; b < VDim; b++)
      {
      this->Hqq[a][b].set_size(k,k);
      this->Hqp[a][b].set_size(k,k);
      this->Hpp[a][b].set_size(k,k);
      }
    }

  SetupMultiThreaded();
}

template <class TFloat, unsigned int VDim>
PointSetHamiltonianSystem<TFloat, VDim>
::~PointSetHamiltonianSystem()
{
  delete thread_pool;
}

template <class TFloat, unsigned int VDim>
void
PointSetHamiltonianSystem<TFloat, VDim>
::ComputeHamiltonianAndGradientThreadedWorker(const Matrix *q, const Matrix *p, ThreadData *tdi)
{
  // Gaussian factor, i.e., K(z) = exp(f * z)
  TFloat f = -0.5 / (sigma * sigma);
  TFloat f_times_2 = 2.0 * f;

  // Storage for the displacement vector Qi-Qj
  TFloat dq[VDim];

  // Get access to pointers to avoid relying on slower VNL access routines
  auto p_da = p->data_array(), q_da = q->data_array();

  // Similarly, get access to output arrays
  TFloat *Hq_da[VDim], *Hp_da[VDim];
  for(unsigned int a = 0; a < VDim; a++)
    {
    Hq_da[a] = tdi->Hq[a].data_block();
    Hp_da[a] = tdi->Hp[a].data_block();
    }

  // Initialize hamiltonian for the subset of indices worked on by the thread
  tdi->H = 0.0;

  // Initialize the output arrays (TODO: move this into initialization)
  for(unsigned int a = 0; a < VDim; a++)
    {
    tdi->Hp[a].fill(0.0);
    tdi->Hq[a].fill(0.0);
    }

  // Loop over all control points
  for(unsigned int i : tdi->rows)
    {
    // Get a pointer to pi for faster access?
    const TFloat *pi = p_da[i], *qi = q_da[i];

    // The diagonal terms
    for(unsigned int a = 0; a < VDim; a++)
      {
      tdi->H += 0.5 * pi[a] * pi[a];
      Hp_da[a][i] += pi[a];
      }

    // The off-diagonal terms - loop over later control points
    for(unsigned int j = i+1; j < k; j++)
      {
      const TFloat *pj = p_da[j], *qj = q_da[j];

      // Dot product of Pi and Pj
      TFloat pi_pj = 0.0;
      TFloat dq_norm_sq = 0.0;

      // Compute above quantities
      for(unsigned int a = 0; a < VDim; a++)
        {
        dq[a] = qi[a] - qj[a];
        pi_pj += pi[a] * pj[a];
        dq_norm_sq += dq[a] * dq[a];
        }

      // Compute the Gaussian and its derivatives
      TFloat g = exp(f * dq_norm_sq);
      TFloat g_pi_pj = g * pi_pj;
      TFloat z = f_times_2 * g_pi_pj;

      // Accumulate the Hamiltonian
      tdi->H += g_pi_pj;

      // Accumulate the derivatives
      for(unsigned int a = 0; a < VDim; a++)
        {
        // First derivatives
        Hq_da[a][i] += z * dq[a];
        Hp_da[a][i] += g * pj[a];

        Hq_da[a][j] -= z * dq[a];
        Hp_da[a][j] += g * pi[a];
        }
      } // loop over j

    // Rider points
    for(unsigned int j = k; j < m; j++)
      {
      const TFloat *qj = q->data_array()[j];

      TFloat delta, d2 = 0;
      for(unsigned int a = 0; a < VDim; a++)
        {
        delta = qi[a] - qj[a];
        d2 += delta * delta;
        }

      TFloat g = exp(f * d2);
      for(unsigned int a = 0; a < VDim; a++)
        Hp_da[a][j] += g * pi[a];
      }

    } // loop over i
}

template <class TFloat, unsigned int VDim>
void
PointSetHamiltonianSystem<TFloat, VDim>
::SetupMultiThreaded()
{
  // Split the indices among threads
  td.resize(n_threads);

  // Create the thread pool
  thread_pool = new ctpl::thread_pool(n_threads);
 
  // Assign lines in pairs, one at the top of the symmetric matrix K and
  // one at the bottom of K. The loop below will not assign the middle
  // line when there is an odd number of points (e.g., line 7 when there are 15)
  for(int i = 0; i < (int) k/2; i++)
    {
    int i_thread = i % n_threads;
    td[i_thread].rows.push_back(i);
    td[i_thread].rows.push_back((k-1) - i);
    }

  // Handle the middle line for odd number of vertices
  if(k % 2 == 1)
    td[(k / 2) % n_threads].rows.push_back(k/2);

  // Allocate the per-thread arrays
  for(unsigned int i = 0; i < n_threads; i++)
    {
    for(unsigned int a = 0; a < VDim; a++)
      {
      td[i].Hp[a] = Vector(m, 0.0);
      td[i].Hq[a] = Vector(k, 0.0);
      td[i].d_alpha[a] = Vector(m, 0.0);
      td[i].d_beta[a] = Vector(k, 0.0);
      }
    }
}

template <class TFloat, unsigned int VDim>
TFloat
PointSetHamiltonianSystem<TFloat, VDim>
::ComputeHamiltonianAndGradientThreaded(const Matrix &q, const Matrix &p)
{
  // Submit the jobs to thread pool
  std::vector<std::future<void>> futures;
  for(auto &tdi : td)
    {
    futures.push_back(
      thread_pool->push(
        [&](int id) { this->ComputeHamiltonianAndGradientThreadedWorker(&q, &p, &tdi); }));
    }

  // Wait for completion
  for(auto &f : futures)
    f.get();

  // Clear the threads
  thread_pool->clear_queue();

  // Compile the results
  TFloat H = 0.0;
  for(unsigned int a = 0; a < VDim; a++)
    {
    Hq[a].fill(0.0); Hp[a].fill(0.0);
    }

  for(unsigned int i = 0; i < td.size(); i++)
    {
    for(unsigned int a = 0; a < VDim; a++)
      {
      Hq[a] += td[i].Hq[a]; 
      Hp[a] += td[i].Hp[a]; 
      }
    H += td[i].H;
    }

  return H;
}

// #define _SHOOTING_USE_EIGEN_

// This is some working code that uses Eigen matrix computations instead of
// hand-crafted code for forward flow. It ended up being quite a bit slower
// though, even with MKL as the backend.
#ifdef _SHOOTING_USE_EIGEN_

#include <Eigen/Eigen>

template <class TFloat, unsigned int VDim>
TFloat
PointSetHamiltonianSystem<TFloat, VDim>
::FlowHamiltonian(const Matrix &p0, Matrix &q_vnl, Matrix &p_vnl)
{
  typedef Eigen::Matrix<TFloat, Eigen::Dynamic, Eigen::Dynamic> EMat;
  typedef Eigen::Matrix<TFloat, Eigen::Dynamic, 1> EVec;

  // A map to encapsulate VNL inputs
  typedef Eigen::Map<Eigen::Matrix<TFloat, Eigen::Dynamic, VDim, Eigen::RowMajor> > VNLWrap;

  // Initialize p and q
  q_vnl = q0; p_vnl = p0;
  VNLWrap q(q_vnl.data_block(), q_vnl.rows(), VDim);
  VNLWrap p(p_vnl.data_block(), p_vnl.rows(), VDim);

  // Allocate the streamline arrays
  Qt.resize(N); Qt[0] = q0;
  Pt.resize(N); Pt[0] = p0;

  // Get the number of points
  unsigned int n = q.rows();

  // The return value
  TFloat H;

  // The partials of the Hamiltonian
  EMat _Hp(n, 3), _Hq(n, 3);

  // Initialize the distance matrix
  EMat K(n, n), KP(n, n);

  // Flow over time
  for(unsigned int t = 1; t < N; t++)
    {
    // Compute the distance matrix
    K = q * q.transpose();
    EVec q_sq = K.diagonal();
    K *= -2.0;
    K.colwise() += q_sq;
    K.rowwise() += q_sq.transpose();

    // Compute the kernel matrix
    TFloat f = -0.5 / (sigma * sigma);
    K = (K * f).array().exp();

    // Compute the Hamiltonian derivatives
    _Hp = K * p;

    // Scale the matrix by outer product of the p's
    KP = K.cwiseProduct(p * p.transpose());

    // Take the row-sums
    _Hq = q;
    _Hq.array().colwise() *= KP.rowwise().sum().array();
    _Hq = 2. * f * (_Hq - KP * q);

    // Update q and p
    q += dt * _Hp;
    p -= dt * _Hq;

    // Store the flow results
    Qt[t] = q_vnl; Pt[t] = p_vnl;

    // store the first hamiltonian value
    if(t == 1)
      {
      H = 0.5 * KP.sum();
      }
    }

  return H;
}

#else

template <class TFloat, unsigned int VDim>
void
PointSetHamiltonianSystem<TFloat, VDim>
::UpdatePQbyHamiltonianGradient(Matrix &q, Matrix &p, TFloat step)
{
  // Euler update for the momentum (only control points)
  for(unsigned int i = 0; i < k; i++)
    for(unsigned int a = 0; a < VDim; a++)
      p(i,a) -= step * Hq[a](i);

  // Euler update for the points (all points)
  for(unsigned int i = 0; i < m; i++)
    for(unsigned int a = 0; a < VDim; a++)
      q(i,a) += step * Hp[a](i);
}

template <class TFloat, unsigned int VDim>
TFloat
PointSetHamiltonianSystem<TFloat, VDim>
::FlowHamiltonian(const Matrix &p0, Matrix &q, Matrix &p)
{
  // Initialize q and p
  q = q0; p = p0;

  // Allocate the streamline arrays
  Qt.resize(N); Qt[0] = q0;
  Pt.resize(N); Pt[0] = p0;

  // The return value
  TFloat H, H0 = 0.0;

  // Allocate additional storage for Ralston
  Qt_ralston.resize(N);
  Pt_ralston.resize(N);

  // Flow over time
  for(unsigned int t = 1; t < N; t++)
    {
    // Compute the hamiltonian
    H = ComputeHamiltonianAndGradientThreaded(q, p);

    if(flag_ralston_integration)
      {
      // Compute the intermediate point x_i
      Pt_ralston[t-1] = p; Qt_ralston[t-1] = q;
      UpdatePQbyHamiltonianGradient(Qt_ralston[t-1], Pt_ralston[t-1], 2 * dt / 3);

      // Update p,q using the initial point gradient
      UpdatePQbyHamiltonianGradient(q, p, dt / 4);

      // Evaluate the system at the mid-point position
      ComputeHamiltonianAndGradientThreaded(Qt_ralston[t-1], Pt_ralston[t-1]);

      // Update using the ralston point gradient
      UpdatePQbyHamiltonianGradient(q, p, 3 * dt / 4);
      }

    else
      {
      // Just one update
      UpdatePQbyHamiltonianGradient(q, p, dt);
      }

    // Store the flow results
    Qt[t] = q; Pt[t] = p;

    // store the first hamiltonian value
    if(t == 1)
      H0 = H;
    }

  return H0;
}
#endif

template <class TFloat, unsigned int VDim>
TFloat
PointSetHamiltonianSystem<TFloat, VDim>
::ComputeHamiltonianJet(const Matrix &q, const Matrix &p, bool flag_hessian)
{
  // Gaussian factor, i.e., K(z) = exp(f * z)
  TFloat f = -0.5 / (sigma * sigma);

  // Initialize the gradient and Hessian terms to zeros
  for(unsigned int a = 0; a < VDim; a++)
    {
    this->Hq[a].fill(0.0);
    this->Hp[a].fill(0.0);

    if(flag_hessian)
      {
      for(unsigned int b = 0; b < VDim; b++)
        {
        this->Hqq[a][b].fill(0.0);
        this->Hqp[a][b].fill(0.0);
        this->Hpp[a][b].fill(0.0);
        }
      }
    }

  // Initialize hamiltonian
  TFloat H = 0.0;

  // Loop over all points
  for(unsigned int i = 0; i < k; i++)
    {
    // Get a pointer to pi for faster access?
    const TFloat *pi = p.data_array()[i], *qi = q.data_array()[i];

    // The diagonal terms
    for(unsigned int a = 0; a < VDim; a++)
      {
      H += 0.5 * pi[a] * pi[a];
      Hp[a](i) += pi[a];
      if(flag_hessian)
        Hpp[a][a](i,i) = 1.0;
      }

    // TODO: you should be able to do this computation on half the matrix, it's symmetric!
    for(unsigned int j = i+1; j < k; j++)
      {
      const TFloat *pj = p.data_array()[j], *qj = q.data_array()[j];

      // Vector Qi-Qj
      VecD dq;

      // Dot product of Pi and Pj
      TFloat pi_pj = 0.0;

      // Compute above quantities
      for(unsigned int a = 0; a < VDim; a++)
        {
        dq[a] = qi[a] - qj[a];
        pi_pj += pi[a] * pj[a];
        }

      // Compute the Gaussian and its derivatives
      TFloat g, g1, g2;
      g = exp(f * dq.squared_magnitude()), g1 = f * g, g2 = f * g1;

      // Accumulate the Hamiltonian
      H += pi_pj * g;

      // Accumulate the derivatives
      for(unsigned int a = 0; a < VDim; a++)
        {
        // First derivatives
        Hq[a](i) += 2 * pi_pj * g1 * dq[a];
        Hp[a](i) += g * pj[a];

        Hq[a](j) -= 2 * pi_pj * g1 * dq[a];
        Hp[a](j) += g * pi[a];

        // Second derivatives
        if(flag_hessian)
          {
          TFloat term_2_g1_dqa = 2.0 * g1 * dq[a];
          for(unsigned int b = 0; b < VDim; b++)
            {
            TFloat val_qq = 2.0 * pi_pj * (2 * g2 * dq[a] * dq[b] + ((a == b) ? g1 : 0.0));
            Hqq[a][b](i,j) -= val_qq;
            Hqq[a][b](i,i) += val_qq;
            Hqq[a][b](j,i) -= val_qq;
            Hqq[a][b](j,j) += val_qq;

            Hqp[a][b](i,j) += term_2_g1_dqa * pi[b];
            Hqp[a][b](i,i) += term_2_g1_dqa * pj[b];
            Hqp[a][b](j,i) -= term_2_g1_dqa * pj[b];
            Hqp[a][b](j,j) -= term_2_g1_dqa * pi[b];
            }

          Hpp[a][a](i,j) = g;
          Hpp[a][a](j,i) = g;
          }
        }
      } // loop over j
    } // loop over i
  return H;
}


template <class TFloat, unsigned int VDim>
void
PointSetHamiltonianSystem<TFloat, VDim>
::ApplyHamiltonianHessianToAlphaBetaThreadedWorker(
  const Matrix *q, const Matrix *p, 
  const Vector alpha[], const Vector beta[],
  ThreadData *tdi)
{
  // Gaussian factor, i.e., K(z) = exp(f * z)
  TFloat f = -0.5 / (sigma * sigma);

  // Storage for the displacement vector Qi-Qj
  TFloat dq[VDim];

  // Get access to pointers to avoid relying on slower VNL access routines
  auto p_da = p->data_array(), q_da = q->data_array();

  // Similarly, get access to output arrays
  const TFloat *alpha_da[VDim], *beta_da[VDim];
  TFloat *d_alpha_da[VDim], *d_beta_da[VDim];
  for(unsigned int a = 0; a < VDim; a++)
    {
    alpha_da[a] = alpha[a].data_block();
    beta_da[a] = beta[a].data_block();
    d_alpha_da[a] = tdi->d_alpha[a].data_block();
    d_beta_da[a] = tdi->d_beta[a].data_block();
    }

  // Initialize the output arrays (TODO: move this into initialization)
  for(unsigned int a = 0; a < VDim; a++)
    {
    tdi->d_alpha[a].fill(0.0);
    tdi->d_beta[a].fill(0.0);
    }

  // Loop over all control points in this thread
  for(unsigned int i : tdi->rows)
    {
    // Get a pointer to pi for faster access?
    const TFloat *pi = p_da[i], *qi = q_da[i];

    // Loop over later control points
    for(unsigned int j = i+1; j < k; j++)
      {
      const TFloat *pj = p_da[j], *qj = q_da[j];

      // Dot product of Pi and Pj
      TFloat pi_pj = 0.0;
      TFloat dq_norm_sq = 0.0;

      // Compute above quantities
      for(unsigned int a = 0; a < VDim; a++)
        {
        dq[a] = qi[a] - qj[a];
        pi_pj += pi[a] * pj[a];
        dq_norm_sq += dq[a] * dq[a];
        }

      // Compute the Gaussian and its derivatives
      TFloat g, g1;
      g = exp(f * dq_norm_sq), g1 = f * g;

      /*
       * d_beta[a] = alpha[b] * Hpp[b][a] =
       *
       * d_beta[b,j] = Sum_i,a (...
       *     alpha[a,i] * Hpp[a][b][i][j] - beta[a][i] * Hqp[a][b][i][j] );
       *
       * d_alpha[b,j] = Sum_i,a (...
       *     alpha[a,i] * Hpq[a][b][i][j] - beta[a][i] * Hqq[a][b][i][j] );
       *
       * d_alpha[b,j] = Sum_i,a (...
       *     alpha[a,i] * Hqp[b][a][j][i] - beta[a][i] * Hqq[a][b][i][j] );
       */

      // Accumulate the derivatives
      for(unsigned int a = 0; a < VDim; a++)
        {
        TFloat term_2_g1_dqa = 2.0 * g1 * dq[a];
        TFloat alpha_j_pi_plus_alpha_i_pj = 0.0;
        TFloat d_beta_ji_a = (beta_da[a][j] - beta_da[a][i]);

        for(unsigned int b = 0; b < VDim; b++)
          {
          TFloat val_qq = 2.0 * pi_pj * (f * term_2_g1_dqa * dq[b] + ((a == b) ? g1 : 0.0));
          TFloat upd = d_beta_ji_a * val_qq;

          // We can take advantage of the symmetry of Hqq.
          d_alpha_da[b][j] -= upd;
          d_alpha_da[b][i] += upd;

          d_beta_da[b][j] += d_beta_ji_a * term_2_g1_dqa * pi[b];
          d_beta_da[b][i] += d_beta_ji_a * term_2_g1_dqa * pj[b];

          alpha_j_pi_plus_alpha_i_pj += alpha_da[b][j] * pi[b] + alpha_da[b][i] * pj[b];
          }

        d_alpha_da[a][i] += term_2_g1_dqa * alpha_j_pi_plus_alpha_i_pj;
        d_alpha_da[a][j] -= term_2_g1_dqa * alpha_j_pi_plus_alpha_i_pj;

        d_beta_da[a][i] += g * alpha_da[a][j];
        d_beta_da[a][j] += g * alpha_da[a][i];
        }
      } // loop over j

    for(unsigned int a = 0; a < VDim; a++)
      {
      d_beta_da[a][i] += alpha_da[a][i];
      }

    // Loop over rider points.
    for(unsigned int j = k; j < m; j++)
      {
      const TFloat *qj = q_da[j];

      // Compute the exponent term
      TFloat dq_norm_sq = 0.0;
      for(unsigned int a = 0; a < VDim; a++)
        {
        dq[a] = qi[a] - qj[a];
        dq_norm_sq += dq[a] * dq[a];
        }

      // Compute the Gaussian and its derivative terms
      TFloat g, g1;
      g = exp(f * dq_norm_sq), g1 = f * g;

      // Accumulate derivatives
      for(unsigned int a = 0; a < VDim; a++)
        {
        TFloat term_2_g1_dqa = 2.0 * g1 * dq[a];
        for(unsigned int b = 0; b < VDim; b++)
          {
          // Update for the control point
          d_alpha_da[a][i] += alpha_da[b][j] * term_2_g1_dqa * pi[b];

          // tdi->d_alpha[b][j] += alpha[b][j];
          d_alpha_da[a][j] -= alpha_da[b][j] * term_2_g1_dqa * pi[b];
          }

        d_beta_da[a][i] += alpha_da[a][j] * g;
        }
      }

    } // loop over i
}


template <class TFloat, unsigned int VDim>
void
PointSetHamiltonianSystem<TFloat, VDim>
::ApplyHamiltonianHessianToAlphaBetaThreaded(
    const Matrix &q, const Matrix &p,
    const Vector alpha[VDim], const Vector beta[VDim],
    Vector d_alpha[VDim], Vector d_beta[VDim])
{
  // Initialize the arrays to be accumulated
  for(unsigned int a = 0; a < VDim; a++)
    {
    d_alpha[a].fill(0.0);
    d_beta[a].fill(0.0);
    }

  // Submit the jobs to thread pool
  std::vector<std::future<void>> futures;
  for(auto &tdi : td)
    {
    futures.push_back(
      thread_pool->push(
        [&](int) { this->ApplyHamiltonianHessianToAlphaBetaThreadedWorker(&q, &p, alpha, beta, &tdi); }));
    }

  // Wait for completion
  for(auto &f : futures)
    f.get();

  // Accumulate the d_alpha and d_beta from threads
  for(unsigned int i = 0; i < td.size(); i++)
    {
    for(unsigned int a = 0; a < VDim; a++)
      {
      d_alpha[a] += td[i].d_alpha[a];
      d_beta[a] += td[i].d_beta[a];
      }
    }
}


template <class TFloat, unsigned int VDim>
void
PointSetHamiltonianSystem<TFloat, VDim>
::FlowGradientBackward(
  const Vector alpha1[VDim],
  const Vector beta1[VDim],
  Vector result[VDim])
{
  // Allocate update vectors for alpha and beta
  Vector alpha[VDim], beta[VDim];
  Vector d_alpha[VDim], d_beta[VDim];

  // What kind of update do we do
  Vector alpha_ralston[VDim], beta_ralston[VDim];
  Vector d_alpha_ralston[VDim], d_beta_ralston[VDim];

  for(unsigned int a = 0; a < VDim; a++)
    {
    alpha[a] = alpha1[a];
    beta[a] = beta1[a];
    d_alpha[a].set_size(m);
    d_beta[a].set_size(k);

    if(flag_ralston_integration)
      {
      d_alpha_ralston[a].set_size(m);
      d_beta_ralston[a].set_size(k);
      }
    }

  // Work our way backwards
  for(int t = N-1; t > 0; t--)
    {
    if(flag_ralston_integration)
      {
      // Worked this out with PyTorch
      // G1 = adjunct_f(x_list_r[i-1], gamma)
      // G0 = adjunct_f(x_list[i-1], gamma + 2 * dt * G1)
      // gamma = gamma + 0.25 * dt * G0 + 0.75 * dt * G1

      // Compute G1
      ApplyHamiltonianHessianToAlphaBetaThreaded(
            Qt_ralston[t - 1], Pt_ralston[t - 1], alpha, beta, d_alpha_ralston, d_beta_ralston);

      // Compute G0
      for(unsigned int a = 0; a < VDim; a++)
        {
        alpha_ralston[a] = alpha[a] + 2 * dt * d_alpha_ralston[a];
        beta_ralston[a] = beta[a] + 2 * dt * d_beta_ralston[a];
        }

      ApplyHamiltonianHessianToAlphaBetaThreaded(
            Qt[t - 1], Pt[t - 1], alpha_ralston, beta_ralston, d_alpha, d_beta);

      // Update the vectors
      for(unsigned int a = 0; a < VDim; a++)
        {
        alpha[a] += d_alpha[a] * (0.25 * dt) +  d_alpha_ralston[a] * (0.75 * dt);
        beta[a] += d_beta[a] * (0.25 * dt) +  d_beta_ralston[a] * (0.75 * dt);
        }
      }
    else
      {
      // Apply Hamiltonian Hessian to get an update in alpha/beta
      ApplyHamiltonianHessianToAlphaBetaThreaded(
            Qt[t - 1], Pt[t - 1], alpha, beta, d_alpha, d_beta);

      // Update the vectors
      for(unsigned int a = 0; a < VDim; a++)
        {
        alpha[a] += dt * d_alpha[a];
        beta[a] += dt * d_beta[a];
        }
      }
    }

  // Finally, what we are really after are the betas
  for(unsigned int a = 0; a < VDim; a++)
    {
    result[a] = beta[a];
    }
}

template <class TFloat, unsigned int VDim>
void
PointSetHamiltonianSystem<TFloat, VDim>
::FlowGradientBackward(
  const Matrix &alpha, const Matrix &beta, Matrix &result)
{
  Vector alpha_v[VDim], beta_v[VDim], result_v[VDim];
  for(unsigned int a = 0; a < VDim; a++)
    {
    alpha_v[a] = alpha.get_column(a);
    beta_v[a] = beta.get_column(a);
    result_v[a].set_size(alpha_v[a].size());
    }

  this->FlowGradientBackward(alpha_v, beta_v, result_v);

  for(unsigned int a = 0; a < VDim; a++)
    result.set_column(a, result_v[a]);
}

template <class TFloat, unsigned int VDim>
void
PointSetHamiltonianSystem<TFloat, VDim>
::ApplyHamiltonianHessianToAlphaBeta(
    const Matrix &q, const Matrix &p,
    const Vector alpha[], const Vector beta[],
    Vector d_alpha[], Vector d_beta[])
{
  TFloat f = -0.5 / (sigma * sigma);

  for(int a = 0; a < VDim; a++)
    {
    d_alpha[a].fill(0.0);
    d_beta[a].fill(0.0);
    }


  // Loop over all points
  for(unsigned int i = 0; i < k; i++)
    {
    // Get a pointer to pi for faster access?
    const TFloat *pi = p.data_array()[i], *qi = q.data_array()[i];

    // TODO: you should be able to do this computation on half the matrix, it's symmetric!
    #pragma omp parallel for
    for(unsigned int j = i+1; j < k; j++)
      {
      const TFloat *pj = p.data_array()[j], *qj = q.data_array()[j];

      // Vector Qi-Qj
      VecD dq;

      // Dot product of Pi and Pj
      TFloat pi_pj = 0.0;

      // Compute above quantities
      for(unsigned int a = 0; a < VDim; a++)
        {
        dq[a] = qi[a] - qj[a];
        pi_pj += pi[a] * pj[a];
        }

      // Compute the Gaussian and its derivatives
      TFloat g, g1, g2;
      g = exp(f * dq.squared_magnitude()), g1 = f * g, g2 = f * g1;

      /*
       * d_beta[a] = alpha[b] * Hpp[b][a] =
       *
       * d_beta[b,j] = Sum_i,a (...
       *     alpha[a,i] * Hpp[a][b][i][j] - beta[a][i] * Hqp[a][b][i][j] );
       *
       * d_alpha[b,j] = Sum_i,a (...
       *     alpha[a,i] * Hpq[a][b][i][j] - beta[a][i] * Hqq[a][b][i][j] );
       *
       * d_alpha[b,j] = Sum_i,a (...
       *     alpha[a,i] * Hqp[b][a][j][i] - beta[a][i] * Hqq[a][b][i][j] );
       */

      // Accumulate the derivatives
      for(unsigned int a = 0; a < VDim; a++)
        {

        TFloat term_2_g1_dqa = 2.0 * g1 * dq[a];
        TFloat alpha_j_pi_plus_alpha_i_pj = 0.0;
        TFloat d_beta_ji_a = (beta[a][j] - beta[a][i]);

        for(unsigned int b = 0; b < VDim; b++)
          {
          TFloat val_qq = 2.0 * pi_pj * (2 * g2 * dq[a] * dq[b] + ((a == b) ? g1 : 0.0));
          TFloat upd = d_beta_ji_a * val_qq;

          // We can take advantage of the symmetry of Hqq.
          d_alpha[b][j] -= upd;
          d_alpha[b][i] += upd;

          d_beta[b][j] += d_beta_ji_a * term_2_g1_dqa * pi[b];
          d_beta[b][i] += d_beta_ji_a * term_2_g1_dqa * pj[b];

          alpha_j_pi_plus_alpha_i_pj += alpha[b][j] * pi[b] + alpha[b][i] * pj[b];
          }

        d_alpha[a][i] += term_2_g1_dqa * alpha_j_pi_plus_alpha_i_pj;
        d_alpha[a][j] -= term_2_g1_dqa * alpha_j_pi_plus_alpha_i_pj;

        d_beta[a][i] += g * alpha[a][j];
        d_beta[a][j] += g * alpha[a][i];
        }
      } // loop over j

    for(unsigned int a = 0; a < VDim; a++)
      {
      d_beta[a][i] += alpha[a][i];
      }

    } // loop over i}
}

#ifdef _LMSHOOT_DIRECT_USE_LAPACK_

extern "C" {
  int dgemm_(char *, char *, int *, int *, int *, double *, double *, int *, 
    double *, int *, double *, double *, int *);

  int sgemm_(char *, char *, int *, int *, int *, float *, float *, int *,
    float *, int *, float *, float *, int *);
};

#endif

/** WARNING - this is only meant for square matrices! */
template <class TFloat> class BlasInterface
{
public:
  typedef vnl_matrix<TFloat> Mat;
  static void add_AB_to_C(const Mat &A, const Mat &B, Mat &C);
  static void add_AtB_to_C(const Mat &A, const Mat &B, Mat &C);

private:

#ifdef _LMSHOOT_DIRECT_USE_LAPACK_
  static void gems(char *opA, char *opB, int *M, int *N, int *K, TFloat *alpha, TFloat *A, int *LDA,
                   TFloat *B, int *LDB, TFloat *beta, TFloat *C, int *LDC);
#endif
};

#include <Eigen/Core>

/** WARNING - this is only meant for square matrices! */
template <class TFloat>
void
BlasInterface<TFloat>
::add_AB_to_C(const Mat &A, const Mat &B, Mat &C)
{
#ifdef _LMSHOOT_DIRECT_USE_LAPACK_
  assert(
    A.rows() == B.rows() && A.rows() == C.rows() && A.rows() == A.columns() 
    && A.rows() == B.columns() && A.rows() == C.columns());

  char opA = 'N', opB = 'N';
  int M=A.rows(), N=M, K=M, LDA=K, LDB=N, LDC=M;
  TFloat alpha = 1.0, beta = 1.0;
  BlasInterface<TFloat>::gems(&opA, &opB, &M,&N,&K,&alpha,
    const_cast<TFloat *>(B.data_block()),&LDA,
    const_cast<TFloat *>(A.data_block()),&LDB,
    &beta,
    C.data_block(),&LDC);
#else
  typedef Eigen::Matrix<TFloat, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrix;
  typedef Eigen::Map<EigenMatrix> EigenMatrixMap;
  typedef Eigen::Map<const EigenMatrix> EigenMatrixConstMap;

  EigenMatrixConstMap map_A(A.data_block(), A.rows(), A.cols());
  EigenMatrixConstMap map_B(B.data_block(), B.rows(), B.cols());
  EigenMatrixMap map_C(C.data_block(), C.rows(), C.cols());

  // Do the Eigen version of GEMS
  map_C.noalias() += map_A * map_B;
#endif
}

template <class TFloat>
void
BlasInterface<TFloat>
::add_AtB_to_C(const Mat &A, const Mat &B, Mat &C)
{
#ifdef _LMSHOOT_DIRECT_USE_LAPACK_
  assert(
    A.rows() == B.rows() && A.rows() == C.rows() && A.rows() == A.columns() 
    && A.rows() == B.columns() && A.rows() == C.columns());

  char opA = 'N', opB = 'T';
  int M=A.rows(), N=M, K=M, LDA=K, LDB=N, LDC=M;
  TFloat alpha = 1.0, beta = 1.0;
  BlasInterface<TFloat>::gems(&opA, &opB, &M,&N,&K,&alpha,
    const_cast<TFloat *>(B.data_block()),&LDA,
    const_cast<TFloat *>(A.data_block()),&LDB,
    &beta,
    C.data_block(),&LDC);
#else
  typedef Eigen::Matrix<TFloat, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrix;
  typedef Eigen::Map<EigenMatrix> EigenMatrixMap;
  typedef Eigen::Map<const EigenMatrix> EigenMatrixConstMap;

  EigenMatrixConstMap map_A(A.data_block(), A.rows(), A.cols());
  EigenMatrixConstMap map_B(B.data_block(), B.rows(), B.cols());
  EigenMatrixMap map_C(C.data_block(), C.rows(), C.cols());

  // Do the Eigen version of GEMS
  map_C.noalias() += map_A.transpose() * map_B;
#endif
}

#ifdef _LMSHOOT_DIRECT_USE_LAPACK_
template <>
void
BlasInterface<double>
::gems(char *opA, char *opB, int *M, int *N, int *K, double *alpha, double *A, int *LDA,
       double *B, int *LDB, double *beta, double *C, int *LDC)
{
  dgemm_(opA, opB, M,N,K,alpha,A,LDA,B,LDB,beta,C,LDC);
}

template <>
void
BlasInterface<float>
::gems(char *opA, char *opB, int *M, int *N, int *K, float *alpha, float *A, int *LDA,
       float *B, int *LDB, float *beta, float *C, int *LDC)
{
  sgemm_(opA, opB, M,N,K,alpha,A,LDA,B,LDB,beta,C,LDC);
}
#endif


template <class TFloat, unsigned int VDim>
void
PointSetHamiltonianSystem<TFloat, VDim>
::FlowTimeVaryingGradientsBackward(const std::vector<Matrix> d_obj__d_qt, Vector result[VDim])
{
  // Allocate update vectors for alpha and beta
  Vector alpha[VDim], beta[VDim];
  Vector d_alpha[VDim], d_beta[VDim];

  for(int a = 0; a < VDim; a++)
    {
    // Initialize alpha with the last time-point q-gradient
    alpha[a] = d_obj__d_qt[N-1].get_column(a);

    // Initialize beta to zero 
    beta[a].set_size(k); beta[a].fill(0.0);

    d_alpha[a].set_size(k);
    d_beta[a].set_size(k);
    }

  // Work our way backwards
  for(int t = N-1; t > 0; t--)
    {
    // Apply Hamiltonian Hessian to get an update in alpha/beta
    ApplyHamiltonianHessianToAlphaBeta(
          Qt[t - 1], Pt[t - 1], alpha, beta, d_alpha, d_beta);

    // Update the vectors
    for(int a = 0; a < VDim; a++)
      {
      alpha[a] += dt * d_alpha[a] + d_obj__d_qt[t-1].get_column(a);
      beta[a] += dt * d_beta[a];
      }
    } 

  // Finally, what we are really after are the betas
  for(int a = 0; a < VDim; a++)
    {
    result[a] = beta[a];
    }
}

template <class TFloat, unsigned int VDim>
void
PointSetHamiltonianSystem<TFloat, VDim>
::InterpolateVelocity(unsigned int t, const TFloat *x, TFloat *v)
{
  // Gaussian factor, i.e., K(z) = exp(f * z)
  TFloat f = -0.5 / (sigma * sigma);

  // Initialize v to zero
  for(unsigned int a = 0; a < VDim; a++)
    v[a] = 0.0;

  // Compute the velocity for this point
  for(unsigned int i = 0; i < k; i++)
    {
    TFloat dsq = 0.0;
    for(unsigned int a = 0; a < VDim; a++)
      {
      TFloat da = Qt[t](i,a) - x[a];
      dsq += da * da;
      }
    TFloat Kq = exp(f * dsq);

    for(unsigned int a = 0; a < VDim; a++)
      v[a] += Kq * Pt[t](i,a);
    }
}

template <class TFloat, unsigned int VDim>
TFloat
PointSetHamiltonianSystem<TFloat, VDim>
::FlowHamiltonianWithGradient(
  const Matrix &p0, Matrix &q, Matrix &p,
  Matrix grad_q[VDim][VDim], Matrix grad_p[VDim][VDim])
{
  // Initialize q and p
  q = q0; p = p0;

  // Allocate the streamline arrays
  Qt.resize(N); Qt[0] = q0;
  Pt.resize(N); Pt[0] = p0;

  // We need temporary matrices to store the updates of gradient
  Matrix gupd_q[VDim][VDim], gupd_p[VDim][VDim];

  // Initialize the gradients
  for(unsigned int a = 0; a < VDim; a++)
    {
    for(unsigned int b = 0; b < VDim; b++)
      {
      grad_q[a][b].fill(0.0);
      if(a == b)
        grad_p[a][b].set_identity();
      else
        grad_p[a][b].fill(0.0);

      gupd_p[a][b].set_size(k,k);
      gupd_q[a][b].set_size(k,k);
      }
    }

  // The return value
  TFloat H, H0;

  // Flow over time
  for(unsigned int t = 1; t < N; t++)
    {
    // Compute the hamiltonian
    H = ComputeHamiltonianJet(q, p, true);

    // Euler update
    #pragma omp parallel for
    for(unsigned int i = 0; i < k; i++)
      {
      for(unsigned int a = 0; a < VDim; a++)
        {
        q(i,a) += dt * Hp[a](i);
        p(i,a) -= dt * Hq[a](i);
        }
      }

    Qt[t] = q; Pt[t] = p;

    // The nastiest part - some matrix multiplications
    for(unsigned int a = 0; a < VDim; a++)
      {
      for(unsigned int b = 0; b < VDim; b++)
        {
        gupd_q[a][b].fill(0.0);
        gupd_p[a][b].fill(0.0);
        
        for(unsigned int c = 0; c < VDim; c++)
          {
          BlasInterface<TFloat>::add_AB_to_C(Hqp[a][c], grad_p[c][b], gupd_p[a][b]);
          BlasInterface<TFloat>::add_AB_to_C(Hqq[a][c], grad_q[c][b], gupd_p[a][b]);
          BlasInterface<TFloat>::add_AB_to_C(Hpp[a][c], grad_p[c][b], gupd_q[a][b]); 
          BlasInterface<TFloat>::add_AtB_to_C(Hqp[c][a], grad_q[c][b], gupd_q[a][b]); 


          // gupd_p[a][b] += Hqp[a][c] * grad_p[c][b] + Hqq[a][c] * grad_q[c][b];
          // gupd_q[a][b] += Hpp[a][c] * grad_p[c][b] + Hqp[c][a].transpose() * grad_q[c][b];
          }
        }
      }

    for(unsigned int a = 0; a < VDim; a++)
      {
      for(unsigned int b = 0; b < VDim; b++)
        {
        grad_q[a][b] += dt * gupd_q[a][b];
        grad_p[a][b] -= dt * gupd_p[a][b];
        }
      }

    // store the first hamiltonian value
    if(t == 1)
      H0 = H;
    }

  return H0;
}

template class PointSetHamiltonianSystem<double, 2>;
template class PointSetHamiltonianSystem<double, 3>;
template class PointSetHamiltonianSystem<float, 2>;
template class PointSetHamiltonianSystem<float, 3>;
