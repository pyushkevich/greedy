/*=========================================================================

  Program:   ALFABIS fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  This program is part of ALFABIS: Adaptive Large-Scale Framework for
  Automatic Biomedical Image Segmentation.

  ALFABIS development is funded by the NIH grant R01 EB017255.

  ALFABIS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ALFABIS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ALFABIS.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================*/
#include "AffineCostFunctions.h"
#include "MultiImageRegistrationHelper.h"
#include "AffineTransformUtilities.h"
#include "GreedyAPI.h"
#include "GreedyParameters.h"


template <unsigned int VDim, typename TReal>
class RigidHelper
{
public:
  typedef vnl_vector_fixed<double, VDim> Vec;
  typedef vnl_matrix_fixed<double, VDim, VDim> Mat;
  typedef vnl_vector<double> CoeffVec;
  typedef vnl_matrix<double> JacMat;

  static unsigned int GetNumberOfCoefficients() { return 0; }

  static Mat GetRotationMatrix(const Vec &, Mat *) { return Mat(); }

  static CoeffVec GetAffineCoefficientsAndJacobian(
      const CoeffVec &, const Mat &, JacMat * = NULL)
    { return CoeffVec(); }

  static CoeffVec GetRigidCoefficientsFromAffine(const CoeffVec &, Mat &)
    { return CoeffVec(); }

  static Vec GetAxisAngle(const Mat &R) { return Vec(); }

  static Mat GetRandomRotation(const vnl_random &, double) { return Mat(); }
};


template <typename TReal>
class RigidHelper<3, TReal>
{
public:
  typedef vnl_vector_fixed<double, 3> Vec;
  typedef vnl_matrix_fixed<double, 3, 3> Mat;
  typedef vnl_vector<double> CoeffVec;
  typedef vnl_matrix<double> JacMat;

  static unsigned int GetNumberOfCoefficients() { return 6; }

  static Mat GetRotationMatrix(const Vec &q, Mat *d_R = NULL)
  {
    // Compute theta
    double theta = q.magnitude();

    // Predefine the rotation matrix
    Mat R; R.set_identity();

    // Create the Q matrix
    Mat Qmat; Qmat.fill(0.0);
    Qmat(0,1) = -q[2]; Qmat(1,0) =  q[2];
    Qmat(0,2) =  q[1]; Qmat(2,0) = -q[1];
    Qmat(1,2) = -q[0]; Qmat(2,1) =  q[0];

    // Compute the square of the matrix
    Mat QQ = vnl_matrix_fixed_mat_mat_mult(Qmat, Qmat);

    // When theta = 0, rotation is identity
    double eps = 1e-4;
    double a1, a2;

    if(theta > eps)
      {
      // Compute the constant terms in the Rodriguez formula
      a1 = sin(theta) / theta;
      a2 = (1 - cos(theta)) / (theta * theta);

      // Compute the rotation matrix
      R += a1 * Qmat + a2 * QQ;
      }
    else
      {
      R += Qmat;
      }

    // Compute the Jacobian of R wrt q if requested
    if(d_R)
      {
      // Compute the matrices d_Qmat
      Mat d_Qmat[3];
      d_Qmat[0].fill(0); d_Qmat[0](1,2) = -1; d_Qmat[0](2,1) =  1;
      d_Qmat[1].fill(0); d_Qmat[1](0,2) =  1; d_Qmat[1](2,0) = -1;
      d_Qmat[2].fill(0); d_Qmat[2](0,1) = -1; d_Qmat[2](1,0) =  1;

      // Compute partial derivatives of R wrt q
      if(theta > eps)
        {
        // Compute the scaling factors in the Rodriguez formula
        double d_a1 = (theta * cos(theta) - sin(theta)) / (theta * theta * theta);
        double d_a2 = (theta * sin(theta) + 2 * cos(theta) - 2) /
                      (theta * theta * theta * theta);

        // Loop over the coordinate and compute the derivative of the rotation matrix wrt x
        for(int p = 0; p < 3; p++)
          {
          // Compute the gradient of the rotation with respect to q[p]
          d_R[p] = d_a1 * q[p] * Qmat +
                   a1 * d_Qmat[p] +
                   d_a2 * q[p] * vnl_matrix_fixed_mat_mat_mult(Qmat, Qmat)
                   + a2 * (vnl_matrix_fixed_mat_mat_mult(d_Qmat[p], Qmat) +
                           vnl_matrix_fixed_mat_mat_mult(Qmat, d_Qmat[p]));
          }
        }
      else
        {
        for(int p = 0; p < 3; p++)
          d_R[p] = d_Qmat[p];
        }

      }

    return R;
  }

  static CoeffVec GetAffineCoefficientsAndJacobian(
      const CoeffVec &x, const Mat &flip, JacMat *jac = NULL)
  {
    // Place parameters into q and b
    Vec q, b;
    q[0] = x[0]; q[1] = x[1]; q[2] = x[2];
    b[0] = x[3]; b[1] = x[4]; b[2] = x[5];

    // Compute the rotation matrix and (if g) derivative
    Mat R, d_R[3];
    R = GetRotationMatrix(q, jac ? d_R : NULL);

    // Now we have a rotation and a translation, convert to parameters for the affine function
    vnl_vector<double> x_affine(12);
    flatten_affine_transform(flip * R, b, x_affine.data_block());

    // Split depending on whether there is gradient to compute
    if(jac)
      {
      // Create a matrix to hold the jacobian
      jac->set_size(12, 6);
      jac->fill(0.0);

      // Zero vector
      Vec zero_vec; zero_vec.fill(0.0);
      Mat zero_mat; zero_mat.fill(0.0);

      // Fill out the jacobian
      for(int p = 0; p < 3; p++)
        {
        // Fill the corresponding column
        vnl_vector<double> jac_col_q(12);
        flatten_affine_transform(flip * d_R[p], zero_vec, jac_col_q.data_block());
        jac->set_column(p, jac_col_q);

        // Also set column on the right (wrt translation)
        vnl_vector<double> jac_col_b(12);
        Vec ep; ep.fill(0.0); ep[p] = 1;
        flatten_affine_transform(zero_mat, ep, jac_col_b.data_block());
        jac->set_column(p+3, jac_col_b);
        }
      }

    return x_affine;
  }

  static Vec GetAxisAngle(const Mat &R)
  {
    double eps = 1e-4;
    double f_thresh = cos(eps);

    // Compute the matrix logarithm of R
    double f = (vnl_trace(R) - 1) / 2;
    Vec q;
    if(f >= f_thresh)
      {
      q[0] = R(2,1) - R(1,2);
      q[1] = R(0,2) - R(2,0);
      q[2] = R(1,0) - R(0,1);
      q *= 0.5;
      }
    else
      {
      double theta = acos(f);
      double sin_theta = sqrt(1 - f * f);
      q[0] = R(2,1) - R(1,2);
      q[1] = R(0,2) - R(2,0);
      q[2] = R(1,0) - R(0,1);
      q *= theta / (2 * sin_theta);
      }

    return q;
  }

  static CoeffVec GetRigidCoefficientsFromAffine(const CoeffVec &x, Mat &flip)
    {
    Mat A; Vec b;
    unflatten_affine_transform(x.data_block(), A, b);

    // If the determinant of A is negative, we need to use the flip
    if(vnl_determinant(A) < 0.0)
      {
      flip(0,0) = -1.0;
      }
    else
      {
      flip(0,0) = 1.0;
      }

    // Compute polar decomposition of the affine matrix
    vnl_svd<double> svd(flip * A);
    Mat R = svd.U() * svd.V().transpose();
    Vec q = GetAxisAngle(R);

    // Make result
    vnl_vector<double> x_rigid(6);
    x_rigid[0] = q[0]; x_rigid[1] = q[1]; x_rigid[2] = q[2];
    x_rigid[3] = b[0]; x_rigid[4] = b[1]; x_rigid[5] = b[2];

    return x_rigid;
    }

  static Mat GetRandomRotation(vnl_random &randy, double alpha)
  {
    // Generate a random axis of rotation. A triple of Gaussian numbers, normalized to
    // unit length gives a uniform distribution over the sphere
    Vec q_axis;
    for(int d = 0; d < 3; d++)
      q_axis[d] = randy.normal();
    q_axis.normalize();

    // Generate the axis-angle representation of the rotation
    Vec q = q_axis * alpha;

    // Generate a random rotation using given angles
    return GetRotationMatrix(q);
  }

};


template <typename TReal>
class RigidHelper<2, TReal>
{
public:
  typedef vnl_vector_fixed<double, 2> Vec;
  typedef vnl_matrix_fixed<double, 2, 2> Mat;
  typedef vnl_vector<double> CoeffVec;
  typedef vnl_matrix<double> JacMat;

  static unsigned int GetNumberOfCoefficients() { return 3; }

  static Mat GetRotationMatrix(const Vec &q, Mat *d_R = NULL)
  {
    TReal theta = q[0];

    Mat R;
    R(0,0) =  cos(theta); R(0,1) = sin(theta);
    R(1,0) = -sin(theta); R(1,1) = cos(theta);

    if(d_R)
      {
      // Compute the matrices d_Qmat (derivative wrt theta)
      d_R[0](0,0) = -sin(theta); d_R[0](0,1) =  cos(theta);
      d_R[0](1,0) = -cos(theta); d_R[0](1,1) = -sin(theta);
      }

    return R;
  }

  static CoeffVec GetAffineCoefficientsAndJacobian(
      const CoeffVec &x, const Mat &flip, JacMat *jac = NULL)
  {
    // Place parameters into theta and b
    Vec b(x[1], x[2]);

    // Compute the rotation matrix
    Mat d_R;
    Mat R = GetRotationMatrix(x, jac ? &d_R : NULL);

    // Now we have a rotation and a translation, convert to parameters for the affine function
    vnl_vector<double> x_affine(6);
    flatten_affine_transform(flip * R, b, x_affine.data_block());

    if(jac)
      {
      // Create a matrix to hold the jacobian
      jac->set_size(6, 3);
      jac->fill(0.0);

      // Zero vector
      Vec zero_vec; zero_vec.fill(0.0);
      Mat zero_mat; zero_mat.fill(0.0);

      // Fill out the rotation column
      vnl_vector<double> jac_col_theta(6);
      flatten_affine_transform(flip * d_R, zero_vec, jac_col_theta.data_block());
      jac->set_column(0, jac_col_theta);

      // Fill out the translation columns
      for(int p = 0; p < 2; p++)
        {
        // Fill the corresponding column
        vnl_vector<double> jac_col_b(6);
        Vec ep; ep.fill(0.0); ep[p] = 1;
        flatten_affine_transform(zero_mat, ep, jac_col_b.data_block());
        jac->set_column(p+1, jac_col_b);
        }
      }

    return x_affine;
  }

  static Vec GetRotationAngle(const Mat &R)
  {
    Vec x(1);
    x[0] = atan2(R(0,1), R(0,0));
    return x;
  }

  static CoeffVec GetRigidCoefficientsFromAffine(const CoeffVec &x, Mat &flip)
  {
    Mat A; Vec b;
    unflatten_affine_transform(x.data_block(), A, b);

    // If the determinant of A is negative, we need to use the flip
    if(vnl_determinant(A) < 0.0)
      {
      flip(0,0) = -1.0;
      }
    else
      {
      flip(0,0) = 1.0;
      }

    // Compute polar decomposition of the affine matrix
    vnl_svd<double> svd(flip * A);
    Mat R = svd.U() * svd.V().transpose();

    // Use the angle
    double theta = GetRotationAngle(R)[0];

    // Make result
    vnl_vector<double> x_rigid(3);
    x_rigid[0] = theta; x_rigid[1] = b[0]; x_rigid[2] = b[1];
    return x_rigid;
  }

  static Mat GetRandomRotation(vnl_random &randy, double alpha)
  {
    vnl_vector<double> q(1, alpha);
    return GetRotationMatrix(q);
  }
};






template<unsigned int VDim, typename TReal>
void PhysicalAffineToVoxelAffineBlock<VDim, TReal>
::map_phys_to_vox(const CoeffVec &x_phys, CoeffVec &x_vox)
{
  Mat A_phys;
  Vec b_phys;

  // unflatten the input parameters into A and b
  unflatten_affine_transform(x_phys.data_block(), A_phys, b_phys);

  // convert into voxel-space affine transform
  Mat A_vox = Q_mov_inv * A_phys * Q_fix;
  Vec b_vox = Q_mov_inv * (A_phys * b_fix + b_phys) + b_mov_inv;

  // Flatten back
  x_vox.set_size(this->m_Downstream->get_number_of_unknowns());
  flatten_affine_transform(A_vox, b_vox, x_vox.data_block());
}

template<unsigned int VDim, typename TReal>
void
PhysicalAffineToVoxelAffineBlock<VDim, TReal>
::SetImageData(OFHelperType *helper, int level)
{
  // The rigid transformation must be rigid in physical space, not in voxel space
  // So in the constructor, we must compute the mappings from the two spaces
  GetVoxelSpaceToNiftiSpaceTransform(helper->GetReferenceSpace(level), Q_fix, b_fix);
  GetVoxelSpaceToNiftiSpaceTransform(helper->GetMovingReferenceSpace(level), Q_mov, b_mov);

  // Compute the inverse transformations
  Q_fix_inv = vnl_matrix_inverse<double>(Q_fix);
  b_fix_inv = - Q_fix_inv * b_fix;

  Q_mov_inv = vnl_matrix_inverse<double>(Q_mov);
  b_mov_inv = - Q_mov_inv * b_mov;

  // Take advantage of the fact that the transformation is linear in A and b to compute
  // the Jacobian of the transformation ahead of time, and "lazily", using finite differences
  int n = VDim * (VDim + 1);
  J_phys_vox.set_size(n, n);
  vnl_vector<double> x_phys(n, 0), x_vox_0(n), x_vox(n);

  // Voxel parameter vector corresponding to zero transform
  this->map_phys_to_vox(x_phys, x_vox_0);

  // Compute each column of the jacobian
  for(int i = 0; i < n; i++)
    {
    x_phys.fill(0);
    x_phys[i] = 1;
    this->map_phys_to_vox(x_phys, x_vox);
    J_phys_vox.set_column(i, x_vox - x_vox_0);
    }
}

template<unsigned int VDim, typename TReal>
void
PhysicalAffineToVoxelAffineBlock<VDim, TReal>
::compute(const CoeffVec &x, double *f, CoeffVec *g)
{
  // Map to voxel space
  CoeffVec x_vox(this->m_Downstream->get_number_of_unknowns());
  this->map_phys_to_vox(x, x_vox);

  // Do we need the gradient?
  if(g)
    {
    // Compute the function and gradient wrt voxel parameters
    vnl_vector<double> g_vox(this->m_Downstream->get_number_of_unknowns());
    this->m_Downstream->compute(x_vox, f, &g_vox);

    // Transform voxel-space gradient into physical-space gradient
    *g = J_phys_vox.transpose() * g_vox;
    }
  else
    {
    // Just compute the function
    this->m_Downstream->compute(x_vox, f, NULL);
    }
}

template<unsigned int VDim, typename TReal>
void
PhysicalAffineToVoxelAffineBlock<VDim, TReal>
::GetTransform(const CoeffVec &x, Mat &A, Vec &b)
{
  CoeffVec x_vox(this->m_Downstream->get_number_of_unknowns());
  this->map_phys_to_vox(x, x_vox);
  this->m_Downstream->GetTransform(x_vox, A, b);
}


template<unsigned int VDim, typename TReal>
typename PhysicalAffineToVoxelAffineBlock<VDim, TReal>::CoeffVec
PhysicalAffineToVoxelAffineBlock<VDim, TReal>
::GetCoefficients(const Mat &A, const Vec &b)
{
  // Call the downstream block and let it provide its coefficients
  CoeffVec vox_coeff = this->m_Downstream->GetCoefficients(A, b);

  // Unpack these coefficients
  Mat A_vox, A_phys;
  Vec b_vox, b_phys;
  unflatten_affine_transform(vox_coeff.data_block(), A_vox, b_vox);

  // convert into physical-space affine transform
  A_phys = Q_mov * A_vox * Q_fix_inv;
  b_phys = Q_mov * (b_vox - b_mov_inv) - A_phys * b_fix;

  // Flatten
  CoeffVec phys_coeff(this->GetOutputSize());
  flatten_affine_transform(A_phys, b_phys, phys_coeff.data_block());
  return phys_coeff;
}

template <unsigned int VDim, typename TReal>
typename PhysicalAffineToVoxelAffineBlock<VDim, TReal>::CoeffVec
PhysicalAffineToVoxelAffineBlock<VDim, TReal>
::GetOptimalParameterScaling(const itk::Size<VDim> &image_dim)
{
  // Initialize the scaling vector
  vnl_vector<double> scaling(this->get_number_of_unknowns());

  // Set the scaling of the parameters based on image dimensions. This makes it
  // possible to set tolerances in units of voxels. The order of change in the
  // parameters is comparable to the displacement of any point inside the image
  Mat A; Vec b;

  for(int i = 0; i < VDim; i++)
    {
    b[i] = 1.0;
    for(int j = 0; j < VDim; j++)
      A(i, j) = image_dim[j];
    }

  flatten_affine_transform(A, b, scaling.data_block());
  return scaling;
}




#define greedy_template_inst(ClassName) \
  template class ClassName<2, float>; \
  template class ClassName<3, float>; \
  template class ClassName<4, float>; \
  template class ClassName<2, double>; \
  template class ClassName<3, double>; \
  template class ClassName<4, double>;



template<unsigned int VDim, typename TReal>
RigidBlock<VDim, TReal>::RigidBlock()
{
  typedef RigidHelper<VDim,TReal> Helper;
  this->set_number_of_unknowns(Helper::GetNumberOfCoefficients());
}


template<unsigned int VDim, typename TReal>
void
RigidBlock<VDim, TReal>
::compute(const RigidBlock::CoeffVec &x, double *f, RigidBlock::CoeffVec *g)
{
  typedef RigidHelper<VDim,TReal> Helper;

  if(g)
    {
    // Compute coefficients and Jacobian
    JacMat jac;
    CoeffVec x_affine = Helper::GetAffineCoefficientsAndJacobian(x, m_Flip, &jac);

    // Create a vector to store the affine gradient
    CoeffVec g_affine(this->m_Downstream->get_number_of_unknowns());
    this->m_Downstream->compute(x_affine, f, &g_affine);

    // Multiply the gradient by the jacobian
    *g = jac.transpose() * g_affine;
    }
  else
    {
    CoeffVec x_affine = Helper::GetAffineCoefficientsAndJacobian(x, m_Flip);
    this->m_Downstream->compute(x_affine, f, NULL);
    }
}

template<unsigned int VDim, typename TReal>
void
RigidBlock<VDim, TReal>
::GetTransform(const CoeffVec &x, Mat &A, Vec &b)
{
  typedef RigidHelper<VDim,TReal> Helper;
  CoeffVec x_affine = Helper::GetAffineCoefficientsAndJacobian(x, m_Flip);
  this->m_Downstream->GetTransform(x_affine, A, b);
}


template<unsigned int VDim, typename TReal>
typename RigidBlock<VDim, TReal>::CoeffVec
RigidBlock<VDim, TReal>
::GetCoefficients(const Mat &A, const Vec &b)
{
  typedef RigidHelper<VDim,TReal> Helper;

  // Get the coefficients from the downstream block
  CoeffVec x_aff = this->m_Downstream->GetCoefficients(A, b);
  CoeffVec x_rigid = Helper::GetRigidCoefficientsFromAffine(x_aff, m_Flip);

  // Use helper to derive rotation parameters
  return x_rigid;
}

template<unsigned int VDim, typename TReal>
typename RigidBlock<VDim, TReal>::CoeffVec
RigidBlock<VDim, TReal>
::GetOptimalParameterScaling(const itk::Size<VDim> &image_dim)
{
  // Initialize the scaling vector
  vnl_vector<double> scaling(this->get_number_of_unknowns());

  // Scaling is harder for rotations. The rotation parameters are in units of
  // radians. We must figure out how many radians are equivalent to a point in
  // the image moving by a single voxel. That actually works out to be 1/dim.

  // So we take the average of the image dimensions and use that as scaling
  double mean_dim = 0;
  for(int i = 0; i < VDim; i++)
    mean_dim += image_dim[i] / VDim;

  // Fill the scaling vector with mean_dim, then fill the last VDim fields
  // with 1 (those are offsets)
  scaling.fill(mean_dim);
  scaling.update(CoeffVec(VDim, 1.0), scaling.size() - VDim);
  return scaling;
}

template<unsigned int VDim, typename TReal>
typename RigidBlock<VDim, TReal>::Mat
RigidBlock<VDim, TReal>
::GetRandomRotation(vnl_random &randy, double alpha)
{
  typedef RigidHelper<VDim,TReal> Helper;
  return Helper::GetRandomRotation(randy, alpha);
}


template<unsigned int VDim, typename TReal>
void
AffineBInverseABlock<VDim, TReal>
::GetTransform(const CoeffVec &x, Mat &A_out, Vec &b_out)
{
  // Split x into two parts
  unsigned int n = this->m_Downstream->get_number_of_unknowns();
  vnl_vector<double> xA = x.extract(n), gA(n,0.0), xB = x.extract(n,n), gB(n, 0.0);
  vnl_vector<double> xM(n, 0.0), gM(n, 0.0);

  // Compute the corresponding transforms (Ax+p) and (Bx+q)
  Mat A, B; Vec p, q;
  unflatten_affine_transform(xA.data_block(), A, p);
  unflatten_affine_transform(xB.data_block(), B, q);

  // Compute the combined transformation (Mx+z)
  Mat B_inv = vnl_matrix_inverse<double>(B).inverse();
  Mat M = B_inv * A;
  Vec z = B_inv * (p-q);
  flatten_affine_transform(M, z, xM.data_block());

  // Pass this on to the downstream transform
  this->m_Downstream->GetTransform(xM, A_out, b_out);
}

template<unsigned int VDim, typename TReal>
void AffineBInverseABlock<VDim, TReal>
::compute(const vnl_vector<double> &x, double *f, vnl_vector<double> *g)
{
  // Split x into two parts
  unsigned int n = this->m_Downstream->get_number_of_unknowns();
  vnl_vector<double> xA = x.extract(n), gA(n,0.0), xB = x.extract(n,n), gB(n, 0.0);
  vnl_vector<double> xM(n, 0.0), gM(n, 0.0);

  // Compute the corresponding transforms (Ax+p) and (Bx+q)
  Mat A, B; Vec p, q;
  unflatten_affine_transform(xA.data_block(), A, p);
  unflatten_affine_transform(xB.data_block(), B, q);

  // Compute the combined transformation (Mx+z)
  Mat B_inv = vnl_matrix_inverse<double>(B).inverse();
  Mat M = B_inv * A;
  Vec z = B_inv * (p-q);
  flatten_affine_transform(M, z, xM.data_block());

  // Pass this on to the downstream transform
  this->m_Downstream->compute(xM, f, g ? &gM : NULL);
  if(g)
    {
    // Get the derivative of loss w.r.t. M and z
    Mat d_M; Vec d_z;
    unflatten_affine_transform(gM.data_block(), d_M, d_z);

    // Get the derivatives of A, B, p and q (TODO: are these backwards?)
    Mat d_A = B_inv * d_M;
    Vec d_p = B_inv * d_z;

    // I think this is right, but we have to check!
    Mat d_B = - (B_inv * d_M) * M - B_inv * outer_product(d_z, z);
    Vec d_q = -B_inv * d_z;

    flatten_affine_transform(d_A, d_p, gA.data_block());
    flatten_affine_transform(d_B, d_q, gB.data_block());
    g->update(gA, 0);
    g->update(gB, n);
    }
}

// greedy_template_inst(PureAffineCostFunction)


template<unsigned int VDim, typename TReal>
void
AffineInVoxelSpaceBlock<VDim, TReal>
::SetImageData(const GreedyParameters *param, ParentType *parent,
               int level, OFHelperType *helper)
{
  // Store the data
  m_Param = param;
  m_OFHelper = helper;
  m_Level = level;
  m_Parent = parent;

  // Allocate the working images, but do not allocate. We will allocate on demand because
  // these affine cost functions may be created without needing to do any computation
  m_Allocated = false;

  m_Phi = VectorImageType::New();
  m_Phi->CopyInformation(helper->GetReferenceSpace(level));
  m_Phi->SetRegions(helper->GetReferenceSpace(level)->GetBufferedRegion());

  m_GradMetric = VectorImageType::New();
  m_GradMetric->CopyInformation(helper->GetReferenceSpace(level));
  m_GradMetric->SetRegions(helper->GetReferenceSpace(level)->GetBufferedRegion());

  m_GradMask = VectorImageType::New();
  m_GradMask->CopyInformation(helper->GetReferenceSpace(level));
  m_GradMask->SetRegions(helper->GetReferenceSpace(level)->GetBufferedRegion());

  m_Metric = ImageType::New();
  m_Metric->CopyInformation(helper->GetReferenceSpace(level));
  m_Metric->SetRegions(helper->GetReferenceSpace(level)->GetBufferedRegion());

  m_Mask = ImageType::New();
  m_Mask->CopyInformation(helper->GetReferenceSpace(level));
  m_Mask->SetRegions(helper->GetReferenceSpace(level)->GetBufferedRegion());
}

template<unsigned int VDim, typename TReal>
void AffineInVoxelSpaceBlock<VDim, TReal>
::compute(const vnl_vector<double> &x, double *f, vnl_vector<double> *g)
{
  // Form a matrix/vector from x
  typename LinearTransformType::Pointer tran = LinearTransformType::New();

  // Set the components of the transform
  unflatten_affine_transform(x.data_block(), tran.GetPointer());

  // Allocate a vector to hold the per-component metric values
  vnl_vector<double> comp_metric;

  // Allocate the memory if needed
  if(!m_Allocated)
    {
    m_Phi->Allocate();
    m_GradMetric->Allocate();
    m_GradMask->Allocate();
    m_Metric->Allocate();
    m_Mask->Allocate();
    m_Allocated = true;
    }

  // Compute the gradient
  double val = 0.0;

  // The scaling of the metric. For some metrics, we need to change sign (to minimize) and also
  // it is more readable if it is scaled by some large factor
  double metric_scale =
      (m_Param->metric == GreedyParameters::NCC
       || m_Param->metric == GreedyParameters::MI
       || m_Param->metric == GreedyParameters::NMI)
      ? -10000.0 : 1.0;

  // The output metric report
  MultiComponentMetricReport out_metric;

  // Gradient output
  typename LinearTransformType::Pointer grad;
  if(g)
    grad = LinearTransformType::New();

  // Perform actual metric computation
  if(m_Param->metric == GreedyParameters::SSD)
    {
    m_OFHelper->ComputeAffineMSDMatchAndGradient(
          m_Level, tran, m_Metric, m_Mask, m_GradMetric, m_GradMask, m_Phi, out_metric, grad);

    }
  else if(m_Param->metric == GreedyParameters::NCC)
    {
    m_OFHelper->ComputeAffineNCCMatchAndGradient(
          m_Level, tran, array_caster<VDim>::to_itkSize(m_Param->metric_radius),
          m_Metric, m_Mask, m_GradMetric, m_GradMask, m_Phi, out_metric, grad);
    }
  else if(m_Param->metric == GreedyParameters::MI || m_Param->metric == GreedyParameters::NMI)
    {
    m_OFHelper->ComputeAffineMIMatchAndGradient(
          m_Level, m_Param->metric == GreedyParameters::NMI,
          tran, m_Metric, m_Mask, m_GradMetric, m_GradMask, m_Phi, out_metric, grad);
    }

  // Handle the gradient
  if(g)
    {
    flatten_affine_transform(grad.GetPointer(), g->data_block());
    (*g) *= metric_scale;
    }

  // Scale the output metric
  out_metric.Scale(metric_scale);

  // Report the output values
  if(f)
    *f = out_metric.TotalMetric;

  // Has the metric improved?
  if(m_Parent->GetMetricLog().size())
    {
    const std::vector<MultiComponentMetricReport> &log = m_Parent->GetMetricLog().back();
    if(log.size() == 0 || log.back().TotalMetric > out_metric.TotalMetric)
      {
      // Record the metric value
      m_Parent->RecordMetricValue(out_metric);

      // Write out the current iteration transform
      if(m_Param->output_intermediate.length())
        {
        vnl_matrix<double> Q_physical = ParentType::MapAffineToPhysicalRASSpace(*m_OFHelper, m_Level, tran);
        m_Parent->WriteAffineMatrixViaCache(m_Param->output_intermediate, Q_physical);
        }
      }
    }
}


template<unsigned int VDim, typename TReal>
void
AffineInVoxelSpaceBlock<VDim, TReal>
::GetTransform(const CoeffVec &x, Mat &A, Vec &b)
{
  unflatten_affine_transform(x.data_block(), A, b);
}


template<unsigned int VDim, typename TReal>
typename AffineInVoxelSpaceBlock<VDim, TReal>::CoeffVec
AffineInVoxelSpaceBlock<VDim, TReal>
::GetCoefficients(const Mat &A, const Vec &b)
{
  vnl_vector<double> x(this->get_number_of_unknowns());
  flatten_affine_transform(A, b, x.data_block());
  return x;
}


template<unsigned int VDim, typename TReal>
void
CoefficientScalingBlock<VDim, TReal>
::compute(const vnl_vector<double> &x, double *f, vnl_vector<double> *g)
{
  // Scale the parameters so they are in unscaled units
  vnl_vector<double> x_scaled = element_quotient(x, m_Scaling);

  // Call the wrapped method
  if(g)
    {
    vnl_vector<double> g_scaled(x_scaled.size());
    this->m_Downstream->compute(x_scaled, f, &g_scaled);
    *g = element_quotient(g_scaled, m_Scaling);
    }
  else
    {
    this->m_Downstream->compute(x_scaled, f, g);
    }
}

template<unsigned int VDim, typename TReal>
void
CoefficientScalingBlock<VDim, TReal>
::GetTransform(const CoeffVec &x, Mat &A, Vec &b)
{
  vnl_vector<double> x_scaled = element_quotient(x, m_Scaling);
  this->m_Downstream->GetTransform(x_scaled, A, b);
}

template<unsigned int VDim, typename TReal>
typename CoefficientScalingBlock<VDim, TReal>::CoeffVec
CoefficientScalingBlock<VDim, TReal>
::GetCoefficients(const Mat &A, const Vec &b)
{
  CoeffVec x = this->m_Downstream->GetCoefficients(A, b);
  return element_product(x, m_Scaling);
}

greedy_template_inst(AffineInVoxelSpaceBlock)
greedy_template_inst(PhysicalAffineToVoxelAffineBlock)
greedy_template_inst(RigidBlock)
greedy_template_inst(AffineBInverseABlock)
greedy_template_inst(CoefficientScalingBlock)
