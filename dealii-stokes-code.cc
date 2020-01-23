/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2016 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Timo Heister, Clemson University, 2016
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>


#include <deal.II/lac/generic_linear_algebra.h>

/* #define FORCE_USE_OF_TRILINOS */

namespace LA
{
using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_qmrs.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>




#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>



#include <cmath>
#include <fstream>
#include <iostream>

namespace Nsinker
{
using namespace dealii;

class QuietException {};



void average(std::vector<double> &values, const std::string averaging_type)
{
  const unsigned int n_q_points = values.size();

  if (averaging_type == "harmonic")
  {
    // harmonic mean:
    double alpha = 0.0;
    for (unsigned int q=0; q<n_q_points; ++q)
      alpha += 1./values[q];
    alpha = 1./(alpha/n_q_points);

    // write back:
    for (unsigned int q=0; q<n_q_points; ++q)
      values[q] = alpha;
  }
  else if (averaging_type == "arithmetic")
  {
    // arithmetic mean:
    double alpha = 0.0;
    for (unsigned int q=0; q<n_q_points; ++q)
      alpha += values[q];
    alpha /= n_q_points;

    // write back:
    for (unsigned int q=0; q<n_q_points; ++q)
      values[q] = alpha;
  }
  else if (averaging_type == "geometric")
  {
    // geometric mean:
    double alpha = 1.0;
    for (unsigned int q=0; q<n_q_points; ++q)
      alpha *= values[q];
    alpha = std::pow(alpha, 1.0/n_q_points);

    // write back:
    for (unsigned int q=0; q<n_q_points; ++q)
      values[q] = alpha;
  }
  else if (averaging_type == "max")
  {
    // find max:
    double alpha = 0.0;
    for (unsigned int q=0; q<n_q_points; ++q)
      if (alpha < values[q])
        alpha = values[q];

    // write back:
    for (unsigned int q=0; q<n_q_points; ++q)
      values[q] = alpha;
  }
  else if (averaging_type == "none")
  {
    // do not average
  }
  else
  {
    std::cout << "Avg Type:   " << averaging_type << std::endl;
    AssertThrow(false, ExcMessage("Averaging type "+averaging_type+" is not yet implemented"));
  }
}


namespace ChangeVectorTypes
{
void import(TrilinosWrappers::MPI::Vector &out,
            const dealii::LinearAlgebra::ReadWriteVector<double> &rwv,
            const VectorOperation::values                 operation)
{
  Assert(out.size() == rwv.size(),
         ExcMessage("Both vectors need to have the same size for import() to work!"));

  Assert(out.locally_owned_elements() == rwv.get_stored_elements(),
         ExcNotImplemented());

  if (operation == VectorOperation::insert)
  {
    for (const auto idx : out.locally_owned_elements())
      out[idx] = rwv[idx];
  }
  else if (operation == VectorOperation::add)
  {
    for (const auto idx : out.locally_owned_elements())
      out[idx] += rwv[idx];
  }
  else
    AssertThrow(false, ExcNotImplemented());

  out.compress(operation);
}


void copy(TrilinosWrappers::MPI::Vector &out,
          const dealii::LinearAlgebra::distributed::Vector<double> &in)
{
  dealii::LinearAlgebra::ReadWriteVector<double> rwv(out.locally_owned_elements());
  rwv.import(in, VectorOperation::insert);
  //This import function doesn't exist until after dealii 9.0
  //Implemented above
  import(out, rwv,VectorOperation::insert);
}

void copy(dealii::LinearAlgebra::distributed::Vector<double> &out,
          const TrilinosWrappers::MPI::Vector &in)
{
  dealii::LinearAlgebra::ReadWriteVector<double> rwv;
  rwv.reinit(in);
  out.import(rwv, VectorOperation::insert);
}

void copy(TrilinosWrappers::MPI::BlockVector &out,
          const dealii::LinearAlgebra::distributed::BlockVector<double> &in)
{
  const unsigned int n_blocks = in.n_blocks();
  for (unsigned int b=0; b<n_blocks; ++b)
    copy(out.block(b),in.block(b));
}

void copy(dealii::LinearAlgebra::distributed::BlockVector<double> &out,
          const TrilinosWrappers::MPI::BlockVector &in)
{
  const unsigned int n_blocks = in.n_blocks();
  for (unsigned int b=0; b<n_blocks; ++b)
    copy(out.block(b),in.block(b));
}
}


/**
 * Implement the block Schur preconditioner for the Stokes system.
 */
template <class ABlockMatrixType, class StokesMatrixType, class MassMatrixType, class PreconditionerMp,class PreconditionerA>
class BlockSchurGMGPreconditioner : public Subscriptor
{
public:
  /**
     * @brief Constructor
     *
     * @param S The entire Stokes matrix
     * @param Spre The matrix whose blocks are used in the definition of
     *     the preconditioning of the Stokes matrix, i.e. containing approximations
     *     of the A and S blocks.
     * @param Mppreconditioner Preconditioner object for the Schur complement,
     *     typically chosen as the mass matrix.
     * @param Apreconditioner Preconditioner object for the matrix A.
     * @param do_solve_A A flag indicating whether we should actually solve with
     *     the matrix $A$, or only apply one preconditioner step with it.
     * @param A_block_tolerance The tolerance for the CG solver which computes
     *     the inverse of the A block.
     * @param S_block_tolerance The tolerance for the CG solver which computes
     *     the inverse of the S block (Schur complement matrix).
     */
  BlockSchurGMGPreconditioner (const StokesMatrixType  &S,
                               const ABlockMatrixType  &A,
                               const MassMatrixType  &Mass,
                               const PreconditionerMp                     &Mppreconditioner,
                               const PreconditionerA                      &Apreconditioner,
                               const bool do_mass_solve,
                               const bool                                  do_solve_A,
                               const double                                A_block_tolerance,
                               const double                                S_block_tolerance);

  /**
     * Matrix vector product with this preconditioner object.
     */
  void vmult (dealii::LinearAlgebra::distributed::BlockVector<double>       &dst,
              const dealii::LinearAlgebra::distributed::BlockVector<double> &src) const;

  unsigned int n_iterations_A() const;
  unsigned int n_iterations_S() const;


private:
  /**
     * References to the various matrix object this preconditioner works on.
     */
  const StokesMatrixType &stokes_matrix;
  const ABlockMatrixType &velocity_matrix;
  const MassMatrixType &mass_matrix;
  const PreconditionerMp                    &mp_preconditioner;
  const PreconditionerA                     &a_preconditioner;

  /**
     * Whether to actually invert the $\tilde A$ part of the preconditioner matrix
     * or to just apply a single preconditioner step with it.
     */
  const bool do_mass_solve;
  const bool do_solve_A;
  mutable unsigned int n_iterations_A_;
  mutable unsigned int n_iterations_S_;
  const double A_block_tolerance;
  const double S_block_tolerance;
};

template <class ABlockMatrixType, class StokesMatrixType, class MassMatrixType, class PreconditionerMp,class PreconditionerA>
BlockSchurGMGPreconditioner<ABlockMatrixType, StokesMatrixType, MassMatrixType, PreconditionerMp, PreconditionerA>::
BlockSchurGMGPreconditioner (const StokesMatrixType  &S,
                             const ABlockMatrixType  &A,
                             const MassMatrixType  &Mass,
                             const PreconditionerMp                     &Mppreconditioner,
                             const PreconditionerA                      &Apreconditioner,
                             const bool do_mass_solve,
                             const bool                                  do_solve_A,
                             const double                                A_block_tolerance,
                             const double                                S_block_tolerance)
  :
    stokes_matrix     (S),
    velocity_matrix   (A),
    mass_matrix     (Mass),
    mp_preconditioner (Mppreconditioner),
    a_preconditioner  (Apreconditioner),
    do_mass_solve (do_mass_solve),
    do_solve_A        (do_solve_A),
    n_iterations_A_(0),
    n_iterations_S_(0),
    A_block_tolerance(A_block_tolerance),
    S_block_tolerance(S_block_tolerance)
{}

template <class ABlockMatrixType, class StokesMatrixType, class MassMatrixType, class PreconditionerMp,class PreconditionerA>
unsigned int
BlockSchurGMGPreconditioner<ABlockMatrixType, StokesMatrixType, MassMatrixType, PreconditionerMp, PreconditionerA>::
n_iterations_A() const
{
  return n_iterations_A_;
}

template <class ABlockMatrixType, class StokesMatrixType, class MassMatrixType, class PreconditionerMp,class PreconditionerA>
unsigned int
BlockSchurGMGPreconditioner<ABlockMatrixType, StokesMatrixType, MassMatrixType, PreconditionerMp, PreconditionerA>::
n_iterations_S() const
{
  return n_iterations_S_;
}

template <class ABlockMatrixType, class StokesMatrixType, class MassMatrixType, class PreconditionerMp,class PreconditionerA>
void
BlockSchurGMGPreconditioner<ABlockMatrixType, StokesMatrixType, MassMatrixType, PreconditionerMp, PreconditionerA>::
vmult (dealii::LinearAlgebra::distributed::BlockVector<double>       &dst,
       const dealii::LinearAlgebra::distributed::BlockVector<double>  &src) const
{
  dealii::LinearAlgebra::distributed::BlockVector<double> utmp(src);
  dealii::LinearAlgebra::distributed::BlockVector<double> ptmp(src);

  if (do_mass_solve)
  {
    // first solve with the bottom left block, which we have built
    // as a mass matrix with the inverse of the viscosity
    SolverControl solver_control(100, src.block(1).l2_norm() * S_block_tolerance,true);

    SolverCG<dealii::LinearAlgebra::distributed::Vector<double> > solver(solver_control);
    // Trilinos reports a breakdown
    // in case src=dst=0, even
    // though it should return
    // convergence without
    // iterating. We simply skip
    // solving in this case.
    if (src.block(1).l2_norm() > 1e-50)
    {
      try
      {
        dst.block(1) = 0.0;
        solver.solve(mass_matrix,
                     dst.block(1), src.block(1),
                     mp_preconditioner);
        n_iterations_S_ += solver_control.last_step();
      }
      // if the solver fails, report the error from processor 0 with some additional
      // information about its location, and throw a quiet exception on all other
      // processors
      catch (const std::exception &exc)
      {
        if (Utilities::MPI::this_mpi_process(src.block(0).get_mpi_communicator()) == 0)
          AssertThrow (false,
                       ExcMessage (std::string("The iterative (bottom right) solver in BlockSchurGMGPreconditioner::vmult "
                                               "did not converge to a tolerance of "
                                               + Utilities::to_string(solver_control.tolerance()) +
                                               ". It reported the following error:\n\n")
                                   +
                                   exc.what()))
              else
              throw QuietException();
      }
    }
  }
  else
  {
    mp_preconditioner.vmult(dst.block(1),src.block(1));
    n_iterations_S_ += 1;
  }

  dst.block(1) *= -1.0;

  {
    ptmp = dst;
    ptmp.block(0) = 0.0;
    stokes_matrix.vmult(utmp, ptmp); // B^T
    utmp.block(0) *= -1.0;
    utmp.block(0) += src.block(0);
  }

  // now either solve with the top left block (if do_solve_A==true)
  // or just apply one preconditioner sweep (for the first few
  // iterations of our two-stage outer GMRES iteration)
  if (do_solve_A == true)
  {
    SolverControl solver_control(1000, utmp.block(0).l2_norm() * A_block_tolerance);
    SolverCG<dealii::LinearAlgebra::distributed::Vector<double>> solver(solver_control);
    try
    {
      dst.block(0) = 0.0;
      solver.solve(velocity_matrix, dst.block(0), utmp.block(0),
                   a_preconditioner);
      n_iterations_A_ += solver_control.last_step();
    }
    // if the solver fails, report the error from processor 0 with some additional
    // information about its location, and throw a quiet exception on all other
    // processors
    catch (const std::exception &exc)
    {
      if (Utilities::MPI::this_mpi_process(src.block(0).get_mpi_communicator()) == 0)
        AssertThrow (false,
                     ExcMessage (std::string("The iterative (top left) solver in BlockSchurGMGPreconditioner::vmult "
                                             "did not converge to a tolerance of "
                                             + Utilities::to_string(solver_control.tolerance()) +
                                             ". It reported the following error:\n\n")
                                 +
                                 exc.what()))
            else
            throw QuietException();
    }

  }
  else
  {
    //          if (Utilities::MPI::this_mpi_process(src.block(0).get_mpi_communicator()) == 0)
    //std::cout << "vmult" << std::endl;
    a_preconditioner.vmult (dst.block(0), utmp.block(0));
    n_iterations_A_ += 1;
  }
}



/**
 * This namespace contains all matrix-free operators used in the Stokes solver.
 */
namespace MatrixFreeStokesOperators
{
/**
   * Operator for the entire Stokes block.
   */
template <int dim, int degree_v, typename number>
class StokesOperator
    : public MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number> >
{
public:

  /**
       * Constructor.
       */
  StokesOperator ();

  /**
       * Reset object.
       */
  void clear () override;

  /**
       * Fills in the viscosity table, sets the value for the pressure scaling constant,
       * and gives information regarding compressibility.
       */
  void fill_cell_data(const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                      const double pressure_scaling,
                      const Triangulation<dim> &tria,
                      const DoFHandler<dim> &dof_handler_for_projection,
                      const bool is_compressible);

  /**
       * Returns the viscosity table.
       */
  const Table<1, VectorizedArray<number> > &
  get_viscosity_x_2_table();

  /**
       * Computes the diagonal of the matrix. Since matrix-free operators have not access
       * to matrix elements, we must apply the matrix-free operator to the unit vectors to
       * recover the diagonal.
       */
  void compute_diagonal () override;

private:

  /**
       * Performs the application of the matrix-free operator. This function is called by
       * vmult() functions MatrixFreeOperators::Base.
       */
  void apply_add (dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
                  const dealii::LinearAlgebra::distributed::BlockVector<number> &src) const override;

  /**
       * Defines the application of the cell matrix.
       */
  void local_apply (const dealii::MatrixFree<dim, number> &data,
                    dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
                    const dealii::LinearAlgebra::distributed::BlockVector<number> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range) const;

  /**
       * Table which stores a viscosity value for each cell.
       */
  Table<1, VectorizedArray<number> > viscosity_x_2;

  /**
       * Pressure scaling constant.
       */
  double pressure_scaling;

  /**
        * Information on the compressibility of the flow.
        */
  bool is_compressible;
};

/**
   * Operator for the pressure mass matrix used in the block preconditioner
   */
template <int dim, int degree_p, typename number>
class MassMatrixOperator
    : public MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>
{
public:

  /**
       * Constructor
       */
  MassMatrixOperator ();

  /**
       * Reset the object.
       */
  void clear () override;

  /**
       * Fills in the viscosity table and sets the value for the pressure scaling constant.
       */
  void fill_cell_data (const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                       const double pressure_scaling,
                       const Triangulation<dim> &tria,
                       const DoFHandler<dim> &dof_handler_for_projection,
                       const bool for_mg);


  /**
       * Computes the diagonal of the matrix. Since matrix-free operators have not access
       * to matrix elements, we must apply the matrix-free operator to the unit vectors to
       * recover the diagonal.
       */
  void compute_diagonal () override;

private:

  /**
       * Performs the application of the matrix-free operator. This function is called by
       * vmult() functions MatrixFreeOperators::Base.
       */
  void apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
                  const dealii::LinearAlgebra::distributed::Vector<number> &src) const override;

  /**
       * Defines the application of the cell matrix.
       */
  void local_apply (const dealii::MatrixFree<dim, number> &data,
                    dealii::LinearAlgebra::distributed::Vector<number> &dst,
                    const dealii::LinearAlgebra::distributed::Vector<number> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range) const;


  /**
       * Computes the diagonal contribution from a cell matrix.
       */
  void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                               dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                               const unsigned int                               &dummy,
                               const std::pair<unsigned int,unsigned int>       &cell_range) const;

  /**
       * Table which stores a viscosity value for each cell.
       */
  Table<1, VectorizedArray<number> > one_over_viscosity;

  /**
       * Pressure scaling constant.
       */
  double pressure_scaling;
};

/**
   * Operator for the A block of the Stokes matrix. The same class is used for both
   * active and level mesh operators.
   */
template <int dim, int degree_v, typename number>
class ABlockOperator
    : public MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>
{
public:

  /**
       * Constructor
       */
  ABlockOperator ();

  /**
       * Reset the operator.
       */
  void clear () override;

  /**
       * Fills in the viscosity table and gives information regarding compressibility.
       */
  void fill_cell_data(const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                      const Triangulation<dim> &tria,
                      const DoFHandler<dim> &dof_handler_for_projection,
                      const bool for_mg,
                      const bool is_compressible);

  /**
       * Computes the diagonal of the matrix. Since matrix-free operators have not access
       * to matrix elements, we must apply the matrix-free operator to the unit vectors to
       * recover the diagonal.
       */
  void compute_diagonal () override;

  /**
       * Manually set the diagonal inside the matrix-free object. This function is needed
       * when using tangential constraints as the function compute_diagonal() cannot handle
       * non-Dirichlet boundary conditions.
       */
  void set_diagonal (const dealii::LinearAlgebra::distributed::Vector<number> &diag);

private:

  /**
       * Performs the application of the matrix-free operator. This function is called by
       * vmult() functions MatrixFreeOperators::Base.
       */
  void apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
                  const dealii::LinearAlgebra::distributed::Vector<number> &src) const override;

  /**
       * Defines the application of the cell matrix.
       */
  void local_apply (const dealii::MatrixFree<dim, number> &data,
                    dealii::LinearAlgebra::distributed::Vector<number> &dst,
                    const dealii::LinearAlgebra::distributed::Vector<number> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range) const;

  /**
       * Computes the diagonal contribution from a cell matrix.
       */
  void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                               dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                               const unsigned int                               &dummy,
                               const std::pair<unsigned int,unsigned int>       &cell_range) const;

  /**
       * Table which stores a viscosity value for each cell.
       */
  Table<1, VectorizedArray<number> > viscosity_x_2;

  /**
        * Information on the compressibility of the flow.
        */
  bool is_compressible;

};
}


/**
 * Implementation of the matrix-free operators.
 *
 * Stokes operator
 */
template <int dim, int degree_v, typename number>
MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>::StokesOperator ()
  :
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number> >()
{}

template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>::clear ()
{
  viscosity_x_2.reinit(TableIndices<1>(0));
  MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::BlockVector<number> >::clear();
}

template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>::
fill_cell_data (const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                const double pressure_scaling,
                const Triangulation<dim> &tria,
                const DoFHandler<dim> &dof_handler_for_projection,
                const bool is_compressible)
{
  const unsigned int n_cells = this->data->n_macro_cells();
  viscosity_x_2.reinit(TableIndices<1>(n_cells));

  std::vector<types::global_dof_index> local_dof_indices(dof_handler_for_projection.get_fe().dofs_per_cell);
  for (unsigned int cell=0; cell<n_cells; ++cell)
    for (unsigned int i=0; i<this->get_matrix_free()->n_components_filled(cell); ++i)
    {
      typename DoFHandler<dim>::active_cell_iterator FEQ_cell = this->get_matrix_free()->get_cell_iterator(cell,i);
      typename DoFHandler<dim>::active_cell_iterator DG_cell(&tria,
                                                             FEQ_cell->level(),
                                                             FEQ_cell->index(),
                                                             &dof_handler_for_projection);
      DG_cell->get_active_or_mg_dof_indices(local_dof_indices);

      //TODO: projection with higher degree
      Assert(local_dof_indices.size() == 1, ExcNotImplemented());
      viscosity_x_2(cell)[i] = 2.0*viscosity_values(local_dof_indices[0]);
    }

  this->pressure_scaling = pressure_scaling;
  this->is_compressible = is_compressible;
}

template <int dim, int degree_v, typename number>
const Table<1, VectorizedArray<number> > &
MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>::get_viscosity_x_2_table()
{
  return viscosity_x_2;
}

template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>
::compute_diagonal ()
{
  // There is currently no need in the code for the diagonal of the entire stokes
  // block. If needed, one could easily construct based on the diagonal of the A
  // block and append zeros to the end for the number of pressure DoFs.
  Assert(false, ExcNotImplemented());
}

template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>
::local_apply (const dealii::MatrixFree<dim, number>                 &data,
               dealii::LinearAlgebra::distributed::BlockVector<number>       &dst,
               const dealii::LinearAlgebra::distributed::BlockVector<number> &src,
               const std::pair<unsigned int, unsigned int>           &cell_range) const
{
  FEEvaluation<dim,degree_v,degree_v+1,dim,number> velocity (data, 0);
  FEEvaluation<dim,degree_v-1,  degree_v+1,1,  number> pressure (data, 1);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    const VectorizedArray<number> &cell_viscosity_x_2 = viscosity_x_2(cell);

    velocity.reinit (cell);
    velocity.read_dof_values (src.block(0));
    velocity.evaluate (false,true,false);
    pressure.reinit (cell);
    pressure.read_dof_values (src.block(1));
    pressure.evaluate (true,false,false);

    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      SymmetricTensor<2,dim,VectorizedArray<number>> sym_grad_u =
          velocity.get_symmetric_gradient (q);
      VectorizedArray<number> pres = pressure.get_value(q);
      VectorizedArray<number> div = trace(sym_grad_u);
      pressure.submit_value(-1.0*pressure_scaling*div, q);

      sym_grad_u *= cell_viscosity_x_2;

      for (unsigned int d=0; d<dim; ++d)
        sym_grad_u[d][d] -= pressure_scaling*pres;

      if (is_compressible)
        for (unsigned int d=0; d<dim; ++d)
          sym_grad_u[d][d] -= cell_viscosity_x_2/3.0*div;

      velocity.submit_symmetric_gradient(sym_grad_u, q);
    }

    velocity.integrate (false,true);
    velocity.distribute_local_to_global (dst.block(0));
    pressure.integrate (true,false);
    pressure.distribute_local_to_global (dst.block(1));
  }
}

template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>
::apply_add (dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
             const dealii::LinearAlgebra::distributed::BlockVector<number> &src) const
{
  MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number> >::
      data->cell_loop(&StokesOperator::local_apply, this, dst, src);
}

/**
 * Mass matrix operator
 */
template <int dim, int degree_p, typename number>
MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>::MassMatrixOperator ()
  :
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number> >()
{}

template <int dim, int degree_p, typename number>
void
MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>::clear ()
{
  one_over_viscosity.reinit(TableIndices<1>(0));
  MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::clear();
}

template <int dim, int degree_p, typename number>
void
MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>::
fill_cell_data (const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                const double pressure_scaling,
                const Triangulation<dim> &tria,
                const DoFHandler<dim> &dof_handler_for_projection,
                const bool for_mg)
{
  const unsigned int n_cells = this->data->n_macro_cells();
  one_over_viscosity.reinit(TableIndices<1>(n_cells));

  std::vector<types::global_dof_index> local_dof_indices(dof_handler_for_projection.get_fe().dofs_per_cell);
  for (unsigned int cell=0; cell<n_cells; ++cell)
    for (unsigned int i=0; i<this->get_matrix_free()->n_components_filled(cell); ++i)
    {
      if (for_mg)
      {
        typename DoFHandler<dim>::level_cell_iterator FEQ_cell = this->get_matrix_free()->get_cell_iterator(cell,i);
        typename DoFHandler<dim>::level_cell_iterator DG_cell(&tria,
                                                              FEQ_cell->level(),
                                                              FEQ_cell->index(),
                                                              &dof_handler_for_projection);
        DG_cell->get_active_or_mg_dof_indices(local_dof_indices);
      }
      else
      {
        typename DoFHandler<dim>::active_cell_iterator FEQ_cell = this->get_matrix_free()->get_cell_iterator(cell,i);
        typename DoFHandler<dim>::active_cell_iterator DG_cell(&tria,
                                                               FEQ_cell->level(),
                                                               FEQ_cell->index(),
                                                               &dof_handler_for_projection);
        DG_cell->get_active_or_mg_dof_indices(local_dof_indices);
      }

      //TODO: projection with higher degree
      Assert(local_dof_indices.size() == 1, ExcNotImplemented());
      one_over_viscosity(cell)[i] = 1.0/viscosity_values(local_dof_indices[0]);
    }

  this->pressure_scaling = pressure_scaling;
}

template <int dim, int degree_p, typename number>
void
MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>
::local_apply (const dealii::MatrixFree<dim, number>                 &data,
               dealii::LinearAlgebra::distributed::Vector<number>       &dst,
               const dealii::LinearAlgebra::distributed::Vector<number> &src,
               const std::pair<unsigned int, unsigned int>           &cell_range) const
{
  FEEvaluation<dim,degree_p,degree_p+2,1,number> pressure (data);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    const VectorizedArray<number> &cell_one_over_viscosity = one_over_viscosity(cell);

    pressure.reinit (cell);
    pressure.read_dof_values(src);
    pressure.evaluate (true, false);
    for (unsigned int q=0; q<pressure.n_q_points; ++q)
      pressure.submit_value(cell_one_over_viscosity*pressure_scaling*pressure_scaling*
                            pressure.get_value(q),q);
    pressure.integrate (true, false);
    pressure.distribute_local_to_global (dst);
  }
}

template <int dim, int degree_p, typename number>
void
MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>
::apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
             const dealii::LinearAlgebra::distributed::Vector<number> &src) const
{
  MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::
      data->cell_loop(&MassMatrixOperator::local_apply, this, dst, src);
}

template <int dim, int degree_p, typename number>
void
MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>
::compute_diagonal ()
{
  this->inverse_diagonal_entries.
      reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number> >());
  this->diagonal_entries.
      reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number> >());

  dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
  dealii::LinearAlgebra::distributed::Vector<number> &diagonal =
      this->diagonal_entries->get_vector();

  unsigned int dummy = 0;
  this->data->initialize_dof_vector(inverse_diagonal);
  this->data->initialize_dof_vector(diagonal);

  this->data->cell_loop (&MassMatrixOperator::local_compute_diagonal, this,
                         diagonal, dummy);

  this->set_constrained_entries_to_one(diagonal);
  inverse_diagonal = diagonal;
  const unsigned int local_size = inverse_diagonal.local_size();
  for (unsigned int i=0; i<local_size; ++i)
  {
    Assert(inverse_diagonal.local_element(i) > 0.,
           ExcMessage("No diagonal entry in a positive definite operator "
                      "should be zero"));
    inverse_diagonal.local_element(i)
        =1./inverse_diagonal.local_element(i);
  }
}

template <int dim, int degree_p, typename number>
void
MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>
::local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                          dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                          const unsigned int &,
                          const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  FEEvaluation<dim,degree_p,degree_p+2,1,number> pressure (data, 0);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    const VectorizedArray<number> &cell_one_over_viscosity = one_over_viscosity(cell);

    pressure.reinit (cell);
    AlignedVector<VectorizedArray<number> > diagonal(pressure.dofs_per_cell);
    for (unsigned int i=0; i<pressure.dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<pressure.dofs_per_cell; ++j)
        pressure.begin_dof_values()[j] = VectorizedArray<number>();
      pressure.begin_dof_values()[i] = make_vectorized_array<number> (1.);

      pressure.evaluate (true,false,false);
      for (unsigned int q=0; q<pressure.n_q_points; ++q)
        pressure.submit_value(cell_one_over_viscosity*pressure_scaling*pressure_scaling*
                              pressure.get_value(q),q);
      pressure.integrate (true,false);

      diagonal[i] = pressure.begin_dof_values()[i];
    }

    for (unsigned int i=0; i<pressure.dofs_per_cell; ++i)
      pressure.begin_dof_values()[i] = diagonal[i];
    pressure.distribute_local_to_global (dst);
  }
}

/**
 * Velocity block operator
 */
template <int dim, int degree_v, typename number>
MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>::ABlockOperator ()
  :
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number> >()
{}

template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>::clear ()
{
  viscosity_x_2.reinit(TableIndices<1>(0));
  MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::clear();
}

template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>::
fill_cell_data (const dealii::LinearAlgebra::distributed::Vector<number> &viscosity_values,
                const Triangulation<dim> &tria,
                const DoFHandler<dim> &dof_handler_for_projection,
                const bool for_mg,
                const bool is_compressible)
{
  const unsigned int n_cells = this->data->n_macro_cells();
  viscosity_x_2.reinit(TableIndices<1>(n_cells));

  std::vector<types::global_dof_index> local_dof_indices(dof_handler_for_projection.get_fe().dofs_per_cell);
  for (unsigned int cell=0; cell<n_cells; ++cell)
    for (unsigned int i=0; i<this->get_matrix_free()->n_components_filled(cell); ++i)
    {

      if (for_mg)
      {
        typename DoFHandler<dim>::level_cell_iterator FEQ_cell = this->get_matrix_free()->get_cell_iterator(cell,i);
        typename DoFHandler<dim>::level_cell_iterator DG_cell(&tria,
                                                              FEQ_cell->level(),
                                                              FEQ_cell->index(),
                                                              &dof_handler_for_projection);
        DG_cell->get_active_or_mg_dof_indices(local_dof_indices);
      }
      else
      {
        typename DoFHandler<dim>::active_cell_iterator FEQ_cell = this->get_matrix_free()->get_cell_iterator(cell,i);
        typename DoFHandler<dim>::active_cell_iterator DG_cell(&tria,
                                                               FEQ_cell->level(),
                                                               FEQ_cell->index(),
                                                               &dof_handler_for_projection);
        DG_cell->get_active_or_mg_dof_indices(local_dof_indices);
      }

      //TODO: projection with higher degree
      Assert(local_dof_indices.size() == 1, ExcNotImplemented());
      viscosity_x_2(cell)[i] = 2.0*viscosity_values(local_dof_indices[0]);
    }

  this->is_compressible = is_compressible;
}

template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>
::local_apply (const dealii::MatrixFree<dim, number>                 &data,
               dealii::LinearAlgebra::distributed::Vector<number>       &dst,
               const dealii::LinearAlgebra::distributed::Vector<number> &src,
               const std::pair<unsigned int, unsigned int>           &cell_range) const
{
  FEEvaluation<dim,degree_v,degree_v+1,dim,number> velocity (data,0);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    const VectorizedArray<number> &cell_viscosity_x_2 = viscosity_x_2(cell);

    velocity.reinit (cell);
    velocity.read_dof_values(src);
    velocity.evaluate (false, true, false);
    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      SymmetricTensor<2,dim,VectorizedArray<number>> sym_grad_u =
          velocity.get_symmetric_gradient (q);
      sym_grad_u *= cell_viscosity_x_2;

      if (is_compressible)
      {
        VectorizedArray<number> div = trace(sym_grad_u);
        for (unsigned int d=0; d<dim; ++d)
          sym_grad_u[d][d] -= 1.0/3.0*div;
      }
      velocity.submit_symmetric_gradient(sym_grad_u, q);
    }
    velocity.integrate (false, true);
    velocity.distribute_local_to_global (dst);
  }
}

template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>
::apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
             const dealii::LinearAlgebra::distributed::Vector<number> &src) const
{
  MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::
      data->cell_loop(&ABlockOperator::local_apply, this, dst, src);
}

template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>
::compute_diagonal ()
{
  this->inverse_diagonal_entries.
      reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number> >());
  dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(inverse_diagonal);
  unsigned int dummy = 0;
  this->data->cell_loop (&ABlockOperator::local_compute_diagonal, this,
                         inverse_diagonal, dummy);

  this->set_constrained_entries_to_one(inverse_diagonal);

  for (unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
  {
    Assert(inverse_diagonal.local_element(i) > 0.,
           ExcMessage("No diagonal entry in a positive definite operator "
                      "should be zero"));
    inverse_diagonal.local_element(i) =
        1./inverse_diagonal.local_element(i);
  }
}

template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>
::local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                          dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                          const unsigned int &,
                          const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  FEEvaluation<dim,degree_v,degree_v+1,dim,number> velocity (data, 0);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    const VectorizedArray<number> &cell_viscosity_x_2 = viscosity_x_2(cell);

    velocity.reinit (cell);
    AlignedVector<VectorizedArray<number> > diagonal(velocity.dofs_per_cell);
    for (unsigned int i=0; i<velocity.dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<velocity.dofs_per_cell; ++j)
        velocity.begin_dof_values()[j] = VectorizedArray<number>();
      velocity.begin_dof_values()[i] = make_vectorized_array<number> (1.);

      velocity.evaluate (false,true,false);
      for (unsigned int q=0; q<velocity.n_q_points; ++q)
      {
        SymmetricTensor<2,dim,VectorizedArray<number>> sym_grad_u =
            velocity.get_symmetric_gradient (q);

        sym_grad_u *= cell_viscosity_x_2;

        if (is_compressible)
        {
          VectorizedArray<number> div = trace(sym_grad_u);
          for (unsigned int d=0; d<dim; ++d)
            sym_grad_u[d][d] -= 1.0/3.0*div;
        }

        velocity.submit_symmetric_gradient(sym_grad_u, q);
      }
      velocity.integrate (false,true);

      diagonal[i] = velocity.begin_dof_values()[i];
    }

    for (unsigned int i=0; i<velocity.dofs_per_cell; ++i)
      velocity.begin_dof_values()[i] = diagonal[i];
    velocity.distribute_local_to_global (dst);
  }
}



template <int dim, int degree_v, typename number>
void
MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>
::set_diagonal (const dealii::LinearAlgebra::distributed::Vector<number> &diag)
{
  this->inverse_diagonal_entries.
      reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number> >());
  dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(inverse_diagonal);

  inverse_diagonal = diag;

  this->set_constrained_entries_to_one(inverse_diagonal);

  for (unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
  {
    Assert(inverse_diagonal.local_element(i) > 0.,
           ExcMessage("No diagonal entry in a positive definite operator "
                      "should be zero"));
    inverse_diagonal.local_element(i) =
        1./inverse_diagonal.local_element(i);
  }
}




template <int dim>
class Viscosity : public Function<dim>
{
public:
  Viscosity ();
  virtual double value (const Point<dim> &p,
                        const unsigned int component = 0) const;
  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0) const;

private:
  double dynamic_viscosity_ratio;
  unsigned int n_sinkers;
  std::vector<Point<3> > centers;
  double delta;
  double omega;
};
template <int dim>
Viscosity<dim>::Viscosity()
  :
    dynamic_viscosity_ratio(1e4)
  , n_sinkers(4)
  , delta(200.0)
  , omega(0.1)
{
  centers.resize(8);
  centers[0] = Point<3>(2.4257829890e-01, 1.3469574514e-02, 3.8313885004e-01);
  centers[1] = Point<3>(4.1465269048e-01, 6.7768972864e-02, 9.9312692973e-01);
  centers[2] = Point<3>(4.8430804651e-01, 7.6533776604e-01, 3.1833815403e-02);
  centers[3] = Point<3>(3.0935481671e-02, 9.3264044027e-01, 8.8787953411e-01);
  centers[4] = Point<3>(5.9132973039e-01, 4.7877868473e-01, 8.3335433660e-01);
  centers[5] = Point<3>(1.8633519681e-01, 7.3565270739e-01, 1.1505317181e-01);
  centers[6] = Point<3>(6.9865863058e-01, 3.5560411138e-01, 6.3830000658e-01);
  centers[7] = Point<3>(9.0821050755e-01, 2.9400041480e-01, 2.6497158886e-01);
}

template <int dim>
double Viscosity<dim>::value (const Point<dim> &p,
                              const unsigned int /*component*/) const
{
  double Chi = 1.0;
  for (unsigned int s=0; s<n_sinkers; ++s)
  {
    double dist = p.distance(centers[s]);
    double temp = 1-std::exp(-delta*
                             std::pow(std::max(0.0,dist-omega/2.0),2));
    Chi *= temp;
  }
  const double sqrt_dynamic_viscosity_ratio = std::sqrt(dynamic_viscosity_ratio);
  return (sqrt_dynamic_viscosity_ratio - 1/sqrt_dynamic_viscosity_ratio)*(1-Chi) + 1/sqrt_dynamic_viscosity_ratio;
}
template <int dim>
void Viscosity<dim>::value_list (const std::vector<Point<dim> > &points,
                                 std::vector<double>            &values,
                                 const unsigned int              component) const
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));
  Assert (component == 0,
          ExcIndexRange (component, 0, 1));
  const unsigned int n_points = points.size();
  for (unsigned int i=0; i<n_points; ++i)
    values[i] = value(points[i],component);
}



template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide();

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &  value) const override;

private:
  double dynamic_viscosity_ratio;
  unsigned int n_sinkers;
  std::vector<Point<3> > centers;
  double delta;
  double omega;
  double beta;
};

template <int dim>
RightHandSide<dim>::RightHandSide()
  :
    n_sinkers(4)
  , delta(200.0)
  , omega(0.1)
  , beta(10)
{
  centers.resize(8);
  centers[0] = Point<3>(2.4257829890e-01, 1.3469574514e-02, 3.8313885004e-01);
  centers[1] = Point<3>(4.1465269048e-01, 6.7768972864e-02, 9.9312692973e-01);
  centers[2] = Point<3>(4.8430804651e-01, 7.6533776604e-01, 3.1833815403e-02);
  centers[3] = Point<3>(3.0935481671e-02, 9.3264044027e-01, 8.8787953411e-01);
  centers[4] = Point<3>(5.9132973039e-01, 4.7877868473e-01, 8.3335433660e-01);
  centers[5] = Point<3>(1.8633519681e-01, 7.3565270739e-01, 1.1505317181e-01);
  centers[6] = Point<3>(6.9865863058e-01, 3.5560411138e-01, 6.3830000658e-01);
  centers[7] = Point<3>(9.0821050755e-01, 2.9400041480e-01, 2.6497158886e-01);
}

template <int dim>
void RightHandSide<dim>::vector_value(const Point<dim> &p,
                                      Vector<double> &  values) const
{
  double Chi = 1.0;
  for (unsigned int s=0; s<n_sinkers; ++s)
  {
    double dist = p.distance(centers[s]);
    double temp = 1-std::exp(-delta*
                             std::pow(std::max(0.0,dist-omega/2.0),2));
    Chi *= temp;
  }

  values[0] = 0.0;
  values[1] = 0.0;
  values[2] = beta*(1.0-Chi);
  values[3] = 0.0;

  return;
}



template <int dim>
class StokesProblem
{
public:
  StokesProblem(unsigned int velocity_degree);

  void run(unsigned int refine_start, unsigned int n_cycles_global,
           unsigned int n_cycles_adaptive);

private:
  double get_workload_imbalance();

  void make_grid(unsigned int ref);
  void setup_system();
  void assemble_system();

  void evaluate_viscosity();
  void correct_stokes_rhs();
  void compute_A_block_diagonals();


  void solve();
  void refine_grid(bool global);
  void output_results(const unsigned int cycle) const;

  unsigned int velocity_degree;
  MPI_Comm     mpi_communicator;

  FESystem<dim>                             fe;
  parallel::distributed::Triangulation<dim> triangulation;
  DoFHandler<dim>                           dof_handler;

  std::vector<IndexSet> owned_partitioning;
  std::vector<IndexSet> relevant_partitioning;

  AffineConstraints<double> constraints;

  LA::MPI::BlockVector       locally_relevant_solution;
  LA::MPI::BlockVector       system_rhs;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;




  DoFHandler<dim> dof_handler_v;
  DoFHandler<dim> dof_handler_p;
  DoFHandler<dim> dof_handler_projection;

  FESystem<dim> stokes_fe;
  FESystem<dim> fe_v;
  FESystem<dim> fe_p;
  FESystem<dim> fe_projection;

  MappingQ1<dim> mapping;

  typedef MatrixFreeStokesOperators::StokesOperator<dim,2,double> StokesMatrixType;
  typedef MatrixFreeStokesOperators::MassMatrixOperator<dim,1,double> MassMatrixType;
  typedef MatrixFreeStokesOperators::ABlockOperator<dim,2,double> ABlockMatrixType;

  StokesMatrixType stokes_matrix;
  ABlockMatrixType velocity_matrix;
  MassMatrixType mass_matrix;

  AffineConstraints<double> constraints_v;
  AffineConstraints<double> constraints_p;
  AffineConstraints<double> constraints_projection;

  MGLevelObject<ABlockMatrixType> mg_matrices_A;
  MGLevelObject<MassMatrixType> mg_matrices_mass;
  MGConstrainedDoFs mg_constrained_dofs_A;
  MGConstrainedDoFs mg_constrained_dofs_mass;
  MGConstrainedDoFs mg_constrained_dofs_projection;

  dealii::LinearAlgebra::distributed::Vector<double> active_coef_dof_vec;
  MGLevelObject<dealii::LinearAlgebra::distributed::Vector<double> > level_coef_dof_vec;

  MGTransferMatrixFree<dim,double> mg_transfer_A;
  MGTransferMatrixFree<dim,double> mg_transfer_mass;


  const double stokes_tol;
  const double A_tol;
  const double mass_tol;
};



template <int dim>
StokesProblem<dim>::StokesProblem(unsigned int velocity_degree)
  : velocity_degree(velocity_degree)
  , mpi_communicator(MPI_COMM_WORLD)
  , fe(FE_Q<dim>(velocity_degree), dim, FE_Q<dim>(velocity_degree - 1), 1)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening |
                    Triangulation<dim>::limit_level_difference_at_vertices),
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
  , dof_handler(triangulation)
  , pcout(std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times),




    dof_handler_v(triangulation),
    dof_handler_p(triangulation),
    dof_handler_projection(triangulation),

    stokes_fe (FE_Q<dim>(velocity_degree),dim,
               FE_Q<dim>(velocity_degree-1),1),
    fe_v (FE_Q<dim>(velocity_degree), dim),
    fe_p (FE_Q<dim>(velocity_degree-1),1),
    fe_projection(FE_DGQ<dim>(0),1)

  , stokes_tol(1e-6)
  , A_tol (1e-2)
  , mass_tol (1e-6)
{}


template <int dim>
double StokesProblem<dim>::get_workload_imbalance ()
{
  unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);
  unsigned int n_global_levels = triangulation.n_global_levels();

  unsigned long long int work_estimate = 0;
  unsigned long long int total_cells_in_hierarchy = 0;

  for (int lvl=n_global_levels-1; lvl>=0; --lvl)
  {
    unsigned long long int work_estimate_this_level;
    unsigned long long int total_cells_on_lvl;
    unsigned long long int n_owned_cells_on_lvl = 0;

    for (const auto &cell: triangulation.cell_iterators_on_level(lvl))
      if (cell->is_locally_owned_on_level())
        n_owned_cells_on_lvl += 1;

    work_estimate_this_level = dealii::Utilities::MPI::max(n_owned_cells_on_lvl,triangulation.get_communicator());

    work_estimate += work_estimate_this_level;

    total_cells_on_lvl = dealii::Utilities::MPI::sum(n_owned_cells_on_lvl,triangulation.get_communicator());

    total_cells_in_hierarchy += total_cells_on_lvl;
  }
  double ideal_work = static_cast<double>(total_cells_in_hierarchy) / static_cast<double>(n_proc);
  double workload_imbalance_ratio = work_estimate / ideal_work;

  return workload_imbalance_ratio;
}


template <int dim>
void StokesProblem<dim>::make_grid(unsigned int ref)
{
  GridGenerator::hyper_cube (triangulation, 0, 1);
  triangulation.refine_global(ref);
}

template <int dim>
void StokesProblem<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "setup");

  dof_handler.distribute_dofs(fe);

  std::vector<unsigned int> stokes_sub_blocks(dim + 1, 0);
  stokes_sub_blocks[dim] = 1;

  DoFRenumbering::hierarchical(dof_handler);
  DoFRenumbering::component_wise(dof_handler, stokes_sub_blocks);

  std::vector<types::global_dof_index> dofs_per_block(2);
  dofs_per_block = DoFTools::count_dofs_per_block(dof_handler,
                                                  stokes_sub_blocks);

  const types::global_dof_index n_u = dofs_per_block[0], n_p = dofs_per_block[1];
  pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
        << n_u << '+' << n_p << ')' << std::endl;

  owned_partitioning.resize(2);
  owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, n_u);
  owned_partitioning[1] =
      dof_handler.locally_owned_dofs().get_view(n_u, n_u + n_p);

  {
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    relevant_partitioning.resize(2);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
    relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    constraints.reinit(locally_relevant_dofs);

    FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(dim+1),
                                             constraints,
                                             fe.component_mask(velocities));
    constraints.close();
  }

  locally_relevant_solution.reinit(owned_partitioning,
                                   relevant_partitioning,
                                   mpi_communicator);
  system_rhs.reinit(owned_partitioning, mpi_communicator);


  // Setup active DoFs
  {
    // Velocity DoFHandler
    {
      dof_handler_v.clear();
      dof_handler_v.distribute_dofs(fe_v);

      DoFRenumbering::hierarchical(dof_handler_v);

      constraints_v.clear();
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs (dof_handler_v,
                                               locally_relevant_dofs);
      constraints_v.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints (dof_handler_v, constraints_v);

      FEValuesExtractors::Vector velocities(0);
      VectorTools::interpolate_boundary_values (mapping,
                                                dof_handler,
                                                0,
                                                Functions::ZeroFunction<dim>(dim+1),
                                                constraints_v,
                                                fe.component_mask(velocities));
      constraints_v.close ();
    }

    // Pressure DoFHandler
    {
      dof_handler_p.clear();
      dof_handler_p.distribute_dofs(fe_p);

      DoFRenumbering::hierarchical(dof_handler_p);

      constraints_p.clear();
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs (dof_handler_p,
                                               locally_relevant_dofs);
      constraints_p.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints (dof_handler_p, constraints_p);
      constraints_p.close();
    }

    // Coefficient transfer objects
    {
      dof_handler_projection.clear();
      dof_handler_projection.distribute_dofs(fe_projection);

      DoFRenumbering::hierarchical(dof_handler_projection);

      active_coef_dof_vec.reinit(dof_handler_projection.locally_owned_dofs(), triangulation.get_communicator());
    }
  }


  // Multigrid DoF setup
  {
    // ABlock GMG
    dof_handler_v.distribute_mg_dofs();

    mg_constrained_dofs_A.clear();
    mg_constrained_dofs_A.initialize(dof_handler_v);

    std::set<types::boundary_id> dirichlet_boundary;
    dirichlet_boundary.insert(0);
    mg_constrained_dofs_A.make_zero_boundary_constraints(dof_handler_v, dirichlet_boundary);

    //Mass matrix GMG
    dof_handler_p.distribute_mg_dofs();

    mg_constrained_dofs_mass.clear();
    mg_constrained_dofs_mass.initialize(dof_handler_p);

    //Coefficient transfer
    dof_handler_projection.distribute_mg_dofs();
  }

  // Setup the matrix-free operators
  {
    // Stokes matrix
    {
      typename MatrixFree<dim,double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
          MatrixFree<dim,double>::AdditionalData::none;
      additional_data.mapping_update_flags = (update_values | update_gradients |
                                              update_JxW_values | update_quadrature_points);

      std::vector<const DoFHandler<dim>*> stokes_dofs;
      stokes_dofs.push_back(&dof_handler_v);
      stokes_dofs.push_back(&dof_handler_p);
      std::vector<const AffineConstraints<double> *> stokes_constraints;
      stokes_constraints.push_back(&constraints_v);
      stokes_constraints.push_back(&constraints_p);

      std::shared_ptr<MatrixFree<dim,double> >
          stokes_mf_storage(new MatrixFree<dim,double>());
      stokes_mf_storage->reinit(mapping,stokes_dofs, stokes_constraints,
                                QGauss<1>(velocity_degree+1), additional_data);
      stokes_matrix.clear();
      stokes_matrix.initialize(stokes_mf_storage);

    }

    // ABlock matrix
    {
      typename MatrixFree<dim,double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
          MatrixFree<dim,double>::AdditionalData::none;
      additional_data.mapping_update_flags = (update_values | update_gradients |
                                              update_JxW_values | update_quadrature_points);
      std::shared_ptr<MatrixFree<dim,double> >
          ablock_mf_storage(new MatrixFree<dim,double>());
      ablock_mf_storage->reinit(mapping,dof_handler_v, constraints_v,
                                QGauss<1>(velocity_degree+1), additional_data);

      velocity_matrix.clear();
      velocity_matrix.initialize(ablock_mf_storage);
    }

    // Mass matrix
    {
      typename MatrixFree<dim,double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
          MatrixFree<dim,double>::AdditionalData::none;
      additional_data.mapping_update_flags = (update_values | update_JxW_values |
                                              update_quadrature_points);
      std::shared_ptr<MatrixFree<dim,double> >
          mass_mf_storage(new MatrixFree<dim,double>());
      mass_mf_storage->reinit(mapping,dof_handler_p, constraints_p,
                              QGauss<1>(velocity_degree+1), additional_data);

      mass_matrix.clear();
      mass_matrix.initialize(mass_mf_storage);
    }

    // GMG matrices
    {
      const unsigned int n_levels = triangulation.n_global_levels();

      //ABlock GMG
      mg_matrices_A.clear_elements();
      mg_matrices_A.resize(0, n_levels-1);

      for (unsigned int level=0; level<n_levels; ++level)
      {
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler_v, level, relevant_dofs);
        AffineConstraints<double> level_constraints;
        level_constraints.reinit(relevant_dofs);
        level_constraints.add_lines(mg_constrained_dofs_A.get_boundary_indices(level));
        level_constraints.close();

        {
          typename MatrixFree<dim,double>::AdditionalData additional_data;
          additional_data.tasks_parallel_scheme =
              MatrixFree<dim,double>::AdditionalData::none;
          additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                                  update_quadrature_points);
          additional_data.mg_level = level;

          std::shared_ptr<MatrixFree<dim,double> >
              mg_mf_storage_level(new MatrixFree<dim,double>());
          mg_mf_storage_level->reinit(mapping, dof_handler_v, level_constraints,
                                      QGauss<1>(velocity_degree+1),
                                      additional_data);

          mg_matrices_A[level].clear();
          mg_matrices_A[level].initialize(mg_mf_storage_level, mg_constrained_dofs_A, level);

        }
      }

      //Mass matrix GMG
      mg_matrices_mass.clear_elements();
      mg_matrices_mass.resize(0, n_levels-1);

      for (unsigned int level=0; level<n_levels; ++level)
      {
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler_p, level, relevant_dofs);
        AffineConstraints<double> level_constraints;
        level_constraints.reinit(relevant_dofs);
        level_constraints.close();

        {
          typename MatrixFree<dim,double>::AdditionalData additional_data;
          additional_data.tasks_parallel_scheme =
              MatrixFree<dim,double>::AdditionalData::none;
          additional_data.mapping_update_flags = (update_values | update_JxW_values |
                                                  update_quadrature_points);
          additional_data.mg_level = level;

          std::shared_ptr<MatrixFree<dim,double> >
              mg_mf_storage_level(new MatrixFree<dim,double>());
          mg_mf_storage_level->reinit(mapping, dof_handler_p, level_constraints,
                                      QGauss<1>(velocity_degree+1),
                                      additional_data);

          mg_matrices_mass[level].clear();
          mg_matrices_mass[level].initialize(mg_mf_storage_level, mg_constrained_dofs_mass, level);

        }
      }
    }
  }

  // Build MG transfer
  {
    mg_transfer_A.clear();
    mg_transfer_A.initialize_constraints(mg_constrained_dofs_A);
    mg_transfer_A.build(dof_handler_v);

    mg_transfer_mass.clear();
    mg_transfer_mass.initialize_constraints(mg_constrained_dofs_mass);
    mg_transfer_mass.build(dof_handler_p);
  }
}



template <int dim>
void StokesProblem<dim>::assemble_system()
{
  TimerOutput::Scope t(computing_timer, "assembly");

  system_rhs            = 0;

  const QGauss<dim> quadrature_formula(velocity_degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double>     cell_rhs(dofs_per_cell);

  const RightHandSide<dim>    right_hand_side;
  std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      cell_rhs     = 0;

      fe_values.reinit(cell);
      right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                        rhs_values);
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const unsigned int component_i =
              fe.system_to_component_index(i).first;
          cell_rhs(i) += fe_values.shape_value(i, q) *
              rhs_values[q](component_i) * fe_values.JxW(q);
        }
      }


      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_rhs,
                                             local_dof_indices,
                                             system_rhs);
    }

  system_rhs.compress(VectorOperation::add);
}



template <int dim>
void StokesProblem<dim>::evaluate_viscosity ()
{
  TimerOutput::Scope t(computing_timer, "evaluate_viscosity");

  {
    const QGauss<dim> quadrature_formula (velocity_degree+1);
    FEValues<dim> fe_values (mapping,
                             fe,
                             quadrature_formula,
                             update_values   |
                             update_gradients |
                             update_quadrature_points |
                             update_JxW_values);

    std::vector<types::global_dof_index> local_dof_indices(fe_projection.dofs_per_cell);
    active_coef_dof_vec = 0.;

    Viscosity<dim> viscosity_function;

    // compute the integral quantities by quadrature
    for (const auto &cell: dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fe_values.reinit (cell);

        std::vector<double> visc_vals(fe_values.get_quadrature_points().size());
        viscosity_function.value_list(fe_values.get_quadrature_points(),
                                      visc_vals);

        average(visc_vals,"harmonic");

        typename DoFHandler<dim>::active_cell_iterator dg_cell(&triangulation,
                                                               cell->level(),
                                                               cell->index(),
                                                               &dof_handler_projection);

        dg_cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < fe_projection.dofs_per_cell; ++i)
          active_coef_dof_vec[local_dof_indices[i]] = visc_vals[0];
      }
    active_coef_dof_vec.compress(VectorOperation::insert);
  }

  stokes_matrix.fill_cell_data(active_coef_dof_vec,
                               1.0,
                               triangulation,
                               dof_handler_projection,
                               /*is_compressible*/false);

  velocity_matrix.fill_cell_data(active_coef_dof_vec,
                                 triangulation,
                                 dof_handler_projection,
                                 /*for_mg*/ false,
                                 /*is_compressible*/false);

  mass_matrix.fill_cell_data(active_coef_dof_vec,
                             1.0,
                             triangulation,
                             dof_handler_projection,
                             /*for_mg*/ false);

  // Project to MG
  const unsigned int n_levels = triangulation.n_global_levels();
  level_coef_dof_vec = 0.;
  level_coef_dof_vec.resize(0,n_levels-1);

  MGTransferMatrixFree<dim,double> transfer(mg_constrained_dofs_A);
  transfer.build(dof_handler_projection);
  transfer.interpolate_to_mg(dof_handler_projection,
                             level_coef_dof_vec,
                             active_coef_dof_vec);

  for (unsigned int level=0; level<n_levels; ++level)
  {
    mg_matrices_A[level].fill_cell_data(level_coef_dof_vec[level],
                                        triangulation,
                                        dof_handler_projection,
                                        /*for_mg*/ true,
                                        /*is_compressible*/false);

    mg_matrices_mass[level].fill_cell_data(level_coef_dof_vec[level],
                                           1.0,
                                           triangulation,
                                           dof_handler_projection,
                                           /*for_mg*/ true);
  }
}


template <int dim>
void StokesProblem<dim>::correct_stokes_rhs()
{
  TimerOutput::Scope t(computing_timer, "correct_stokes_rhs");

  dealii::LinearAlgebra::distributed::BlockVector<double> rhs_correction(2);
  dealii::LinearAlgebra::distributed::BlockVector<double> u0(2);

  stokes_matrix.initialize_dof_vector(rhs_correction);
  stokes_matrix.initialize_dof_vector(u0);

  rhs_correction.collect_sizes();
  u0.collect_sizes();

  u0 = 0;
  rhs_correction = 0;
  constraints.distribute(u0);
  u0.update_ghost_values();

  FEEvaluation<dim,2,3,dim,double>
      velocity (*stokes_matrix.get_matrix_free(), 0);
  FEEvaluation<dim,1,3,1,double>
      pressure (*stokes_matrix.get_matrix_free(), 1);

  for (unsigned int cell=0; cell<stokes_matrix.get_matrix_free()->n_macro_cells(); ++cell)
  {
    const VectorizedArray<double> &cell_viscosity_x_2 = stokes_matrix.get_viscosity_x_2_table()(cell);

    velocity.reinit (cell);
    velocity.read_dof_values_plain (u0.block(0));
    velocity.evaluate (false,true,false);
    pressure.reinit (cell);
    pressure.read_dof_values_plain (u0.block(1));
    pressure.evaluate (true,false,false);

    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      SymmetricTensor<2,dim,VectorizedArray<double>> sym_grad_u =
          velocity.get_symmetric_gradient (q);
      VectorizedArray<double> pres = pressure.get_value(q);
      VectorizedArray<double> div = -trace(sym_grad_u);
      pressure.submit_value   (-1.0*1.0*div, q);

      sym_grad_u *= cell_viscosity_x_2;

      for (unsigned int d=0; d<dim; ++d)
        sym_grad_u[d][d] -= 1.0*pres;

      velocity.submit_symmetric_gradient(-1.0*sym_grad_u, q);
    }

    velocity.integrate (false,true);
    velocity.distribute_local_to_global (rhs_correction.block(0));
    pressure.integrate (true,false);
    pressure.distribute_local_to_global (rhs_correction.block(1));
  }
  rhs_correction.compress(VectorOperation::add);

  LA::MPI::BlockVector stokes_rhs_correction (owned_partitioning, mpi_communicator);
  ChangeVectorTypes::copy(stokes_rhs_correction,rhs_correction);
  system_rhs.block(0) += stokes_rhs_correction.block(0);
  system_rhs.block(1) += stokes_rhs_correction.block(1);
}

template <int dim>
void StokesProblem<dim>::compute_A_block_diagonals()
{
  TimerOutput::Scope t(computing_timer, "compute_A_block_diagonals");

  for (unsigned int level=0; level < triangulation.n_global_levels(); ++level)
  {
    //Mass matrix GMG
    mg_matrices_mass[level].compute_diagonal();

    // ABlock GMG
    mg_matrices_A[level].compute_diagonal();
  }
}



template <int dim>
void StokesProblem<dim>::solve()
{
  Timer timer(mpi_communicator,true);

  TimerOutput::Scope t(computing_timer, "solve");


  // Below we define all the objects needed to build the GMG preconditioner:
  using vector_t = dealii::LinearAlgebra::distributed::Vector<double>;

  // ABlock GMG Smoother: Chebyshev, degree 4
  typedef PreconditionChebyshev<ABlockMatrixType,vector_t> ABlockSmootherType;
  mg::SmootherRelaxation<ABlockSmootherType, vector_t>
      mg_smoother_A;
  {
    MGLevelObject<typename ABlockSmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level = 0; level<triangulation.n_global_levels(); ++level)
    {
      if (level > 0)
      {
        smoother_data[level].smoothing_range = 15.;
        smoother_data[level].degree = 4;
        smoother_data[level].eig_cg_n_iterations = 10;
      }
      else
      {
        smoother_data[0].smoothing_range = 1e-3;
        smoother_data[0].degree = numbers::invalid_unsigned_int;
        smoother_data[0].eig_cg_n_iterations = 100; /*mg_matrices_A[0].m();*/
      }
      smoother_data[level].preconditioner = mg_matrices_A[level].get_matrix_diagonal_inverse();
    }
    mg_smoother_A.initialize(mg_matrices_A, smoother_data);
  }

  // Mass matrix GMG Smoother: Chebyshev, degree 4
  typedef PreconditionChebyshev<MassMatrixType,vector_t> MassSmootherType;
  mg::SmootherRelaxation<MassSmootherType, vector_t>
      mg_smoother_mass;
  {
    MGLevelObject<typename MassSmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level = 0; level<triangulation.n_global_levels(); ++level)
    {
      if (level > 0)
      {
        smoother_data[level].smoothing_range = 15.;
        smoother_data[level].degree = 4;
        smoother_data[level].eig_cg_n_iterations = 10;
      }
      else
      {
        smoother_data[0].smoothing_range = 1e-3;
        smoother_data[0].degree = numbers::invalid_unsigned_int;
        smoother_data[0].eig_cg_n_iterations = 100; /*mg_matrices_mass[0].m();*/
      }
      smoother_data[level].preconditioner = mg_matrices_mass[level].get_matrix_diagonal_inverse();
    }
    mg_smoother_mass.initialize(mg_matrices_mass, smoother_data);
  }


  // Coarse Solver is just an application of the Chebyshev smoother setup
  // in such a way to be a solver
  //ABlock GMG
  MGCoarseGridApplySmoother<vector_t> mg_coarse_A;
  mg_coarse_A.initialize(mg_smoother_A);
  //Mass matrix GMG
  MGCoarseGridApplySmoother<vector_t> mg_coarse_mass;
  mg_coarse_mass.initialize(mg_smoother_mass);

  // Interface matrices
  // Ablock GMG
  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<ABlockMatrixType> > mg_interface_matrices_A;
  mg_interface_matrices_A.resize(0, triangulation.n_global_levels()-1);
  for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
    mg_interface_matrices_A[level].initialize(mg_matrices_A[level]);
  mg::Matrix<vector_t > mg_interface_A(mg_interface_matrices_A);

  // Mass matrix GMG
  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<MassMatrixType> > mg_interface_matrices_mass;
  mg_interface_matrices_mass.resize(0, triangulation.n_global_levels()-1);
  for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
    mg_interface_matrices_mass[level].initialize(mg_matrices_mass[level]);
  mg::Matrix<vector_t > mg_interface_mass(mg_interface_matrices_mass);



  // MG Matrix
  mg::Matrix<vector_t > mg_matrix_A(mg_matrices_A);
  mg::Matrix<vector_t > mg_matrix_mass(mg_matrices_mass);

  // MG object
  // ABlock GMG
  Multigrid<vector_t > mg_A(mg_matrix_A,
                            mg_coarse_A,
                            mg_transfer_A,
                            mg_smoother_A,
                            mg_smoother_A);
  mg_A.set_edge_matrices(mg_interface_A, mg_interface_A);
  // Mass matrix GMG
  Multigrid<vector_t > mg_mass(mg_matrix_mass,
                               mg_coarse_mass,
                               mg_transfer_mass,
                               mg_smoother_mass,
                               mg_smoother_mass);
  mg_mass.set_edge_matrices(mg_interface_mass, mg_interface_mass);

  // GMG Preconditioner
  // Ablock GMG
  typedef PreconditionMG<dim, vector_t, MGTransferMatrixFree<dim,double> > APreconditioner;
  APreconditioner prec_A(dof_handler_v, mg_A, mg_transfer_A);

  // Mass matrix GMG
  typedef PreconditionMG<dim, vector_t, MGTransferMatrixFree<dim,double> > MassPreconditioner;
  APreconditioner prec_S(dof_handler_p, mg_mass, mg_transfer_mass);


  // Many parts of the solver depend on the block layout (velocity = 0,
  // pressure = 1). For example the linearized_stokes_initial_guess vector or the StokesBlock matrix
  // wrapper. Let us make sure that this holds (and shorten their names):
  const unsigned int block_vel = 0;
  const unsigned int block_p = 1;

  LA::MPI::BlockVector distributed_stokes_solution (owned_partitioning,
                                                    mpi_communicator);
  // extract Stokes parts of rhs vector
  LA::MPI::BlockVector distributed_stokes_rhs(owned_partitioning,
                                              mpi_communicator);

  distributed_stokes_rhs.block(block_vel) = system_rhs.block(block_vel);
  distributed_stokes_rhs.block(block_p) = system_rhs.block(block_p);

  // create a completely distributed vector that will be used for
  // the scaled and denormalized solution and later used as a
  // starting guess for the linear solver
  LA::MPI::BlockVector linearized_stokes_initial_guess (owned_partitioning,
                                                        mpi_communicator);

  // copy the velocity and pressure from current_linearization_point into
  // the vector linearized_stokes_initial_guess. We need to do the copy because
  // linearized_stokes_variables has a different
  // layout than current_linearization_point, which also contains all the
  // other solution variables.
  linearized_stokes_initial_guess.block (block_vel) = 0;
  linearized_stokes_initial_guess.block (block_p) = 0;
  constraints.set_zero (linearized_stokes_initial_guess);


  double solver_tolerance = 0;
  {
    // (ab)use the distributed solution vector to temporarily put a residual in
    // (we don't care about the residual vector -- all we care about is the
    // value (number) of the initial residual). The initial residual is returned
    // to the caller (for nonlinear computations). This value is computed before
    // the solve because we want to compute || A^{k+1} U^k - F^{k+1} ||, which is
    // the nonlinear residual. Because the place where the nonlinear residual is
    // checked against the nonlinear tolerance comes after the solve, the system
    // is solved one time too many in the case of a nonlinear Picard solver.

    // We must copy between Trilinos/dealii vector types
    dealii::LinearAlgebra::distributed::BlockVector<double> solution_copy(2);
    dealii::LinearAlgebra::distributed::BlockVector<double> initial_copy(2);
    dealii::LinearAlgebra::distributed::BlockVector<double> rhs_copy(2);

    stokes_matrix.initialize_dof_vector(solution_copy);
    stokes_matrix.initialize_dof_vector(initial_copy);
    stokes_matrix.initialize_dof_vector(rhs_copy);

    solution_copy.collect_sizes();
    initial_copy.collect_sizes();
    rhs_copy.collect_sizes();

    ChangeVectorTypes::copy(solution_copy,distributed_stokes_solution);
    ChangeVectorTypes::copy(initial_copy,linearized_stokes_initial_guess);
    ChangeVectorTypes::copy(rhs_copy,distributed_stokes_rhs);

    // Compute residual l2_norm
    stokes_matrix.vmult(solution_copy,initial_copy);
    solution_copy.sadd(-1,1,rhs_copy);

    // Note: the residual is computed with a zero velocity, effectively computing
    // || B^T p - g ||, which we are going to use for our solver tolerance.
    // We do not use the current velocity for the initial residual because
    // this would not decrease the number of iterations if we had a better
    // initial guess (say using a smaller timestep). But we need to use
    // the pressure instead of only using the norm of the rhs, because we
    // are only interested in the part of the rhs not balanced by the static
    // pressure (the current pressure is a good approximation for the static
    // pressure).
    initial_copy.block(0) = 0.;
    stokes_matrix.vmult(solution_copy,initial_copy);
    solution_copy.block(0).sadd(-1,1,rhs_copy.block(0));

    const double residual_u = solution_copy.block(0).l2_norm();

    const double residual_p = rhs_copy.block(1).l2_norm();

    solver_tolerance = stokes_tol *
        std::sqrt(residual_u*residual_u+residual_p*residual_p);
  }

  // Now overwrite the solution vector again with the current best guess
  // to solve the linear system
  distributed_stokes_solution = linearized_stokes_initial_guess;

  // Again, copy solution and rhs vectors to solve with matrix-free operators
  dealii::LinearAlgebra::distributed::BlockVector<double> solution_copy(2);
  dealii::LinearAlgebra::distributed::BlockVector<double> rhs_copy(2);

  stokes_matrix.initialize_dof_vector(solution_copy);
  stokes_matrix.initialize_dof_vector(rhs_copy);

  solution_copy.collect_sizes();
  rhs_copy.collect_sizes();

  ChangeVectorTypes::copy(solution_copy,distributed_stokes_solution);
  ChangeVectorTypes::copy(rhs_copy,distributed_stokes_rhs);

  SolverControl solver_control_cheap (200,
                                      solver_tolerance, true);

  solver_control_cheap.enable_history_data();


  const bool do_mass_solve = false;


  // create a cheap preconditioner that consists of only a single V-cycle
  const BlockSchurGMGPreconditioner<ABlockMatrixType, StokesMatrixType, MassMatrixType, MassPreconditioner, APreconditioner>
      preconditioner_cheap (stokes_matrix, velocity_matrix, mass_matrix,
                            prec_S, prec_A,
                            do_mass_solve,
                            false,
                            A_tol,
                            mass_tol);

  {
    dealii::LinearAlgebra::distributed::BlockVector<double> tmp_dst = solution_copy;
    dealii::LinearAlgebra::distributed::BlockVector<double> tmp_scr = rhs_copy;
    preconditioner_cheap.vmult(tmp_dst, tmp_scr);
    tmp_scr = tmp_dst;

    double vmult_time = 0.0;
    for (unsigned int j=0; j<5; ++j)
    {
      timer.restart();
      preconditioner_cheap.vmult(tmp_dst, tmp_scr);
      timer.stop();
      vmult_time += timer.last_wall_time();

      tmp_scr = tmp_dst;
    }
    pcout << "   PrecVmult timings:              " << vmult_time/5.0 << std::endl;
  }

  {
    dealii::LinearAlgebra::distributed::BlockVector<double> tmp_dst = solution_copy;
    dealii::LinearAlgebra::distributed::BlockVector<double> tmp_scr = rhs_copy;
    stokes_matrix.vmult(tmp_dst, tmp_scr);
    tmp_scr = tmp_dst;

    double vmult_time = 0.0;
    for (unsigned int j=0; j<10; ++j)
    {
      timer.restart();
      stokes_matrix.vmult(tmp_dst, tmp_scr);
      timer.stop();
      vmult_time += timer.last_wall_time();

      tmp_scr = tmp_dst;
    }
    pcout << "   MatVmult timings:               " << vmult_time/10.0 << std::endl;
  }

  PrimitiveVectorMemory<dealii::LinearAlgebra::distributed::BlockVector<double> > mem;
  SolverFGMRES<dealii::LinearAlgebra::distributed::BlockVector<double> >
      solver(solver_control_cheap, mem,
             SolverFGMRES<dealii::LinearAlgebra::distributed::BlockVector<double> >::
             AdditionalData(50));

  timer.restart();
  solver.solve (stokes_matrix,
                solution_copy,
                rhs_copy,
                preconditioner_cheap);
  timer.stop();
  const double solve_time = timer.last_wall_time();
  unsigned int gmres_m = solver_control_cheap.last_step();
  pcout << "   FGMRES Solve timings:           " << solve_time << "  (" << gmres_m << " iterations)"
        << std::endl;

  solution_copy.update_ghost_values();
  LA::MPI::BlockVector distributed_locally_relevant_solution(owned_partitioning,mpi_communicator);
  ChangeVectorTypes::copy(distributed_locally_relevant_solution,solution_copy);
  locally_relevant_solution.block(0) = distributed_locally_relevant_solution.block(0);
  locally_relevant_solution.block(1) = distributed_locally_relevant_solution.block(1);
}



template <int dim>
void StokesProblem<dim>::refine_grid(bool global)
{
  TimerOutput::Scope t(computing_timer, "refine");

  if (global)
    triangulation.refine_global();
  else
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    for (auto & cell: triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        estimated_error_per_cell(cell->active_cell_index())
            = cell->center().norm();
    }

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
          triangulation, estimated_error_per_cell, 0.1428, 0.0);

    triangulation.execute_coarsening_and_refinement();
  }
}

template <int dim>
void StokesProblem<dim>::output_results (const unsigned int cycle) const
{
  TimerOutput::Scope t(computing_timer, "output_results");

  dealii::LinearAlgebra::distributed::BlockVector<double> solution(2);
  stokes_matrix.initialize_dof_vector(solution);
  ChangeVectorTypes::copy(solution,locally_relevant_solution);


  solution.update_ghost_values();

  std::vector<std::string> solution_names (dim, "velocity");
  solution_names.push_back ("pressure");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation
      (dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation
      .push_back (DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution,
                            solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);

  Vector<double> subdomain (triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector (subdomain, "subdomain");

  Vector<double> visc_values (triangulation.n_active_cells());
  {
    Viscosity<dim> viscosity;
    typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
    for (;cell!=triangulation.end();++cell)
      if (cell->is_locally_owned())
        visc_values(cell->active_cell_index()) = viscosity.value(cell->center());
    data_out.add_data_vector (visc_values, "viscosity");
  }
  data_out.build_patches ();

  const std::string filename = ("solution-" +
                                Utilities::int_to_string (cycle, 2) +
                                "." +
                                Utilities::int_to_string
                                (triangulation.locally_owned_subdomain(), 4));
  std::ofstream output ((filename + ".vtu").c_str());
  data_out.write_vtu (output);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i=0;
         i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
         ++i)
      filenames.push_back ("solution-" +
                           Utilities::int_to_string (cycle, 2) +
                           "." +
                           Utilities::int_to_string (i, 4) +
                           ".vtu");

    std::ofstream master_output (("solution-" +
                                  Utilities::int_to_string (cycle, 2) +
                                  ".pvtu").c_str());
    data_out.write_pvtu_record (master_output, filenames);
  }
}



template <int dim>
void StokesProblem<dim>::run(unsigned int refine_start, unsigned int n_cycles_global,
                             unsigned int n_cycles_adaptive)
{
  const unsigned int n_vect_doubles =
      VectorizedArray<double>::n_array_elements;
  const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;
  pcout << "Vectorization over " << n_vect_doubles
        << " doubles = " << n_vect_bits << " bits ("
        << Utilities::System::get_current_vectorization_level()
        << "), VECTORIZATION_LEVEL=" << DEAL_II_COMPILER_VECTORIZATION_LEVEL
        << std::endl;
  pcout << "running on " << Utilities::MPI::n_mpi_processes(mpi_communicator) << " ranks." << std::endl;


  {
    // dof estimate:

    long long start_dofs = 26.2*std::pow(8.0,refine_start);
    pcout << "estimate: " << std::endl;

    for (unsigned int i=0;i<n_cycles_global;++i)
    {
      pcout << start_dofs << std::endl;
      start_dofs*= 8;
    }
    for (unsigned int i=0;i<n_cycles_adaptive;++i)
    {
      pcout << start_dofs << std::endl;
      start_dofs*= 2;
    }
  }

  for (unsigned int cycle = 0; cycle < n_cycles_global+n_cycles_adaptive; ++cycle)
  {
    pcout << "Cycle " << cycle << ':' << std::endl;

    if (cycle == 0)
      make_grid(refine_start);
    else
      refine_grid(cycle<=n_cycles_global);

    Timer timer(mpi_communicator,true);

    timer.restart();
    setup_system();
    timer.stop();
    pcout << std::endl
          << "   Setup DoFs timings:             " << timer.last_wall_time() << std::endl;

    timer.restart();
    assemble_system();
    evaluate_viscosity();
    compute_A_block_diagonals();
    correct_stokes_rhs();
    timer.stop();
    pcout << "   Assemble System (RHS) timings:  " << timer.last_wall_time() << std::endl;

    solve();

    //output_results(cycle);

    pcout << "   Workload Imbalance:             " << get_workload_imbalance() << std::endl;

    Utilities::System::MemoryStats mem;
    Utilities::System::get_memory_stats(mem);
    pcout << "   VM Peak:                        " << Utilities::MPI::max(mem.VmPeak, MPI_COMM_WORLD) << std::endl;

    pcout << std::endl;
    computing_timer.print_summary();
    computing_timer.reset();
  }
}
} // namespace Nsinker



int main(int argc, char *argv[])
{

  if (argc!=4)
  {
    std::cout << "usage: start_refinements n_cycles_global n_cycles_adaptive" << std::endl;
    return 1;
  }

  unsigned int refine_start = dealii::Utilities::string_to_int(argv[1]);
  unsigned int n_cycles_global = dealii::Utilities::string_to_int(argv[2]);
  unsigned int n_cycles_adaptive = dealii::Utilities::string_to_int(argv[3]);

  try
  {
    using namespace dealii;
    using namespace Nsinker;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    StokesProblem<3> problem(2);
    problem.run(refine_start, n_cycles_global, n_cycles_adaptive);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
