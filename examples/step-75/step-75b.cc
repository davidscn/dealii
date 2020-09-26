/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2020 by the deal.II authors
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
  * Author:
  */



#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "adapter.h"

using namespace dealii;

struct CouplingParamters
{
  std::string config_file      = "precice-config.xml";
  std::string participant_name = "dummy-participant";
  std::string mesh_name        = "dummy-mesh";
  std::string write_data_name  = "dummy";
  std::string read_data_name   = "solution";
};

template <int dim>
class Step75b
{
public:
  Step75b();
  void run();

private:
  void make_grid();
  void setup_system();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> dummy_vector;

  CouplingParamters  parameters;
  const unsigned int interface_boundary_id;
  Adapter::Adapter<dim, Vector<double>, CouplingParamters> adapter;
};



template <int dim>
Step75b<dim>::Step75b()
  : fe(1)
  , dof_handler(triangulation)
  , interface_boundary_id(0)
  , adapter(parameters, interface_boundary_id)
{}



template <int dim>
void Step75b<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1, false);
  triangulation.refine_global(4);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}



template <int dim>
void Step75b<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  dummy_vector.reinit(dof_handler.n_dofs());
}



template <int dim>
void Step75b<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches();

  std::ofstream output(dim == 2 ? "mapped-solution-2d.vtk" :
                                  "mapped-solution-3d.vtk");
  data_out.write_vtk(output);
}



template <int dim>
void Step75b<dim>::run()
{
  make_grid();
  setup_system();
  adapter.initialize(dof_handler, dummy_vector, solution);

  while (adapter.precice.isCouplingOngoing())
    {
      adapter.advance(dummy_vector, solution, 1);
      output_results();
    }
}



int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
    {
      Step75b<2> laplace_problem_3d;
      laplace_problem_3d.run();
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
