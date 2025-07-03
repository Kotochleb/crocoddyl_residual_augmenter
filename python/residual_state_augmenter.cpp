#include "crocoddyl_residual_augmenter/residual_state_augmenter.hpp"

#include "crocoddyl_residual_augmenter/python.hpp"

namespace residual_augmenter {

namespace python {

namespace bp = boost::python;

void exposeCostResidualAugmenter() {
  bp::register_ptr_to_python<std::shared_ptr<CostModelResidualAugmenter>>();

  bp::class_<CostModelResidualAugmenter, bp::bases<CostModelAbstract>>(
      "CostModelResidualAugmenter",
      "Class wrapping standard crocoddyl::CostModelResidualAugmenter class, "
      "hiding extended state vector from inner class allowing it to work with "
      "custom actuation models. Assumes first state variables are preserved as "
      "q and dq.",
      bp::init<std::shared_ptr<CostModelAbstract> &>(
          bp::args("self", "cost"),
          "Initialize the residual cost model.\n\n"
          ":param cost: Cost Model object to wrap with state reduction."))
      .def<void (CostModelResidualAugmenter::*)(
          const std::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &CostModelResidualAugmenter::calc,
          bp::args("self", "data", "x", "u"),
          "Forward state and control to compute the residual cost of the inner "
          "model.\n\n"
          ":param data: cost residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (CostModelResidualAugmenter::*)(
          const std::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"),
          "Forward state and control to compute the residual cost of "
          "the inner model with respect to the state only.\n\n"
          "It updates the total cost based on the state only.\n"
          "This function is used in the terminal nodes of an optimal control "
          "problem.\n"
          ":param data: cost data\n"
          ":param x: state point (dim. state.nx)")
      .def<void (CostModelResidualAugmenter::*)(
          const std::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &CostModelResidualAugmenter::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Forward state and control to compute the derivatives of the inner "
          "model.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: cost residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (CostModelResidualAugmenter::*)(
          const std::shared_ptr<CostDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &CostModelAbstract::calcDiff,
          bp::args("self", "data", "x"),
          "Forward state and control to compute the derivatives of the inner "
          "model "
          "with respect to the state only.\n\n"
          "It updates the Jacobian and Hessian of the cost function based on "
          "the state only.\n"
          "This function is used in the terminal nodes of an optimal control "
          "problem.\n"
          ":param data: cost residual data\n"
          ":param x: state point (dim. state.nx)")
      .def("createData", &CostModelResidualAugmenter::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the residual cost data of both this and inner model.\n\n"
           "Each cost model has its own data that needs to be allocated. This "
           "function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property(
          "wrapped_model",
          bp::make_function(&CostModelResidualAugmenter::get_wrapped_model,
                            bp::return_value_policy<bp::return_by_value>()),
          "Inner cost model");

  bp::register_ptr_to_python<std::shared_ptr<CostDataResidualAugmenter>>();

  bp::class_<CostDataResidualAugmenter, bp::bases<CostDataAbstract>>(
      "CostDataResidualAugmenter", "Data for residual cost.\n\n",
      bp::init<CostModelResidualAugmenter *, DataCollectorAbstract *>(
          bp::args("self", "model", "data"),
          "Create residual cost data.\n\n"
          ":param model: residual cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3>>()])
      .add_property(
          "wrapped_data",
          bp::make_getter(&CostDataResidualAugmenter::wrapped_data,
                          bp::return_value_policy<bp::return_by_value>()),
          "Inner cost model's data");
}

}  // namespace python
}  // namespace residual_augmenter
