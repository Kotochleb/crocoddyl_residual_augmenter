#include "crocoddyl_residual_augmenter/python.hpp"

BOOST_PYTHON_MODULE(crocoddyl_residual_augmenter) {
  namespace bp = boost::python;

  bp::import("crocoddyl");
  // Enabling eigenpy support, i.e. numpy/eigen compatibility.
  eigenpy::enableEigenPy();
  eigenpy::enableEigenPySpecific<Eigen::VectorXi>();
  residual_augmenter::python::exposeCostResidualAugmenter();
}
