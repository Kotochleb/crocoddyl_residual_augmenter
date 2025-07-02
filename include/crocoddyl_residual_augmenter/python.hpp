#ifndef CROCODDYL_RESIDUAL_AUGMENTER_PYTHON_HPP_
#define CROCODDYL_RESIDUAL_AUGMENTER_PYTHON_HPP_

#include "crocoddyl_residual_augmenter/fwd.hpp"
// include fwd first
#include <eigenpy/eigenpy.hpp>

namespace residual_augmenter {
namespace python {

void exposeCostResidualAugmenter();

}  // namespace python
}  // namespace residual_augmenter

#endif  // CROCODDYL_RESIDUAL_AUGMENTER_PYTHON_HPP_
