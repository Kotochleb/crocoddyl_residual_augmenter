#ifndef CROCODDYL_RESIDUAL_AUGMENTER_FWD_HPP_
#define CROCODDYL_RESIDUAL_AUGMENTER_FWD_HPP_

namespace residual_augmenter {

template <typename Scalar>
class CostModelResidualAugmenterTpl;
typedef CostModelResidualAugmenterTpl<double> CostModelResidualAugmenter;
template <typename Scalar>
class CostDataResidualAugmenterTpl;
typedef CostDataResidualAugmenterTpl<double> CostDataResidualAugmenter;

}  // namespace residual_augmenter

#endif  // CROCODDYL_RESIDUAL_AUGMENTER_FWD_HPP_
