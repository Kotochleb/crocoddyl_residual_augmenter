#ifndef CROCODDYL_RESIDUAL_AUGMENTER__RESIDUAL_STATE_AUGMENTER_HPP_
#define CROCODDYL_RESIDUAL_AUGMENTER__RESIDUAL_STATE_AUGMENTER_HPP_

#include "crocoddyl_residual_augmenter/fwd.hpp"
// include fwd first

#include <crocoddyl/core/cost-base.hpp>
#include <crocoddyl/core/fwd.hpp>
#include <crocoddyl/core/residual-base.hpp>

namespace residual_augmenter {

using namespace crocoddyl;

/**
 * @brief Residual-based cost wrapper reducing state dimension
 *
 * Class wrapping standard crocoddyl::CostModelResidualAugmenter class, hiding
 * extended state vector from inner class allowing it to work with custom
 * actuation models. Assumes first state variables are preserved as q and dq.
 *
 * \sa `CostModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelResidualAugmenterTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataResidualAugmenterTpl<Scalar> Data;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ResidualModelAbstractTpl<Scalar> ResidualModelAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the residual cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] residual    Residual model
   */
  CostModelResidualAugmenterTpl(const std::shared_ptr<Base> &model);

  virtual ~CostModelResidualAugmenterTpl();

  /**
   * @brief Forward state and control to compute the residual cost
   * of the inner model
   *
   * @param[in] data  Residual cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<CostDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  /**
   * @brief Forward state and control to compute the residual cost
   * of the inner model with respect to the state only
   *
   * It updates the total cost based on the state only. This function is used in
   * the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Residual cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<CostDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x);

  /**
   * @brief Forward state and control to compute the derivatives
   * of the inner model
   *
   * @param[in] data  Residual cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<CostDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);

  /**
   * @brief Forward state and control to compute the derivatives
   * of the inner model with respect to the state only
   *
   * It updates the Jacobian and Hessian of the cost function based on the state
   * only. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  Residual cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<CostDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x);

  /**
   * @brief Create the residual cost data of both this and inner model
   */
  virtual std::shared_ptr<CostDataAbstract> createData(
      DataCollectorAbstract *const data);

  /**
   * @brief Print relevant information of the cost-residual model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream &os) const;

  /**
   * @brief Return the inner cost model
   */
  const std::shared_ptr<Base> &get_wrapped_model() const;

 protected:
  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

  std::shared_ptr<Base> wrapped_model_;
};

template <typename _Scalar>
struct CostDataResidualAugmenterTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  template <template <typename Scalar> class Model>
  CostDataResidualAugmenterTpl(Model<Scalar> *const model,
                               DataCollectorAbstract *const shared_data)
      : Base(model, shared_data) {
    wrapped_data = model->get_wrapped_model()->createData(shared_data);
  }

  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::residual;
  using Base::shared;

  std::shared_ptr<Base> wrapped_data;
};

}  // namespace residual_augmenter

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl_residual_augmenter/residual_state_augmenter.hxx"

#endif  // CROCODDYL_RESIDUAL_AUGMENTER__RESIDUAL_STATE_AUGMENTER_HPP_
