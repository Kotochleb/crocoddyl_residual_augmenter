#include "crocoddyl/core/utils/exception.hpp"

namespace residual_augmenter {
template <typename Scalar>
CostModelResidualAugmenterTpl<Scalar>::CostModelResidualAugmenterTpl(
    const std::shared_ptr<Base> &model)
    : Base(model->get_state(), model->get_activation(), model->get_nu()),
      wrapped_model_(model),
{}

template <typename Scalar>
CostModelResidualAugmenterTpl<Scalar>::CostModelResidualAugmenterTpl(
    std::shared_ptr<typename Base::StateAbstract> state,
    std::shared_ptr<ResidualModelAbstract> residual)
    : Base(state, residual) {}

template <typename Scalar>
CostModelResidualAugmenterTpl<Scalar>::~CostModelResidualAugmenterTpl() {}

template <typename Scalar>
void CostModelResidualAugmenterTpl<Scalar>::calc(
    const std::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u) {
  Data *d = static_cast<Data *>(data.get());

  const std::size_t nr = wrapped_model_.get_nr();
  const std::size_t nu = wrapped_model_.get_nu();
  const std::size_t nx = wrapped_model_.get_nx();

  wrapped_model_->calc(d->wrapped_data, x.head(nx), u.head(nu));

  d->cost = d->wrapped_data_->cost;
}

template <typename Scalar>
void CostModelResidualAugmenterTpl<Scalar>::calc(
    const std::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x) {
  Data *d = static_cast<Data *>(data.get());

  const std::size_t nr = wrapped_model_.get_nr();
  const std::size_t nx = wrapped_model_.get_nx();

  wrapped_model_->calc(d->wrapped_data, x.head(nx));

  d->cost = d->wrapped_data_->cost;
}

template <typename Scalar>
void CostModelResidualAugmenterTpl<Scalar>::calcDiff(
    const std::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u) {
  Data *d = static_cast<Data *>(data.get());

  const std::size_t nr = wrapped_model_.get_nr();
  const std::size_t nu = wrapped_model_.get_nu();
  const std::size_t nx = wrapped_model_.get_nx();

  wrapped_model_->calcDiff(d->wrapped_data, x.head(nx), u.head(nu));

  d->Lu.head(nu) = d->wrapped_data_->Lu;
  d->Luu.topLeftCorner(nu, nu) = d->wrapped_data_->Luu;
  d->Lx.head(nx) = d->wrapped_data_->Lx;
  d->Lxu.topLeftCorner(nx, nu) = d->wrapped_data_->Lxu;
  d->Lxx.topLeftCorner(nu, nu) = d->wrapped_data_->Lxx;
}

template <typename Scalar>
void CostModelResidualAugmenterTpl<Scalar>::calcDiff(
    const std::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x) {
  Data *d = static_cast<Data *>(data.get());

  const std::size_t nr = wrapped_model_.get_nr();
  const std::size_t nu = wrapped_model_.get_nu();
  const std::size_t nx = wrapped_model_.get_nx();

  wrapped_model_->calcDiff(d->wrapped_data, x.head(nx));

  d->Lu.head(nu).setZero();
  d->Luu.topLeftCorner(nu, nu).setZero();
  d->Lx.head(nx) = d->wrapped_data_->Lx;
  d->Lxu.topLeftCorner(nx, nu).setZero();
  d->Lxx.topLeftCorner(nu, nu) = d->wrapped_data_->Lxx;
}

template <typename Scalar>
std::shared_ptr<CostDataAbstractTpl<Scalar>>
CostModelResidualAugmenterTpl<Scalar>::createData(
    DataCollectorAbstract *const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
void CostModelResidualAugmenterTpl<Scalar>::print(std::ostream &os) const {
  os << "CostModelResidual {" << *residual_ << ", " << *activation_ << "}";
}

template <typename Scalar>
const std::shared_ptr<CostModelAbstractTpl<Scalar>> &
CostModelResidualAugmenterTpl<Scalar>::get_wrapped_model() const {
  return wrapped_model_;
}

}  // namespace residual_augmenter
