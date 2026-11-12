#ifndef HGCAL_CM_APPLY_HELPERS_HXX
#define HGCAL_CM_APPLY_HELPERS_HXX
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace hgcal_cm_apply {
using ROOT::VecOps::RVec;
using Matrix = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
struct Prediction {
  RVec<float> prediction, adc_corrected;
};
struct Model {
  RVec<int> columns, fitted;
  Matrix weights;
  Eigen::VectorXd intercept;
  unsigned global_size;
  Model(const RVec<int>& selected, const RVec<double>& coefficients,
        const RVec<double>& offsets, const RVec<int>& available, unsigned size)
    : columns(selected), fitted(available), weights(offsets.size(),selected.size()),
      intercept(offsets.size()), global_size(size) {
    if (fitted.size()!=offsets.size() || coefficients.size()!=offsets.size()*selected.size())
      throw std::runtime_error("Invalid application model dimensions");
    for (int j:columns) if (j<0 || unsigned(j)>=global_size)
      throw std::runtime_error("Invalid application feature mapping");
    for (unsigned c=0;c<offsets.size();++c) {
      intercept[c]=fitted[c] ? offsets[c] : 0.;
      for (unsigned j=0;j<selected.size();++j)
        weights(c,j)=fitted[c] ? coefficients[c*selected.size()+j] : 0.;
    }
  }
};

class Predictor {
  std::shared_ptr<const Model> model;
public:
  Predictor(const RVec<int>& columns,const RVec<double>& coefficients,
            const RVec<double>& intercept,const RVec<int>& fitted,unsigned global_size)
    : model(std::make_shared<Model>(columns,coefficients,intercept,fitted,global_size)) {}
  Prediction operator()(const RVec<float>& features,const RVec<float>& targets) const {
    const auto& m=*model;
    if (features.size()!=m.global_size || targets.size()!=m.fitted.size())
      throw std::runtime_error("Application feature/target width changed");
    Eigen::VectorXd x(m.columns.size());
    for (unsigned j=0;j<m.columns.size();++j) {
      x[j]=features[m.columns[j]];
      if (!std::isfinite(x[j])) throw std::runtime_error("Nonfinite filled application feature");
    }
    const Eigen::VectorXd values=m.weights*x+m.intercept;
    const auto nan=std::numeric_limits<float>::quiet_NaN();
    Prediction result{RVec<float>(targets.size(),nan),RVec<float>(targets.size(),nan)};
    for (unsigned c=0;c<targets.size();++c) {
      if (!m.fitted[c] || !std::isfinite(targets[c])) continue;
      result.prediction[c]=float(values[c]);
      result.adc_corrected[c]=targets[c]-result.prediction[c];
      if (!std::isfinite(result.prediction[c]) || !std::isfinite(result.adc_corrected[c]))
        throw std::runtime_error("Nonfinite application prediction or corrected ADC");
    }
    return result;
  }
};

inline ROOT::RDF::RNode define_predictions(ROOT::RDF::RNode node,const Predictor& predictor) {
  return node.Define("application_result",predictor,{"application_features","adc_selected"})
             .Define("prediction","application_result.prediction")
             .Define("adc_corrected","application_result.adc_corrected");
}
}
#endif
