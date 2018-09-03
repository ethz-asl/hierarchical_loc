#ifndef GLOBAL_LOC_PCA_REDUCTION_H_
#define GLOBAL_LOC_PCA_REDUCTION_H_

#include <Eigen/Core>

class PcaReduction {
  public:
    PcaReduction(const Eigen::MatrixXf& descriptors) {
        mean_ = descriptors.rowwise().mean();
        Eigen::MatrixXf centered = descriptors.colwise() - mean_;
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(centered, Eigen::ComputeFullU);
        projection_ = svd.matrixU().transpose();
    }

    void project(Eigen::MatrixXf* descriptors,
                 unsigned num_projected_dimensions) {
        CHECK_NOTNULL(descriptors);
        CHECK_EQ(descriptors->rows(), projection_.cols());
        CHECK_LE(num_projected_dimensions, projection_.cols());

        Eigen::MatrixXf centered = descriptors->colwise() - mean_;
        *descriptors = projection_.topRows(num_projected_dimensions)*centered;
    }

  private:
    Eigen::VectorXf mean_;
    Eigen::MatrixXf projection_;
};

#endif  // GLOBAL_LOC_PCA_REDUCTION_H_
