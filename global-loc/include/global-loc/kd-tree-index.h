#ifndef GLOBAL_LOC_KD_TREE_INDEX_H_
#define GLOBAL_LOC_KD_TREE_INDEX_H_

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <aslam/common/memory.h>
#include <glog/logging.h>
#include <nabo/nabo.h>

class KDTreeIndex {
  public:
    typedef Eigen::MatrixXf DescriptorMatrixType;
    typedef Nabo::NearestNeighbourSearch<float> NNSearch;
    // Switch touch statistics (NNSearch::TOUCH_STATISTICS) off for performance.
    static constexpr int kCollectTouchStatistics = 0;
    // Kd-tree search options. ALLOW_SELF_MATCH means that a point which is
    // equal to the query will be returned in the results.
    static constexpr unsigned kSearchOptionsDefault =
            NNSearch::ALLOW_SELF_MATCH | NNSearch::SORT_RESULTS;
    // Epsilon approximation factor for kd-tree backtracking.
    static constexpr float kSearchNNEpsilon = 0.1;

    KDTreeIndex(const unsigned descriptor_size): descriptor_size_(descriptor_size) {}

    // Adds descriptors to an internal waiting list. These descriptors will be
    // added on the next time the index is queried.
    void AddDescriptors(const DescriptorMatrixType& descriptors) {
        pending_descriptor_blocks_.push_back(
                aligned_shared<Eigen::MatrixXf>(descriptors));
    }

    void RefreshIndex() const {
        if (pending_descriptor_blocks_.empty())
            return;

        int total_num_descriptors_to_add = 0;
        for (const std::shared_ptr<Eigen::MatrixXf>& descriptor_block :
             pending_descriptor_blocks_) {
            CHECK(descriptor_block != nullptr);
            total_num_descriptors_to_add += descriptor_block->cols();
        }
        int old_num_descriptors = index_data_.cols();
        int new_num_descriptors =
                total_num_descriptors_to_add + old_num_descriptors;

        index_data_.conservativeResize(descriptor_size_, new_num_descriptors);

        int curr_offset = old_num_descriptors;
        for (const std::shared_ptr<Eigen::MatrixXf>& descriptor_block :
             pending_descriptor_blocks_) {
            const DescriptorMatrixType& descriptors = *descriptor_block;
            int num_descriptors = descriptors.cols();
            index_data_.block(0, curr_offset, descriptor_size_, num_descriptors) =
                    descriptors;
            curr_offset += num_descriptors;
        }

        if (index_data_.cols() == 0) {
            pending_descriptor_blocks_.clear();
            index_.reset();
            return;
        }
        index_.reset(
                NNSearch::createKDTreeLinearHeap(
                    index_data_, descriptor_size_, kCollectTouchStatistics));
        pending_descriptor_blocks_.clear();
    }

     // Finds the n nearest neighbors for a given query feature.
     // This function is thread-safe.
    inline void GetNNearestNeighbors(
            const Eigen::MatrixXf& query_features, int num_neighbors,
            Eigen::MatrixXi* indices, Eigen::MatrixXf* distances,
            const float max_distance=std::numeric_limits<float>::infinity()) const {
        CHECK_NOTNULL(indices);
        CHECK_NOTNULL(distances);
        CHECK_EQ(indices->rows(), num_neighbors)
                << "The indices parameter must be pre-allocated to hold all results.";

        CHECK_EQ(distances->rows(), num_neighbors)
                << "The distances parameter must be pre-allocated to hold all results.";

        // Lazy refresh of the index if more data was added in the meantime.
        RefreshIndex();
        if (!index_) {
            indices->setConstant(-1);
            distances->setConstant(std::numeric_limits<float>::infinity());
            LOG(WARNING) << "The kd-tree index is not available.";
            return;
        }
        const NNSearch& index_ref = *index_;
        index_ref.knn(
                query_features, *indices, *distances, num_neighbors, kSearchNNEpsilon,
                kSearchOptionsDefault, max_distance);
    }

  protected:
    mutable std::shared_ptr<NNSearch> index_;
    mutable Eigen::MatrixXf index_data_;
    mutable std::vector<std::shared_ptr<Eigen::MatrixXf> > pending_descriptor_blocks_;
    const unsigned descriptor_size_;
};

#endif // GLOBAL_LOC_KD_TREE_INDEX_H_
