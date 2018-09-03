#ifndef GLOBAL_LOC_PLACE_RETRIEVAL_H_
#define GLOBAL_LOC_PLACE_RETRIEVAL_H_

#include <memory>
#include <vector>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <vi-map/vi-map.h>
#include <vi-map/unique-id.h>

#include "global-loc/tensorflow-net.h"
#include "global-loc/kd-tree-index.h"
#include "global-loc/pca-reduction.h"
#include "global-loc/descriptor_index.pb.h"

class PlaceRetrieval {
  public:
    PlaceRetrieval(const std::string model_path);

    void BuildIndexFromMap(
            const vi_map::VIMap& map,
            global_loc::proto::DescriptorIndex* proto_index);

    void LoadIndex(const global_loc::proto::DescriptorIndex& proto_index);

    void RetrieveNearestNeighbors(
            const cv::Mat& input_image, const unsigned num_neighbors,
            const float max_distance,
            vi_map::VisualFrameIdentifierList* retrieved_frame_identifiers);

  private:
    TensorflowNet network_;
    std::mutex network_mutex_;
    std::unique_ptr<KDTreeIndex> index_;
    std::unique_ptr<PcaReduction> pca_reduction_;
    vi_map::VisualFrameIdentifierList indexed_frame_identifiers_;
};

#endif  // GLOBAL_LOC_PLACE_RETRIEVAL_H_
