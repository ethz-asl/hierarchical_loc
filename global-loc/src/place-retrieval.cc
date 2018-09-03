#include <string>
#include <math.h>

#include "global-loc/place-retrieval.h"
#include "global-loc/pca-reduction.h"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include <vi-map/vi-map.h>
#include <vi-map/unique-id.h>
#include <posegraph/pose-graph.h>
#include <posegraph/unique-id.h>
#include <aslam/common/pose-types.h>
#include <aslam/common/timer.h>
#include <maplab-common/aslam-id-proto.h>
#include <maplab-common/eigen-proto.h>
#include <maplab-common/progress-bar.h>
#include <kindr/minimal/quat-transformation.h>
#include <map-resources/resource-common.h>

DEFINE_uint64(
        subsampling, 1, "Interval at which the frames should be indexed.");
DEFINE_bool(
        index_pose, false,
        "Whether the frame pose (i.e. position vector and rotation matrix) should be"
        "added to the index.");
DEFINE_string(
        target_mission, "", "ID of the mission to be indexed.");
DEFINE_bool(
        use_pca, false,
        "Whether a PCA projection matrix should be computed from the descriptors.");
DEFINE_uint64(
        pca_dims, 40, "The number of dimensions of the PCA projection.");

PlaceRetrieval::PlaceRetrieval(const std::string model_path):
        network_(model_path, "image", "descriptor") {
    unsigned index_dims = FLAGS_use_pca ? FLAGS_pca_dims : network_.descriptor_size();
    index_.reset(new KDTreeIndex(index_dims));
}

void PlaceRetrieval::BuildIndexFromMap(
        const vi_map::VIMap& map,
        global_loc::proto::DescriptorIndex* proto_index) {
    CHECK_NOTNULL(proto_index);
    proto_index->set_descriptor_size(network_.descriptor_size());

    vi_map::MissionIdList mission_ids;
    if(!FLAGS_target_mission.empty()) {
        vi_map::MissionId target_mission;
        map.ensureMissionIdValid(FLAGS_target_mission, &target_mission);
        CHECK(target_mission.isValid());
        mission_ids.push_back(target_mission);
    } else {
        map.getAllMissionIdsSortedByTimestamp(&mission_ids);
        CHECK(!mission_ids.empty());
    }

    for (const vi_map::MissionId& mission_id : mission_ids) {
        pose_graph::VertexIdList vertex_ids;
        map.getAllVertexIdsInMissionAlongGraph(mission_id, &vertex_ids);
        CHECK(!vertex_ids.empty());

        unsigned num_indexed = 1 + (vertex_ids.size() - 1) / FLAGS_subsampling;
        LOG(INFO) << "Computing " << num_indexed
                  << " descriptors for mission " << mission_id.printString();
        common::ProgressBar progress_bar(num_indexed);

        for (int i = 0; i < vertex_ids.size(); i += FLAGS_subsampling) {
            progress_bar.increment();
            const pose_graph::VertexId& vertex_id = vertex_ids[i];
            const vi_map::Vertex& vertex = map.getVertex(vertex_id);
            if(!vertex.numFrames())
                continue;

            unsigned frame_index = 0;  // Add only the first frame

            global_loc::proto::DescriptorIndex::Frame* proto_frame =
                    proto_index->add_frames();
            vertex_id.serialize(proto_frame->mutable_vertex_id());
            proto_frame->set_frame_index(frame_index);
            backend::ResourceIdSet resource_ids;
            vertex.getFrameResourceIdsOfType(
                    frame_index, backend::ResourceType::kRawImage, &resource_ids);
            if(!resource_ids.size()) {
                LOG(WARNING)
                    << "Frame " << frame_index << " of vertex " << vertex_id
                    << " had no resource, skipping.";
                proto_index->mutable_frames()->RemoveLast();
                continue;
            }
            proto_frame->set_resource_name(resource_ids.begin()->hexString());

            cv::Mat image;
            CHECK(map.getRawImage(vertex, frame_index, &image))
                << "Vertex " << vertex_id << " does not have a raw image for frame "
                << frame_index;
            TensorflowNet::DescriptorType descriptor;
            descriptor.resize(network_.descriptor_size(), Eigen::NoChange);
            network_.PerformInference(image, &descriptor);
            Eigen::MatrixXf generic_descriptor = descriptor;  // copy constructor crashes
            common::eigen_proto::serialize(
                    generic_descriptor, proto_frame->mutable_global_descriptor());

            if(FLAGS_index_pose) {
                aslam::Transformation transf = map.getVertex_T_G_I(vertex_id);
                common::eigen_proto::serialize(
                        Eigen::MatrixXd(transf.getPosition()),
                        proto_frame->mutable_position_vector());
                common::eigen_proto::serialize(
                        Eigen::MatrixXd(transf.getRotationMatrix()),
                        proto_frame->mutable_rotation_matrix());
            }
        }
    }
}

void PlaceRetrieval::LoadIndex(
        const global_loc::proto::DescriptorIndex& proto_index) {
    CHECK_EQ(proto_index.descriptor_size(), network_.descriptor_size());

    KDTreeIndex::DescriptorMatrixType descriptors(
            network_.descriptor_size(), proto_index.frames().size());
    unsigned i = 0;

    LOG(INFO) << "Loading " << proto_index.frames_size()
              << " reference descriptors into index.";
    common::ProgressBar progress_bar(proto_index.frames_size());
    for (const global_loc::proto::DescriptorIndex::Frame& proto_frame :
            proto_index.frames()) {
        pose_graph::VertexId vertex_id;
        vertex_id.deserialize(proto_frame.vertex_id());
        size_t frame_index = proto_frame.frame_index(); // static_cast<size_t>()
        indexed_frame_identifiers_.emplace_back(vertex_id, frame_index);

        KDTreeIndex::DescriptorMatrixType descriptor;
        common::eigen_proto::deserialize(proto_frame.global_descriptor(), &descriptor);
        CHECK_EQ(descriptor.rows(), network_.descriptor_size());

        descriptors.col(i++) = descriptor;
        progress_bar.increment();
    }

    if(FLAGS_use_pca) {
        pca_reduction_.reset(new PcaReduction(descriptors));
        pca_reduction_->project(&descriptors, FLAGS_pca_dims);
    }
    index_->AddDescriptors(descriptors);
    index_->RefreshIndex();
}

void PlaceRetrieval::RetrieveNearestNeighbors(
        const cv::Mat& input_image, const unsigned num_neighbors,
        const float max_distance,
        vi_map::VisualFrameIdentifierList* retrieved_frame_identifiers) {
    TensorflowNet::DescriptorType descriptor;
    descriptor.resize(network_.descriptor_size(), Eigen::NoChange);
    network_mutex_.lock();
    timing::Timer timer_inference("Deep Relocalization: Compute descriptor");
    network_.PerformInference(input_image, &descriptor);
    timer_inference.Stop();
    network_mutex_.unlock();


    KDTreeIndex::DescriptorMatrixType descriptor_matrix(descriptor);
    if(FLAGS_use_pca) {
        CHECK_NOTNULL(pca_reduction_);
        pca_reduction_->project(&descriptor_matrix, FLAGS_pca_dims);
    }

    Eigen::MatrixXi indices;
    indices.resize(num_neighbors, 1);
    Eigen::MatrixXf distances;
    distances.resize(num_neighbors, 1);
    timing::Timer timer_get_nn("Deep Relocalization: Get neighbors");
    index_->GetNNearestNeighbors(
            descriptor_matrix, num_neighbors, &indices, &distances, max_distance);
    timer_get_nn.Stop();

    for (unsigned nn_search_idx = 0; nn_search_idx < num_neighbors; ++nn_search_idx) {
        const int nn_database_idx = indices(nn_search_idx, 0);
        const float nn_distance = distances(nn_search_idx, 0);
        if (nn_database_idx == -1 ||
                nn_distance == std::numeric_limits<float>::infinity()) {
            break;  // No more results
        }
        retrieved_frame_identifiers->push_back(
                indexed_frame_identifiers_[nn_database_idx]);
    }
}
