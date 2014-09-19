#ifndef IMAGE_RESOURCE_H
#define IMAGE_RESOURCE_H

#include <cstring>
#include <cstdlib>
#include <stdint.h>
#include <fcntl.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <tr1/unordered_map>

#include <glog/logging.h>
#include <google/gflags.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

DECLARE_int32(top_n_limit);
DECLARE_bool(open_search_function);

namespace image {

typedef std::vector<size_t> PostingList;
typedef std::map<int, PostingList> ImageCategoryIndex;
typedef std::vector<float> SimWeightVector;
typedef std::vector<SimWeightVector> SimWeightIndex;
typedef std::vector<std::string> IdToFileName;
typedef std::map<std::string, size_t> ImageUniquer;
typedef std::map<int, std::string> CategoryIdToName;
typedef struct sFeature {
  std::string token;
  float weight;
  sFeature(const std::string &token_, const std::string &weight_) : token(token_) {
    weight = atof(weight_.c_str());
  };
} Feature;
typedef std::vector<Feature> FeatureVec;
typedef std::map<std::string, FeatureVec> FeatureVecMap;

class ImageResource {
 public:
  bool Init();

  // for resource
  std::vector<std::string> name_vector_;
  CategoryIdToName paipai_name_vector_;

  int LoadClassNameVector();
  // for index
  int LoadImageIndex(const char* filename);
  // for semantic
  int LoadImageSemanticFeature();
  int LoadImageSemanticIndex(const char* filename);

  ImageUniquer image_uniquer_;
  // for search
  IdToFileName index_id_to_filename_;
  SimWeightIndex sim_weight_index_;
  ImageCategoryIndex image_class_index_;
  // for semantic
  IdToFileName index_id_to_filename_semantic_;
  SimWeightIndex sim_weight_index_semantic_;
  ImageCategoryIndex image_class_index_semantic_;
  // for semantic
  FeatureVecMap feature_vec_map_;
};

}

#endif
