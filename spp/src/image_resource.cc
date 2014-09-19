#include "image_resource.h"

// DEFINE_string(synset, "../data/imagenet_category_name.txt", "The imagenet synset file.");
// DEFINE_string(paipai_synset, "../data/paipai_category_name.txt", "The paipai synset file.");
DEFINE_string(synset, "/data/vincentyao/gdt_creek_image/data/imagenet_category_name.txt", "The imagenet synset file.");
DEFINE_string(paipai_synset, "/data/vincentyao/gdt_creek_image/data/paipai_category_name.txt", "The paipai synset file.");
//index_image.dat里只有7w条数据,使用的是20层的特征;allpic_index_image.dat里有30多w条数据,使用的是23层的特征.
//index_image_paipai_semantic.dat是最新的拍拍数据,23层特征.
//DEFINE_string(image_index_file, "../examples/index_image.dat", "image index file");

// DEFINE_string(image_index_file_for_search, "../examples/allpic_index_image.dat", "image index file");
// DEFINE_string(image_index_file_for_semantic, "../examples/index_image_paipai_semantic.dat", "image index file");
// DEFINE_string(image_feature, "../data/image_semantic_feature.dat", "");
DEFINE_string(image_index_file_for_search, "/data/vincentyao/gdt_creek_image/data/index_image.dat", "image index file");
DEFINE_string(image_index_file_for_semantic, "/data/vincentyao/gdt_creek_image/data/index_image.dat", "image index file");
DEFINE_string(image_feature, "/data/vincentyao/gdt_creek_image/data/image_semantic.dat", "");
DEFINE_int32(top_n_limit, 5, "use top_n_limit");
DEFINE_bool(open_search_function, true, "false,只保留image classify功能; true,增加image search功能");

namespace image {

bool SplitString(const std::string& input, const std::string& split_char,
                 std::vector<std::string>* split_result) {
  if (split_result == NULL) {
    return false;
  } else {
    split_result->clear();
  }

  std::string substring = "";
  size_t delim_length = split_char.size();
  for (std::string::size_type begin_index = 0; begin_index < input.size();) {
    std::string::size_type end_index = input.find(split_char, begin_index);
    if (end_index == std::string::npos) {
      substring = input.substr(begin_index);
      split_result->push_back(substring);
      return true;
    }
    if (end_index > begin_index) {
      substring = input.substr(begin_index, (end_index - begin_index));
      split_result->push_back(substring);
    }
    begin_index = end_index + delim_length;
  }
  return true;
}

bool ImageResource::Init() {
  if (0 != LoadClassNameVector()) {
    LOG(ERROR) << "LoadClassNameVector error";
    return false;
  }
  if (FLAGS_open_search_function) {
    if (0 != LoadImageIndex(FLAGS_image_index_file_for_search.c_str())) {
      LOG(ERROR) << "LoadImageIndex error";
      return false;
    }
    // for semantic
    if (0 != LoadImageSemanticFeature()) {
      LOG(ERROR) << "LoadImageSemanticFeature error";
      return false;
    }
    if (0 != LoadImageSemanticIndex(FLAGS_image_index_file_for_semantic.c_str())) {
      LOG(ERROR) << "LoadImageIndex error";
      return false;
    }
  }
  return true;
}

int ImageResource::LoadClassNameVector() {
  std::ifstream ifs;
  ifs.open(FLAGS_synset.c_str(), std::ifstream::in);
  if (!ifs.is_open()) {
    LOG(ERROR) << "open file failed. " << FLAGS_synset;
    return -1;
  }
  std::string synset_line;
  name_vector_.clear();
  while (getline(ifs, synset_line)) {
    std::size_t found = synset_line.find_last_of("\t");
    std::string name = synset_line.substr(found + 1);
    name_vector_.push_back(name);
  }
  LOG(ERROR) << "imagenet name_vector_.size : " << name_vector_.size();
  ifs.close();
  CHECK_EQ(name_vector_.size(), 1000) << "name_vector_.size != 1000";

  ifs.open(FLAGS_paipai_synset.c_str(), std::ifstream::in);
  if (!ifs.is_open()) {
    LOG(ERROR) << "open file failed. " << FLAGS_paipai_synset;
    return -1;
  }
  paipai_name_vector_.clear();
  while (getline(ifs, synset_line)) {
    std::size_t found = synset_line.find_last_of("\t");
    std::cout << synset_line << std::endl;
    int id = atoi(synset_line.substr(0, found).c_str());
    const std::string& name = synset_line.substr(found + 1);
    // std::cout << id << "," << name << std::endl;
    paipai_name_vector_.insert(std::make_pair<int, std::string>(id,name));
  }
  LOG(ERROR) << "paipai_name_vector_.size : " << paipai_name_vector_.size();
  ifs.close();
  return 0;
}

int ImageResource::LoadImageIndex(const char* filename) {
  std::ifstream ifs;
  ifs.open(filename, std::ifstream::in);
  std::string line;
  std::vector<std::string> split_result;
  int line_num = 0;

  image_uniquer_.clear();
  index_id_to_filename_.clear();
  sim_weight_index_.clear();
  image_class_index_.clear();

  while (getline(ifs, line)) {
    SplitString(line, " ", &split_result);
    int class_num = atoi(split_result[1].c_str());
    int weight_num = atoi(split_result[2].c_str());
    if (class_num < 5
        || weight_num != 4096
        || split_result.size() != (3 + 2 * class_num + weight_num)) {
      std::cerr << "format error:" << class_num
                << "," << class_num
                << "," << split_result.size() << std::endl;
      continue;
    }
    size_t image_id = 0;
    ImageUniquer::iterator it = image_uniquer_.find(split_result[0]);
    if (it == image_uniquer_.end()) {
      image_id = index_id_to_filename_.size();
      image_uniquer_.insert(std::make_pair(split_result[0], image_id));

      std::size_t found = split_result[0].find_last_of("/");
      const std::string& imagename = split_result[0].substr(found + 1);
      //index_id_to_filename_.push_back(split_result[0]);
      index_id_to_filename_.push_back(imagename); //去掉dir前缀,只保留image名字

      SimWeightVector weight_vec;
      for (size_t i = (class_num * 2 + 3); i < split_result.size(); i++) {
        weight_vec.push_back(atof(split_result[i].c_str()));
      }
      sim_weight_index_.push_back(weight_vec);
    } else {
      continue;
    }

    for (size_t i = 3; i < 3 + class_num * 2 && i < 3 + FLAGS_top_n_limit * 2; i = i + 2) {
      int class_id = atoi(split_result[i].c_str());
      ImageCategoryIndex::iterator it1 = image_class_index_.find(class_id);
      if (it1 == image_class_index_.end()) {
        PostingList posting_list;
        posting_list.push_back(image_id);
        image_class_index_.insert(std::make_pair(class_id, posting_list));
      } else {
        it1->second.push_back(image_id);
      }
    }
    line_num++;
    if (line_num % 100000 == 0) {
      LOG(INFO) << "Load image index. line:" << line_num;
    }

  }

  LOG(ERROR) << "image_uniquer_.size(): " << image_uniquer_.size()
          << " index_id_to_filename_.size(): " << index_id_to_filename_.size()
          << " sim_weight_index_.size(): " << sim_weight_index_.size()
          << " image_class_index_.size(): " << image_class_index_.size() << std::endl;

  return 0;
}

int ImageResource::LoadImageSemanticFeature() {
  std::ifstream ifs;
  ifs.open(FLAGS_image_feature.c_str(), std::ifstream::in);
  std::string synset_line;
  feature_vec_map_.clear();
  std::vector<std::string> split_result;
  while (getline(ifs, synset_line)) {
    SplitString(synset_line, "\t", &split_result);
    if (split_result.size() < 4)
      continue;
    int feature_num = atoi(split_result[1].c_str());
    FeatureVec feature_vec;
    for (int i = 0; i < feature_num; ++i) {
      feature_vec.push_back(
          Feature(split_result[2 + i * 2], split_result[2 + i * 2 + 1]));
    }
    feature_vec_map_.insert(std::make_pair(split_result[0], feature_vec));
  }
  LOG(ERROR) << "feature_vec_map_.size : " << feature_vec_map_.size();
  ifs.close();
  return 0;
}


int ImageResource::LoadImageSemanticIndex(const char* filename) {
  std::ifstream ifs;
  ifs.open(filename, std::ifstream::in);
  std::string line;
  std::vector<std::string> split_result;
  int line_num = 0;

  image_uniquer_.clear();
  index_id_to_filename_semantic_.clear();
  sim_weight_index_semantic_.clear();
  image_class_index_semantic_.clear();

  while (getline(ifs, line)) {
    SplitString(line, " ", &split_result);
    int class_num = atoi(split_result[1].c_str());
    int weight_num = atoi(split_result[2].c_str());
    if (class_num < 5
        || weight_num != 4096
        || split_result.size() != (3 + 2 * class_num + weight_num)) {
      std::cerr << "format error:" << class_num
                << "," << class_num
                << "," << split_result.size() << std::endl;
      continue;
    }
    size_t image_id = 0;
    ImageUniquer::iterator it = image_uniquer_.find(split_result[0]);
    if (it == image_uniquer_.end()) {
      image_id = index_id_to_filename_semantic_.size();
      image_uniquer_.insert(std::make_pair(split_result[0], image_id));

      std::size_t found = split_result[0].find_last_of("/");
      const std::string& imagename = split_result[0].substr(found + 1);
      //index_id_to_filename_semantic_.push_back(split_result[0]);
      index_id_to_filename_semantic_.push_back(imagename); //去掉dir前缀,只保留image名字

      SimWeightVector weight_vec;
      for (size_t i = (class_num * 2 + 3); i < split_result.size(); i++) {
        weight_vec.push_back(atof(split_result[i].c_str()));
      }
      sim_weight_index_semantic_.push_back(weight_vec);
    } else {
      continue;
    }

    for (size_t i = 3; i < 3 + class_num * 2 && i < 3 + FLAGS_top_n_limit * 2; i = i + 2) {
      int class_id = atoi(split_result[i].c_str());
      ImageCategoryIndex::iterator it1 = image_class_index_semantic_.find(class_id);
      if (it1 == image_class_index_semantic_.end()) {
        PostingList posting_list;
        posting_list.push_back(image_id);
        image_class_index_semantic_.insert(std::make_pair(class_id, posting_list));
      } else {
        it1->second.push_back(image_id);
      }
    }
    line_num++;
    if (line_num % 100000 == 0) {
      //LOG(INFO) << "Load image index. line:" << line_num;
    }
  }

  LOG(ERROR) << "image_uniquer_.size(): "
             << image_uniquer_.size()
             << " index_id_to_filename_semantic_.size(): "
             << index_id_to_filename_semantic_.size()
             << " sim_weight_index_semantic_.size(): "
             << sim_weight_index_semantic_.size()
             << " image_class_index_semantic_.size(): "
             << image_class_index_semantic_.size() << std::endl;

  return 0;
}

}
