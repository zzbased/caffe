#include <sstream>
#include "classify_impl.h"
#include "caffe/caffe.hpp"

// DEFINE_string(model_def, "../examples/imagenet_deploy.prototxt", "The model definition file.");
// DEFINE_string(pretrained_model, "../examples/imagenet_model", "The pretrained model.");
DEFINE_string(model_def, "/data/vincentyao/gdt_creek_image/data/imagenet_deploy.prototxt", "The model definition file.");
DEFINE_string(pretrained_model, "/data/vincentyao/gdt_creek_image/data/imagenet_model", "The pretrained model.");
DEFINE_int32(row_col_num, 227, "Row/Column number");
DEFINE_bool(gpu, false, "use gpu for computation");
DEFINE_int32(similarity_weight_layer, 23, "");
// DEFINE_string(maxent_model, "../examples/paipai_model_23layer_8_feature", "The paipai model file.");
DEFINE_string(maxent_model, "/data/vincentyao/gdt_creek_image/data/paipai_model_23layer_8_feature", "The paipai model file.");

namespace image {

bool ClassifyImpl::Init(const ImageResource * resource) {

  Caffe::set_phase(Caffe::TEST);
  if (FLAGS_gpu) {
    LOG(INFO) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  //读入模型定义
  NetParameter test_net_param;
  ReadProtoFromTextFile(FLAGS_model_def, &test_net_param);

  caffe_test_net_ = new Net<float>(test_net_param);
  NetParameter trained_net_param;
  //初始化模型参数
  ReadProtoFromBinaryFile(FLAGS_pretrained_model, &trained_net_param);
  caffe_test_net_->CopyTrainedLayersFrom(trained_net_param);

  if (!me_model_.load_from_file(FLAGS_maxent_model)) {
    LOG(ERROR) << "me_model_ load_from_file error";
    return false;
  }

  //赋值持有资源
  image_resource_ = resource;
  return true;
}

bool ClassifyImpl::UnInit() {
  if (caffe_test_net_) {
    delete caffe_test_net_;
    caffe_test_net_ = NULL;
  }
  return true;
}

std::string itoa(int value) {
  std::string str;
  std::stringstream ss;
  ss << value;
  ss >> str;
  return str;
}

bool SortMaxentResultVectorFunction(const std::pair<std::string, double> & x,
                                    const std::pair<std::string, double> & y) {
  return x.second > y.second;
}

int ClassifyImpl::ImageClassify(const std::string & filename, int top_n_res, int class_type) {
  if (top_n_res > kTopNumber) {
    return -1;
  }

  Datum datum;
  const static int kUndefinedLabel = 0;
  try {
    if (!ReadImageToDatum(filename.c_str(), kUndefinedLabel,
                          FLAGS_row_col_num, FLAGS_row_col_num, &datum)) {
      LOG(ERROR) << "ReadImageToDatum Error";
      return -1;
    }
  } catch (...) {
    LOG(ERROR) << "Catch a exception!";
    return -1;
  }

  vector<Blob<float>*>& input_blobs = caffe_test_net_->input_blobs();
  //now set input_blobs with datum
  //uint8_t -> float
  //net , data layer, prefetch data
  CHECK_EQ(input_blobs.size(), 1) << "input_blobs.size() != 1";
  Blob<float>* blob = input_blobs[0];
  float* input_data = blob->mutable_cpu_data();
  float scale = 1;
  const string& data = datum.data();
  int datum_size = datum.channels() * datum.height() * datum.width();
  VLOG(1) << "Datum size:" << datum_size;
  if (data.size()) {
    for (int j = 0; j < datum_size; ++j) {
      input_data[j] = (static_cast<float>((uint8_t)data[j])) * scale;
    }
  } else {
    for (int j = 0; j < datum_size; ++j) {
      input_data[j] = (datum.float_data(j)) * scale;
    }
  }

  //predict
  const vector<Blob<float>*>&  output_blobs = caffe_test_net_->ForwardPrefilled();
  similarity_weight_vec_.clear();
  if (class_type == image::ClassifyRequest::SEARCH) {
    //store layer weight to calc similarity
    const vector<Blob<float>*>& weight_vec = caffe_test_net_->top_vecs_[FLAGS_similarity_weight_layer];
    for (size_t i = 0; i < weight_vec[0]->count(); i++) {
      similarity_weight_vec_.push_back(weight_vec[0]->cpu_data()[i]);
    }
  } else if (class_type == image::ClassifyRequest::CLASSIFY_PAIPAI) {
    //paipai分类器用第23层特征,这个是固定下来的.
    const vector<Blob<float>*>& weight_vec = caffe_test_net_->top_vecs_[23];
    for (size_t i = 0; i < weight_vec[0]->count(); i++) {
      similarity_weight_vec_.push_back(weight_vec[0]->cpu_data()[i]);
    }
  }
  //output result
  VLOG(1) << "output blobs size:" << output_blobs.size();
  for (size_t i = 0; i < output_blobs.size(); i++) {
    int num = output_blobs[i]->num();
    int dim = output_blobs[i]->count() / output_blobs[i]->num();
    VLOG(1) << "output_blobs:" << output_blobs[i]->num() << " " << output_blobs[i]->count();
    const float* bottom_data = output_blobs[i]->cpu_data();

    for (int k = 0; k < num; ++k) {
      //top kTopNumber
      float maxval[kTopNumber] = { -1};
      int max_id[kTopNumber] = {0};
      for (int j = 0; j < dim; ++j) {
        float temp_value = bottom_data[k * dim + j];
        if (temp_value > maxval[0]) {
          for (int d = kTopNumber - 1; d >= 0; d--) {
            if (maxval[d] < temp_value) {
              for (int d2 = 0; d2 < d; d2++) {
                maxval[d2] = maxval[d2 + 1];
                max_id[d2] = max_id[d2 + 1];
              }
              maxval[d] = temp_value;
              max_id[d] = j;
              break;
            }
          }
        }
      }

      printf("%s\n", filename.c_str());
      //for (int d=kTopNumber-1; d>=0; d--) {
      //  printf("Rank%d : %d, %s, %f\n", kTopNumber-d, max_id[d], image_resource_->name_vector_[max_id[d]].c_str(), maxval[d]);
      //}
      printf("class_type:%d\n", class_type);
      if (class_type == image::ClassifyRequest::CLASSIFY || class_type == image::ClassifyRequest::SEARCH) {
        //只取前top_n_res
        for (int d = kTopNumber - 1; d >= kTopNumber - top_n_res; d--) {
          ::image::ClassifyResult* result = response_message_.add_rsp_res();
          result->set_category_name(image_resource_->name_vector_[max_id[d]]);
          result->set_category_id(max_id[d]);
          result->set_category_weight(maxval[d]);
        }
      } else if (class_type == image::ClassifyRequest::CLASSIFY_PAIPAI) {
        //maxent classify
        ME_Sample sample;
        for (int d = kTopNumber - 1; d >= 0; d--) {
          sample.add_feature(itoa(max_id[d]), maxval[d]);
        }
        for (size_t d = 0; d < similarity_weight_vec_.size(); d++) {
          sample.add_feature(itoa(4096 + 2000 + d), similarity_weight_vec_[d]);
        }
        std::vector<double>  membp = me_model_.classify(sample);
        std::vector< std::pair<std::string, float> > sort_vec;
        for (int d = 0; d < membp.size(); d++) {
          sort_vec.push_back(std::make_pair(me_model_.get_class_label(d), membp[d]));
        }
        std::sort(sort_vec.begin(), sort_vec.end(), SortMaxentResultVectorFunction);

        for (int d = 0; d < sort_vec.size() && d < top_n_res; d++) {
          ::image::ClassifyResult* result = response_message_.add_rsp_res();
          int id = atoi(sort_vec[d].first.c_str());
          result->set_category_name(image_resource_->paipai_name_vector_[id]);
          result->set_category_id(id);
          result->set_category_weight(sort_vec[d].second);
        }
      }
    }
  }
  return 0;
}

int ClassifyImpl::ImageSimilarity(const std::string & filename, const std::string & filename2) {
  int feature_layer_num = request_message_.feature_layer();
  if (feature_layer_num >= 27 || feature_layer_num <= 1) {
    // feature layer should in correct interval;
    feature_layer_num = 23;
  }

  Datum datum;
  const static int kUndefinedLabel = 0;
  try {
    if (!ReadImageToDatum(filename.c_str(), kUndefinedLabel, FLAGS_row_col_num, FLAGS_row_col_num, &datum)) {
      LOG(ERROR) << "ReadImageToDatum Error";
      return -1;
    }
  } catch (...) {
    LOG(ERROR) << "Catch a exception!";
    return -1;
  }
  vector<Blob<float>*>& input_blobs = caffe_test_net_->input_blobs();
  //now set input_blobs with datum
  //uint8_t -> float
  //net , data layer, prefetch data
  CHECK_EQ(input_blobs.size(), 1) << "input_blobs.size() != 1";
  Blob<float>* blob = input_blobs[0];
  float* input_data = blob->mutable_cpu_data();
  float scale = 1;
  const string& data = datum.data();
  int datum_size = datum.channels() * datum.height() * datum.width();

  if (data.size()) {
    for (int j = 0; j < datum_size; ++j) {
      input_data[j] = (static_cast<float>((uint8_t)data[j])) * scale;
    }
  } else {
    for (int j = 0; j < datum_size; ++j) {
      input_data[j] = (datum.float_data(j)) * scale;
    }
  }

  //predict
  const vector<Blob<float>*>&  output_blobs = caffe_test_net_->ForwardPrefilled();
  const vector<Blob<float>*>& caffe_feature_out = caffe_test_net_->top_vecs_[feature_layer_num];
  float * caffe_feature_res = new float[caffe_feature_out[0]->count()];
  for (int k = 0; k < caffe_feature_out[0]->count(); k++) {
    caffe_feature_res[k] = caffe_feature_out[0]->cpu_data()[k];
  }

  //another picture
  datum.Clear();
  try {
    if (!ReadImageToDatum(filename2.c_str(), kUndefinedLabel, FLAGS_row_col_num, FLAGS_row_col_num, &datum)) {
      LOG(ERROR) << "ReadImageToDatum Error";
      return -1;
    }
  } catch (...) {
    LOG(ERROR) << "Catch a exception!";
    return -1;
  }
  input_blobs = caffe_test_net_->input_blobs();
  //now set input_blobs with datum
  //uint8_t -> float
  //net , data layer, prefetch data
  CHECK_EQ(input_blobs.size(), 1) << "input_blobs.size() != 1";
  blob = input_blobs[0];
  input_data = blob->mutable_cpu_data();
  const string& data2 = datum.data();
  datum_size = datum.channels() * datum.height() * datum.width();
  if (data2.size()) {
    for (int j = 0; j < datum_size; ++j) {
      input_data[j] = (static_cast<float>((uint8_t)data2[j])) * scale;
    }
  } else {
    for (int j = 0; j < datum_size; ++j) {
      input_data[j] = (datum.float_data(j)) * scale;
    }
  }

  //predict
  const vector<Blob<float>*>&  output_blobs2 = caffe_test_net_->ForwardPrefilled();
  const vector<Blob<float>*>& caffe_feature_out_2 = caffe_test_net_->top_vecs_[feature_layer_num];

  float ab = 0, a2 = 0, b2 = 0;
  for (int k = 0; k < caffe_feature_out_2[0]->count(); k++) {
    ab += (caffe_feature_out_2[0]->cpu_data()[k] * caffe_feature_res[k]);
    a2 += (caffe_feature_res[k] * caffe_feature_res[k]);
    b2 += (caffe_feature_out_2[0]->cpu_data()[k] * caffe_feature_out_2[0]->cpu_data()[k]);
  }
  float similarity = 0;
  if (a2 < 0.000001 || b2 < 0.000001) {
    similarity = 0;
  } else {
    similarity = ab / sqrt(a2) / sqrt(b2);
  }
  response_message_.set_similarity(similarity);

  if (caffe_feature_res) {
    delete []caffe_feature_res;
    caffe_feature_res = NULL;
  }
  return 0;
}

int ClassifyImpl::CalcSimilairtyInSearch(const SimWeightVector & other_weight_vec, float & similarity) {

  if (other_weight_vec.size() != similarity_weight_vec_.size()) {
    similarity = 0;
    return -1;
  }
  float aa = 0, ab = 0, bb = 0;
  for (size_t i = 0; i < similarity_weight_vec_.size(); i++) {
    aa += similarity_weight_vec_[i] * similarity_weight_vec_[i];
    bb += other_weight_vec[i] * other_weight_vec[i];
    ab += similarity_weight_vec_[i] * other_weight_vec[i];
  }
  if (aa < 0.000001 || bb < 0.000001) {
    similarity = 0;
    return -1;
  } else {
    similarity = ab / sqrt(aa) / sqrt(bb);
    if (similarity > 1.0)
      similarity = 1.0;
  }
  return 0;
}

//for 排序search结果
bool SortSearchResultVectorFunction(const std::pair<int, float> & x, const std::pair<int, float> & y) {
  return x.second > y.second;
}

int ClassifyImpl::SortSearchResultMap(const SearchResultContainer & search_map, SearchResultSortVector & search_vec) {
  for (SearchResultContainer::const_iterator it = search_map.begin(); it != search_map.end(); it++) {
    search_vec.push_back(std::make_pair(it->first, it->second));
  }
  std::sort(search_vec.begin(), search_vec.end(), SortSearchResultVectorFunction);
  return 0;
}

//for 排序feature token结果
bool SortMapFunction(const std::pair<std::string, float> & lhs, const std::pair<std::string, float> & rhs) {
  return lhs.second > rhs.second;
}
void SortTokenMap(const std::map<std::string, float> & map, std::vector< std::pair<std::string, float> > *vec) {
  for (std::map<std::string, float>::const_iterator it = map.begin(); it != map.end(); it++) {
    vec->push_back(std::make_pair(it->first, it->second));
  }
  std::sort(vec->begin(), vec->end(), SortMapFunction);
}

int ClassifyImpl::ImageSearch(const std::string & filename, int top_n_res, int class_type) {

  const IdToFileName* index_id_to_filename = NULL;
  const SimWeightIndex * sim_weight_index = NULL;
  const ImageCategoryIndex * image_class_index = NULL;
  //search与semantic用的图片索引不一样,search是对gdt广告库所建的索引,而semantic是paipai分析的图片
  if (class_type == image::ClassifyRequest::SEARCH) {
    index_id_to_filename = &(image_resource_->index_id_to_filename_);
    sim_weight_index = &(image_resource_->sim_weight_index_);
    image_class_index = &(image_resource_->image_class_index_);
  } else if (class_type == image::ClassifyRequest::SEMANTIC) {
    index_id_to_filename = &(image_resource_->index_id_to_filename_semantic_);
    sim_weight_index = &(image_resource_->sim_weight_index_semantic_);
    image_class_index = &(image_resource_->image_class_index_semantic_);
  } else {
    return -1;
  }

  if (ImageClassify(filename, 5, image::ClassifyRequest::SEARCH) != 0) {
    return -1;
  }
  //according class id, search similarity images in the index.
  search_res_container_.clear();
  ImageCategoryIndex::const_iterator it;
  VLOG(1) << "response_message_.rsp_res_size(): " << response_message_.rsp_res_size();
  for (int i = 0; i < response_message_.rsp_res_size() && i < FLAGS_top_n_limit; i++) {
    int class_id = response_message_.rsp_res(i).category_id();
    it = image_class_index->find(class_id);
    if (it != image_class_index->end()) {
      const PostingList &postinglist = it->second;
      for (size_t k = 0; k < postinglist.size(); k++) {
        search_res_container_.insert(std::make_pair(postinglist[k], 0.0));
      }
    }
  }

  //calc similarity for candidates
  VLOG(0) << "Search candidates num:" << search_res_container_.size();
  for (SearchResultContainer::iterator it1 = search_res_container_.begin();
       it1 != search_res_container_.end(); it1++) {
    float sim = 0;
    const SimWeightVector & other_weight_vec = (*sim_weight_index)[it1->first];
    CalcSimilairtyInSearch(other_weight_vec, sim);
    it1->second = sim;
  }

  //sort map
  search_res_sort_vec_.clear();
  SortSearchResultMap(search_res_container_, search_res_sort_vec_);
  VLOG(0) << "Search candidates num1:" << search_res_container_.size()
          << "; num2:" << search_res_sort_vec_.size();

  int add_num = 0;
  if (class_type == image::ClassifyRequest::SEARCH) {
    for (int i = 0; i < search_res_sort_vec_.size() && add_num < top_n_res; i++) {
      if (search_res_sort_vec_[i].second >= request_message_.min_sim_thres()
          && search_res_sort_vec_[i].second <= request_message_.max_sim_thres()) {
        ::image::SearchResult* add_search_res =  response_message_.add_search_res();
        add_search_res->set_search_file_name((*index_id_to_filename)[search_res_sort_vec_[i].first]);
        add_search_res->set_search_similarity(search_res_sort_vec_[i].second);
        add_num++;
      } else {
        continue;
      }
    }

  } else if (class_type == image::ClassifyRequest::SEMANTIC) {
    //语义检索
    std::map<std::string, float> token_map;
    //100可调
    for (int i = 0; i < search_res_sort_vec_.size() && add_num < 100; i++) {
      const std::string & imagename = (*index_id_to_filename)[search_res_sort_vec_[i].first];
      FeatureVecMap::const_iterator it = image_resource_->feature_vec_map_.find(imagename);
      if (it != image_resource_->feature_vec_map_.end()) {
        for (size_t k = 0; k < it->second.size(); ++k) {
          if (token_map.find(it->second[k].token) != token_map.end()) {
            token_map[it->second[k].token] += it->second[k].weight * search_res_sort_vec_[i].second;
          } else {
            token_map[it->second[k].token] = it->second[k].weight * search_res_sort_vec_[i].second;
          }
        }
        add_num++;
      }
    }

    //sort tokenmap;
    VLOG(0) << "token_map.size() : " << token_map.size();
    if (token_map.size() < 1) {
      return -1;
    }
    std::vector< std::pair<std::string, float> > token_vec;
    SortTokenMap(token_map, &token_vec);
    for (size_t i = 0; i < token_vec.size() && i < top_n_res; ++i) {
      ::image::SearchFeature* add_search_feature =  response_message_.add_search_feature();
      add_search_feature->set_feature_token(token_vec[i].first);
      add_search_feature->set_feature_weight(token_vec[i].second);
    }
  }

  return 0;
}

}