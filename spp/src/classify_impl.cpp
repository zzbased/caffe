#include "classify_impl.h"
#include "caffe/caffe.hpp"

DEFINE_int32(row_col_num, 227, "Row/Column number");
DEFINE_string(model_def, "../../examples/imagenet_deploy.prototxt", "The model definition file.");
DEFINE_string(pretrained_model, "../../examples/imagenet_model", "The pretrained model.");
DEFINE_string(synset, "../../data/imagenet_category_name.txt", "The imagenet synset file.");
DEFINE_bool(gpu, false, "use gpu for computation");
DEFINE_int32(similarity_weight_layer, 20, "");
DEFINE_string(image_index_file, "../../examples/index_image.dat", "image index file");
DEFINE_int32(top_n_limit, 5, "use top_n_limit");

bool SplitString(const std::string& input, const std::string& split_char, std::vector<std::string>* split_result) 
{
        if (split_result == NULL)
            return false;
        else 
            split_result->clear();

        std::string substring = "";
        size_t delim_length = split_char.size();
        for (std::string::size_type begin_index = 0; begin_index < input.size(); )
        {
            std::string::size_type end_index = input.find(split_char, begin_index);
            if (end_index == std::string::npos)
            {
                substring = input.substr(begin_index);
                split_result->push_back(substring);
                return true;
            }
            if (end_index > begin_index)
            {
                substring = input.substr(begin_index, (end_index - begin_index));
                split_result->push_back(substring);
            }
            begin_index = end_index + delim_length;
        }
        return true;
}


    

int ClassifyImpl::Init() {
  Caffe::set_phase(Caffe::TEST);

  if (FLAGS_gpu) {
    LOG(INFO) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  NetParameter test_net_param;
  ReadProtoFromTextFile(FLAGS_model_def, &test_net_param);

  caffe_test_net_ = new Net<float>(test_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(FLAGS_pretrained_model, &trained_net_param);
  caffe_test_net_->CopyTrainedLayersFrom(trained_net_param);

  if (0 != LoadClassNameVector())
    return -1;

  if (0 != LoadImageIndex(FLAGS_image_index_file.c_str())) 
    return -1;
       
  return 0;
}

int ClassifyImpl::UnInit() {
  if (caffe_test_net_) {
    delete caffe_test_net_;
    caffe_test_net_ = NULL;
  }
}
int ClassifyImpl::LoadClassNameVector() {
  std::ifstream ifs;
  ifs.open(FLAGS_synset.c_str(), std::ifstream::in);
  std::string synset_line;
  name_vector_.clear();
  while (getline(ifs, synset_line)) {
    std::size_t found = synset_line.find_last_of("\t");
    std::string name = synset_line.substr(found+1);
    //printf("name:%s\n", name.c_str());
    name_vector_.push_back(name);
  }
  //LOG(INFO) << "name_vector_.size : " << name_vector_.size();
  CHECK_EQ(name_vector_.size(), 1000) << "name_vector_.size != 1000";
  return 0;
}

int ClassifyImpl::LoadImageIndex(const char* filename) {
  std::ifstream ifs;
  ifs.open(filename, std::ifstream::in);
  std::string line;
  std::vector<std::string> split_result;
  while (getline(ifs, line)) {
    SplitString(line, " ", &split_result);
    //std::cout << split_result.size() << std::endl;
    //continue;
    int class_num = atoi(split_result[1].c_str());
    int weight_num = atoi(split_result[2].c_str());
    if (class_num != 5 || weight_num != 4096 || split_result.size() != (4096+10+3)) {
      std::cerr << "format error:" << class_num << "," << class_num << "," << split_result.size() << std::endl;
      continue;
    }
    size_t image_id = 0;
    ImageUniquer::iterator it = image_uniquer_.find(split_result[0]);
    if (it == image_uniquer_.end()) {
      image_id = index_id_to_filename_.size();
      image_uniquer_.insert( std::make_pair(split_result[0],image_id) );
      index_id_to_filename_.push_back(split_result[0]);

      SimWeightVector weight_vec;
      for (size_t i=13; i<(4096+10+3); i++) {
        weight_vec.push_back( atof(split_result[i].c_str()) );
      }
      sim_weight_index_.push_back(weight_vec);
    } else {
      continue;
    }

    for (size_t i=3; i<3+10; i=i+2) {
      int class_id = atoi(split_result[i].c_str());
      ImageCategoryIndex::iterator it1 = image_class_index_.find(class_id);
      if (it1 == image_class_index_.end()) {
        PostingList posting_list;
        posting_list.push_back(image_id);
        image_class_index_.insert( std::make_pair(class_id, posting_list) );
      } else {
        it1->second.push_back(image_id);
      }
    }
  }

  std::cout << "image_uniquer_.size(): " << image_uniquer_.size() 
  << " index_id_to_filename_.size(): " << index_id_to_filename_.size()
  << " sim_weight_index_.size(): " << sim_weight_index_.size()
  << " image_class_index_.size(): " << image_class_index_.size() << std::endl;
    
  return 0;
}

int ClassifyImpl::ImageClassify(const std::string & filename, int top_n_res, int is_search) {

  Datum datum;
  const static int kUndefinedLabel = 0;
  if (!ReadImageToDatum(filename.c_str(), kUndefinedLabel, FLAGS_row_col_num, FLAGS_row_col_num, &datum)) {
    LOG(ERROR) << "ReadImageToDatum Error"; 
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
  int datum_size_ = datum.channels() * datum.height() * datum.width();
  //printf("Datum size: %d\n", datum_size_);
  if (data.size()) {
    for (int j = 0; j < datum_size_; ++j) {
      input_data[j] = (static_cast<float>((uint8_t)data[j])) * scale;
    }
  } else {
    for (int j = 0; j < datum_size_; ++j) {
      input_data[j] = (datum.float_data(j)) * scale;
    }
  }
  
  //predict
  const vector<Blob<float>*>&  output_blobs = caffe_test_net_->ForwardPrefilled();

  if (is_search == 1) {
    similarity_weight_vec_.clear();
    //store layer weight to calc similarity
    const vector<Blob<float>*>& weight_vec = caffe_test_net_->top_vecs_[FLAGS_similarity_weight_layer];
    for (size_t i=0; i<weight_vec[0]->count(); i++) {
      similarity_weight_vec_.push_back(weight_vec[0]->cpu_data()[i]);
    }
  }
  //output result
  LOG(INFO) << "output blobs size:" << output_blobs.size();
  for (size_t i=0; i<output_blobs.size(); i++) {
      int num = output_blobs[i]->num();
      int dim = output_blobs[i]->count() / output_blobs[i]->num();
      LOG(INFO) << "output_blobs:" << output_blobs[i]->num() << " " << output_blobs[i]->count();
      const float* bottom_data = output_blobs[i]->cpu_data();
      LOG(INFO) << "";
      
      for (int k = 0; k < num; ++k) {        
        //top 5
        float maxval[kTopNumber] = {-1};
        int max_id[kTopNumber] = {0};
        for (int j = 0; j < dim; ++j) {
          float temp_value = bottom_data[k * dim + j];
          if (temp_value > maxval[0]) {
            for (int d=kTopNumber-1; d>=0; d--) {
              if (maxval[d] < temp_value) {
                for (int d2=0; d2 < d; d2++) {
                  maxval[d2] = maxval[d2+1];
                  max_id[d2] = max_id[d2+1];
                }
                maxval[d] = temp_value;
                max_id[d] = j;
                break;
              }
            }        
          }
        }

        //LOG(INFO) << "Maxid:" << max_id << ",maxval:" << maxval;
        printf("%s\n", filename.c_str());
        for (int d=kTopNumber-1; d>=0; d--) {
          printf("Rank%d : %d, %s, %f\n", kTopNumber-d, max_id[d], name_vector_[max_id[d]].c_str(), maxval[d]);
        }
	      for (int d=kTopNumber-1; d>=0; d--) {
    			::image::ClassifyResult* result = response_message_.add_rsp_res();
    			result->set_category_name(name_vector_[max_id[d]]);
    			result->set_category_id(max_id[d]);
    			result->set_category_weight(maxval[d]);
		    }
        
      }
  }
  return 0;
}

int ClassifyImpl::ImageSimilarity(const std::string & filename, const std::string & filename2) {
  int feature_layer_num = request_message_.feature_layer();
  if (feature_layer_num >= 27 || feature_layer_num <= 1) {
    // feature layer should in correct interval;
    feature_layer_num = 20;
  }

  Datum datum;
  const static int kUndefinedLabel = 0;
  if (!ReadImageToDatum(filename.c_str(), kUndefinedLabel, FLAGS_row_col_num, FLAGS_row_col_num, &datum)) {
    LOG(ERROR) << "ReadImageToDatum Error"; 
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
  int datum_size_ = datum.channels() * datum.height() * datum.width();
  //printf("Datum size: %d\n", datum_size_);
  if (data.size()) {
    for (int j = 0; j < datum_size_; ++j) {
      input_data[j] = (static_cast<float>((uint8_t)data[j])) * scale;
    }
  } else {
    for (int j = 0; j < datum_size_; ++j) {
      input_data[j] = (datum.float_data(j)) * scale;
    }
  }
  
  //predict
  const vector<Blob<float>*>&  output_blobs = caffe_test_net_->ForwardPrefilled();
  const vector<Blob<float>*>& caffe6_out = caffe_test_net_->top_vecs_[feature_layer_num];
  float * caffe6_out_res = new float[caffe6_out[0]->count()]; 
  for (int k=0; k<caffe6_out[0]->count(); k++) {
    caffe6_out_res[k] = caffe6_out[0]->cpu_data()[k];
  }


  //another picture
  datum.Clear();
  if (!ReadImageToDatum(filename2.c_str(), kUndefinedLabel, FLAGS_row_col_num, FLAGS_row_col_num, &datum)) {
    LOG(ERROR) << "ReadImageToDatum Error"; 
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
  datum_size_ = datum.channels() * datum.height() * datum.width();
  if (data2.size()) {
    for (int j = 0; j < datum_size_; ++j) {
      input_data[j] = (static_cast<float>((uint8_t)data2[j])) * scale;
    }
  } else {
    for (int j = 0; j < datum_size_; ++j) {
      input_data[j] = (datum.float_data(j)) * scale;
    }
  }

  //predict
  const vector<Blob<float>*>&  output_blobs2 = caffe_test_net_->ForwardPrefilled();
  const vector<Blob<float>*>& caffe6_out_2 = caffe_test_net_->top_vecs_[feature_layer_num];

  float ab = 0, a2 = 0, b2 = 0;
  for (int k=0; k<caffe6_out_2[0]->count(); k++) {
    ab += (caffe6_out_2[0]->cpu_data()[k] * caffe6_out_res[k]);
    a2 += (caffe6_out_res[k] * caffe6_out_res[k]);
    b2 += (caffe6_out_2[0]->cpu_data()[k] * caffe6_out_2[0]->cpu_data()[k]);
  }
  float similarity = 0;
  if (a2 < 0.00001 || b2 < 0.00001) {
    similarity = 0;
  } else {
    similarity = ab/sqrt(a2)/sqrt(b2);
  }
  response_message_.set_similarity(similarity);

  if (caffe6_out_res) {
    delete []caffe6_out_res;
    caffe6_out_res = NULL;
  }
  return 0;
}

int ClassifyImpl::CalcSimilairtyInSearch(int image_id, float & similarity) {
  const SimWeightVector & other_weight_vec = sim_weight_index_[image_id];
  if (other_weight_vec.size() != similarity_weight_vec_.size()) {
    similarity = 0;
    return -1;
  }
  float aa = 0, ab = 0, bb = 0;
  for (size_t i=0; i<similarity_weight_vec_.size(); i++) {
    aa += similarity_weight_vec_[i] * similarity_weight_vec_[i];
    bb += other_weight_vec[i] * other_weight_vec[i];
    ab += similarity_weight_vec_[i] * other_weight_vec[i];
  }
  if (aa < 0.00001 || bb < 0.00001) {
    similarity = 0;
    return -1;
  } else {
    similarity = ab/sqrt(aa)/sqrt(bb);
    if (similarity > 1.0)
      similarity = 1.0;
  }
  
  return 0;
}

bool SortSearchResultVectorFunction(const std::pair<int, float> & x, const std::pair<int, float> & y) {
  return x.second > y.second;
}
int ClassifyImpl::SortSearchResultMap(const SearchResultContainer & search_map, SearchResultSortVector & search_vec) {
  for (SearchResultContainer::const_iterator it = search_map.begin(); it != search_map.end(); it++) {
    search_vec.push_back( std::make_pair(it->first, it->second) );
  }
  std::sort(search_vec.begin(), search_vec.end(), SortSearchResultVectorFunction);
  return 0;
}
int ClassifyImpl::ImageSearch(const std::string & filename, int top_n_res) {
  if (ImageClassify(filename, top_n_res, 1)!=0) {
    return -1;
  }
  //according class id, search similarity images in the index.
  search_res_container_.clear();
  ImageCategoryIndex::iterator it;
  for (int i =0; i<response_message_.rsp_res_size() && i<FLAGS_top_n_limit; i++) {
    int class_id = response_message_.rsp_res(i).category_id();
    it = image_class_index_.find(class_id);
    if (it != image_class_index_.end()) {
      PostingList &postinglist = it->second;
      for (size_t k=0; k<postinglist.size(); k++) {
        search_res_container_.insert(std::make_pair(postinglist[k], 0.0) );
      }  
    }
  }

  //calc similarity for candidates
  std::cout << "Search candidates num:" << search_res_container_.size() << std::endl;
  for (SearchResultContainer::iterator it1 = search_res_container_.begin();
    it1 != search_res_container_.end(); it1++) {
    float sim = 0;
    CalcSimilairtyInSearch(it1->first, sim);
    it1->second = sim;
  }

  //sort map
  search_res_sort_vec_.clear();
  SortSearchResultMap(search_res_container_, search_res_sort_vec_);

  //std::cout << "Search file:" << request_message_.file_name() << std::endl;
  int add_num = 0;
  for (int i=0; i<search_res_sort_vec_.size() && add_num<request_message_.top_n_result(); i++) {
    if (search_res_sort_vec_[i].second >= request_message_.min_sim_thres() 
      && search_res_sort_vec_[i].second <= request_message_.max_sim_thres()) {
      ::image::SearchResult* add_search_res =  response_message_.add_search_res();
      add_search_res->set_search_file_name( index_id_to_filename_[search_res_sort_vec_[i].first] );
      add_search_res->set_search_similarity( search_res_sort_vec_[i].second );
      add_num++;
      //std::cout << "Search result: " << index_id_to_filename_[search_res_sort_vec_[i].first]  
      //<< ", weight:" << search_res_sort_vec_[i].second << std::endl;
    } else {
      continue;
    }
  }

  return 0;
}