#include <cstring>
#include <cstdlib>
#include <google/gflags.h>
#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "caffe/caffe.hpp"

using namespace caffe;


//different to ReadImageToDatum, this is to read image and convert gray;
static bool ReadImageToGaryDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  cv::Mat cv_img, cv_gray_img;
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
    cv::cvtColor(cv_img, cv_gray_img, CV_BGR2GRAY);
  } else {
    cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(cv_img, cv_gray_img, CV_BGR2GRAY);
  }
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  if (height > 0 && width > 0) {

  }
  datum->set_channels(1); //only gray image
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  /*for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }*/
  for (int h = 0; h < cv_img.rows; ++h) {
    for (int w = 0; w < cv_img.cols; ++w) {
      datum_string->push_back(static_cast<char>(cv_img.at<uint8_t>(h, w)));
    }
  }    
  return true;
}


inline static bool ReadImageToGaryDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, datum);
}

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


DEFINE_string(file, "35590867467.jpg", "The file that contains images.");
DEFINE_int32(row_col_num, 227, "Row/Column number");
DEFINE_string(model_def, "examples/imagenet_deploy.prototxt", "The model definition file.");
DEFINE_string(pretrained_model, "examples/imagenet_model", "The pretrained model.");
DEFINE_string(synset, "data/imagenet_category_name.txt", "The imagenet synset file.");
DEFINE_bool(gpu, false, "use gpu for computation");
DEFINE_string(index_file, "index_image.dat", "image index file");
DEFINE_string(similarity_weight_layer, "20", "feature layer vector, format is 20,23,25");

int Classify(const char * image_file, Net<float> & caffe_test_net, std::vector<std::string> & name_vector, 
  std::vector<float> & similarity_weight_vec, std::vector< std::pair<int, float> > & class_vec) {
  class_vec.clear();
  similarity_weight_vec.clear();

  Datum datum;
  const static int kUndefinedLabel = 0;
  if (!ReadImageToDatum(image_file, kUndefinedLabel, FLAGS_row_col_num, FLAGS_row_col_num, &datum)) {
    LOG(ERROR) << "ReadImageToDatum Error"; 
    return -1;
  }
  vector<Blob<float>*>& input_blobs = caffe_test_net.input_blobs();
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
  const vector<Blob<float>*>&  output_blobs = caffe_test_net.ForwardPrefilled();

  //process feature layer vector
  std::vector<std::string> split_vector;
  SplitString(FLAGS_similarity_weight_layer, ",", &split_vector);
  for (size_t k=0; k<split_vector.size(); k++) {
    int layer_num = atoi(split_vector[k].c_str());
    const vector<Blob<float>*>& weight_vec = caffe_test_net.top_vecs_[layer_num];
    for (size_t i=0; i<weight_vec[0]->count(); i++) {
      similarity_weight_vec.push_back(weight_vec[0]->cpu_data()[i]);
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
      const static int kTopNumber = 8;
      for (int k = 0; k < num; ++k) {        
        //top kTopNumber
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
        //printf("%s\n", image_file);
        for (int d=kTopNumber-1; d>=0; d--) {
          //printf("Rank%d : %d, %s, %f\n", kTopNumber-d, max_id[d], name_vector[max_id[d]].c_str(), maxval[d]);
          class_vec.push_back( std::make_pair(max_id[d], maxval[d]) );
        }
      }
  }
  return 0;
}


int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  if (argc < 2) {
    LOG(ERROR) << "classify_main test_file_dir(!!!absolute dir!!!)";
    return -1;
  }

  
  //cudaSetDevice(0);
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
  Net<float> caffe_test_net(test_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(FLAGS_pretrained_model, &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

  std::ifstream ifs;
  ifs.open(FLAGS_synset.c_str(), std::ifstream::in);
  std::string synset_line;
  std::vector<std::string> name_vector;
  while (getline(ifs, synset_line)) {
    std::size_t found = synset_line.find_last_of("\t");
    std::string name = synset_line.substr(found+1);
    //printf("name:%s\n", name.c_str());
    name_vector.push_back(name);
  }
  //LOG(INFO) << "name_vector.size : " << name_vector.size();
  CHECK_EQ(name_vector.size(), 1000) << "name_vector.size != 1000";

  std::ofstream fout;
  std::string dir, filepath;
  DIR *dp;
  struct dirent *dirp;
  struct stat filestat;

  dir = argv[1];
  dp = opendir(dir.c_str());
  if (!dp) {
    std::cerr << "Opendir error:" << argv[1] << std::endl;
    return -1;
  }
  fout.open(FLAGS_index_file.c_str(), std::ofstream::out);
  std::vector< std::pair<int, float> > class_vec;
  std::vector<float> similarity_weight_vec;
  while ((dirp = readdir(dp))) {

    filepath = dir + "/" + dirp->d_name;
    
    if (stat(filepath.c_str(), &filestat)) continue;
    if (S_ISDIR(filestat.st_mode)) continue;
    //std::cout << filepath << std::endl;

    //now classify this image, and store class id to build index
    // top - 5
    try {
      Classify(filepath.c_str(), caffe_test_net, name_vector, similarity_weight_vec, class_vec);
    } catch (...) {
      std::cerr << "===============error================" << std::endl;
      continue;
    }
    fout << filepath << " " << class_vec.size() << " " << similarity_weight_vec.size();
    for (size_t i=0; i<class_vec.size(); i++) {
      fout << " " << class_vec[i].first << " " << class_vec[i].second;
    }
    for (size_t i=0; i<similarity_weight_vec.size(); i++) {
      fout << " " << similarity_weight_vec[i];
    }
    fout << std::endl;  
  }
  fout.close();
  return 0;
}
