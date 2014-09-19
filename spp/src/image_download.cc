#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <sstream>
#include <city.h>
#include <google/gflags.h>
#include "image_download.h"
#include "glog/logging.h"

DEFINE_string(download_dir, "/data/vincentyao/appdemo/app/uploads/", "image下载目录,在部署白板的时候,必须和白板的uploads目录一致");
DEFINE_string(proxy_ip_port, "10.130.24.42:80", "proxy to download");
DEFINE_string(proxy_user_pwd, "", "user and pwd of proxy");

namespace image {

template <typename T> std::string to_string(const T& n) {
  std::ostringstream stm ;
  stm << n ;
  return stm.str() ;
}

std::string FileNameEncode(const std::string & filename) {
  std::string indeed_filename = FLAGS_download_dir + "/"
      + to_string(CityHash64(filename.c_str(), filename.size())) + ".jpg";
  return indeed_filename;
}

inline bool IsFileExist(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

size_t ProcessData(void *buffer, size_t size, size_t nmemb, void *user_p) {
  FILE *fp = (FILE *)user_p;
  size_t return_size = fwrite(buffer, size, nmemb, fp);
  return return_size;
}

bool ImageDownload::Download(const std::string & url,
                             std::string & download_file) {
  download_file = FileNameEncode(url);
  if (IsFileExist(download_file)) {
    //已下载
    return true;
  }

  CURLcode return_code = curl_global_init(CURL_GLOBAL_ALL);
  if (return_code != CURLE_OK) {
    LOG(ERROR) << "init libcurl failed.";
    return false;
  }
  // 获取easy handle
  CURL *easy_handle = curl_easy_init();
  if (!easy_handle) {
    LOG(ERROR) << "get a easy handle failed.";
    curl_global_cleanup();
    return false;
  }

  FILE *fp = fopen(download_file.c_str(), "wb");
  if (!fp) {
    LOG(ERROR) << "file open failed.";
    curl_global_cleanup();
    return false;
  }
  // 设置easy handle属性
  curl_easy_setopt(easy_handle, CURLOPT_URL, url.c_str());
  curl_easy_setopt(easy_handle, CURLOPT_WRITEFUNCTION, &ProcessData);
  curl_easy_setopt(easy_handle, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt(easy_handle, CURLOPT_TIMEOUT_MS, 3000);  // Set timeout by ms
  // 设置下载代理
  if (!FLAGS_proxy_ip_port.empty()) {
    curl_easy_setopt(easy_handle, CURLOPT_PROXY, FLAGS_proxy_ip_port.c_str());
  }
  if (!FLAGS_proxy_user_pwd.empty()) {
    curl_easy_setopt(easy_handle, CURLOPT_PROXYUSERPWD, FLAGS_proxy_user_pwd.c_str());
  }

  // 执行数据请求
  return_code = curl_easy_perform(easy_handle);
  // Check for errors
  if (return_code != CURLE_OK) {
    LOG(ERROR) << "curl_easy_perform() failed: %s" << curl_easy_strerror(return_code);
  }
  // 释放资源
  fclose(fp);
  curl_easy_cleanup(easy_handle);
  curl_global_cleanup();
  return true;
}

}  // namespace image
