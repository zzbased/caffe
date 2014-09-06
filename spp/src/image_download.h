#ifndef CLASSIFY_IMAGE_DOWNLOAD_H
#define CLASSIFY_IMAGE_DOWNLOAD_H
#include <curl/curl.h>
#include <string>

namespace image {

class ImageDownload {
 public:
  static bool Download(const std::string & url, std::string & download_file);
 private:
};

}
#endif