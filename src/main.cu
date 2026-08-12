#include "cuda_processor.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
struct Image{int w,h;std::vector<unsigned char> p;};
std::string Arg(int n,char**v,const std::string&k,const std::string&d){for(int i=1;i+1<n;i++)if(v[i]==k)return v[i+1];return d;}
Image Read(const fs::path& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot open input");std::string m;int max;Image x{};f>>m>>x.w>>x.h>>max;f.get();if(m!="P6"||max!=255)throw std::runtime_error("invalid PPM");x.p.resize((size_t)x.w*x.h*3);f.read((char*)x.p.data(),x.p.size());return x;}
void Write(const fs::path&p,const Image&x){std::ofstream f(p,std::ios::binary);f<<"P6\n"<<x.w<<" "<<x.h<<"\n255\n";f.write((char*)x.p.data(),x.p.size());}
int main(int n,char**v){try{
 fs::path in=Arg(n,v,"--input","data/input"),out=Arg(n,v,"--output","data/output");int lim=std::stoi(Arg(n,v,"--limit","1000")),threads=std::stoi(Arg(n,v,"--threads","256"));
 int dc=0;cudaGetDeviceCount(&dc);if(!dc)throw std::runtime_error("No CUDA GPU found");cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,0);
 std::vector<fs::path> files;for(auto&e:fs::directory_iterator(in))if(e.path().extension()==".ppm")files.push_back(e.path());std::sort(files.begin(),files.end());if(files.empty())throw std::runtime_error("No PPM files");if((int)files.size()>lim)files.resize(lim);fs::create_directories(out);
 auto st=std::chrono::steady_clock::now();float kt=0;size_t pixels=0;
 for(auto&p:files){Image a=Read(p),b{a.w,a.h,std::vector<unsigned char>(a.p.size())};float ms=0;ProcessImage(a.p.data(),b.p.data(),a.w,a.h,threads,&ms);Write(out/p.filename(),b);kt+=ms;pixels+=(size_t)a.w*a.h;}
 double total=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-st).count();
 std::cout<<std::fixed<<std::setprecision(3)<<"CUDA GPU: "<<prop.name<<"\nImages processed: "<<files.size()<<"\nTotal pixels: "<<pixels<<"\nCUDA kernel time (ms): "<<kt<<"\nTotal runtime (ms): "<<total<<"\nThroughput (images/sec): "<<files.size()/(total/1000.0)<<"\n";
 return 0;}catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 1;}}
