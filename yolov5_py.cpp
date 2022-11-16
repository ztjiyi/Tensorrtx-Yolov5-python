#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "common.hpp"
#include <vector>
#include <fstream>
#include <string.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "yololayer.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "calibrator.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !

namespace py = pybind11;

struct alignas(float) s_result {
    std::array<float, 4>bbox;// x0,y0,x1,y1
    //float conf;置信度
    float class_id;
};

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input) {

    if (input.ndim() != 3)
        throw std::runtime_error("3-channel image must be 3 dims ");

    py::buffer_info buf = input.request();

    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

    return mat;
}

class yolov5
{
public:
    bool initialization(std::string& engine_name, int input_h, int input_w, int class_num);
    std::vector<s_result>& get_result(py::array_t<unsigned char>& img_numpy);
    bool mo_sta;
private:
    int INPUT_H;
    int INPUT_W;
    int CLASS_NUM;
    int OUTPUT_SIZE;
    char* INPUT_BLOB_NAME;
    char* OUTPUT_BLOB_NAME;
    void* buffers[2];
    float* data;
    float* prob;
    Logger gLogger;
    IExecutionContext* context;
    cudaStream_t stream;
private:
    void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize);
    std::array<float, 4> get_roi(cv::Mat& img, float bbox[4]);
    float iou(float lbox[4], float rbox[4]);
    void nms(std::vector<Yolo::Detection>& res, float* output, float conf_thresh, float nms_thresh = 0.5);
};

bool yolov5::initialization(std::string& engine_name, int input_h, int input_w, int class_num)
{
    //参数设置
    INPUT_H = input_h;
    INPUT_W = input_w;
    CLASS_NUM = class_num;
    OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
    INPUT_BLOB_NAME = "data";
    OUTPUT_BLOB_NAME = "prob";
    data = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    prob = new float[BATCH_SIZE * OUTPUT_SIZE];
    //读取模型
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return false;
    }
    char* trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    //载入模型
    IRuntime* runtime = createInferRuntime(gLogger); //创建 tensorrt对象
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);//反序列化读取引擎
    assert(engine != nullptr);
    context = engine->createExecutionContext();//执行上下文
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME); //检索命名张量绑定的索引
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);//检索命名张量绑定的索引
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float))); //cuda分配输入 输出内存
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    return true;
}
std::vector<s_result>& yolov5::get_result(py::array_t<unsigned char>& img_numpy)
{
    auto start = std::chrono::system_clock::now();
    cv::Mat img = numpy_uint8_3c_to_cv_mat(img_numpy);
    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
    int i = 0;
    int b = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
            data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
    
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    std::vector<Yolo::Detection> result;
    nms(result, &prob[0 * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
    s_result res_i = {};
    std::vector<s_result> res;
    for (size_t i = 0; i < result.size(); i++)
    {
        res_i.bbox=get_roi( img, result[i].bbox);
        res_i.class_id = result[i].class_id;

        res.push_back(res_i);
    }
    
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    return res;
}

void yolov5::doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}
std::array<float, 4> yolov5::get_roi( cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else {
        l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    std::array<float, 4 >roi_res = {l, t, r, b};
    return roi_res;
}
float yolov5::iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}
void yolov5::nms(std::vector<Yolo::Detection>& res, float* output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}


PYBIND11_MODULE(yolov5_py, m) {
    py::class_<s_result>(m,"s_result")
        .def(py::init<>())
        .def_readwrite("bbox", &s_result::bbox)
        .def_readwrite("class_id", &s_result::class_id);
    py::class_<yolov5>(m, "yolov5")
        .def(py::init<>())
        .def("initialization", &yolov5::initialization)
        .def_readwrite("mo_sta", &yolov5::mo_sta)
        .def("get_result", &yolov5::get_result);
}


//https://blog.csdn.net/qq_52859223/article/details/124362671
//https://blog.csdn.net/weixin_41552975/article/details/115340333?spm=1001.2014.3001.5501