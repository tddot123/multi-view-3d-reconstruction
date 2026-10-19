// V5: synchronized views, float relative depth and order-independent soft carving.
// The existing axis-aligned / orthographic camera convention is retained.
// This is not calibrated metric reconstruction. See docs/usage.md for controls.
#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <unordered_map>
#include <queue>
#include <vector>

namespace fs = std::filesystem;
using cv::Mat;
using cv::Vec3b;
using cv::Vec3f;

struct Config
{
    // INPUT SELECTION: edit these five filenames to select another object.
    // Example: "f03.png", "b03.png", "l03.png", "r03.png", "t03.png".
    // Relative filenames are resolved under inputDir; absolute paths also work.
    // Front/left/top are required. Back/right may be "" to use mirrored priors.
    fs::path frontFile = "f04.png";
    fs::path backFile = "b04.png";
    fs::path leftFile = "l04.png";
    fs::path rightFile = "r04.png";
    fs::path topFile = "t04.png";

    int maxGrid = 256;
    int inputLongSide = 518; // Multiple-of-14, aspect-preserving inference.
    int threads = 2;
    int projectionSamples = 3;
    float depthLowQuantile = 0.02f;
    float depthHighQuantile = 0.98f;
    float depthScale = 1.f;
    float depthOffset = 0.f;
    std::array<float, 5> viewScale{{1.f, 1.f, 1.f, 1.f, 1.f}};
    std::array<float, 5> viewOffset{{0.f, 0.f, 0.f, 0.f, 0.f}};
    float rejectStrength = 2.5f;
    float priorWeight = 0.15f;
    float depthRelief = 1.5f; // Maximum surface refinement, in voxels, inside the silhouette hull.
    bool anchorDepth = true;
    bool useCache = true;
    int smoothIterations = 3;
    bool continuousSurface = true;
    float surfaceMargin = 0.35f;
    float frontPhotoPitch = 0.f;
    float backPhotoPitch = 0.f;
    bool headless = false;
    bool silhouetteOnly = false;
    bool selfTest = false;
    bool snapshot = false;
    bool snapshotViews = false;
    fs::path inputDir = "data/samples/multiview";
    fs::path outputDir = "outputs/v5";
    fs::path model = "models/depth_anything_v2_large.onnx";
    fs::path previewPly;
    fs::path cacheDir = ".cache/v5-depth-cache";
    float snapshotYaw = 55.f;
    float snapshotPitch = 25.f;
};

enum class ViewId
{
    Front,
    Back,
    Left,
    Right,
    Top
};
static const char *viewName(ViewId id)
{
    return std::array<const char *, 5>{"front", "back", "left", "right", "top"}[(int)id];
}
static bool positiveSide(ViewId id)
{
    return id == ViewId::Front || id == ViewId::Left || id == ViewId::Top;
}
static int depthAxis(ViewId id)
{
    return id == ViewId::Left || id == ViewId::Right ? 0 : id == ViewId::Top ? 1 : 2;
}
static float clamp01(float v)
{
    return std::clamp(v, 0.f, 1.f);
}
static float quantile(std::vector<float> values, float q)
{
    if (values.empty())
        throw std::runtime_error("Cannot measure an empty depth range");
    const float position = clamp01(q) * (values.size() - 1);
    const size_t k = (size_t)position;
    std::nth_element(values.begin(), values.begin() + k, values.end());
    const float low = values[k];
    if (k + 1 == values.size() || position == (float)k)
        return low;
    std::nth_element(values.begin(), values.begin() + k + 1, values.end());
    return low + (values[k + 1] - low) * (position - k);
}

// One bundle owns all three aligned arrays. Geometry never transforms depth alone.
struct View
{
    ViewId id = ViewId::Front;
    bool real = true;
    bool authoritativeMask = false;
    Mat rgb, mask, depth, edgeDistance;
    cv::Rect crop;
    float rawLow = 0.f, rawHigh = 0.f;
    float depthReliability = 0.f;
};

static void verifyBundle(const View &v)
{
    if (v.rgb.empty() || v.mask.empty() || v.depth.empty() || v.rgb.size() != v.mask.size() ||
        v.rgb.size() != v.depth.size() || v.rgb.type() != CV_8UC3 || v.mask.type() != CV_8UC1 ||
        v.depth.type() != CV_32FC1)
        throw std::runtime_error(std::string("Misaligned RGB/mask/depth bundle: ") + viewName(v.id));
}
static void flipBundle(View &v)
{
    cv::flip(v.rgb, v.rgb, 1);
    cv::flip(v.mask, v.mask, 1);
    cv::flip(v.depth, v.depth, 1);
}
static void rotateBundle(View &v)
{
    cv::rotate(v.rgb, v.rgb, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(v.mask, v.mask, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(v.depth, v.depth, cv::ROTATE_90_COUNTERCLOCKWISE);
}
static void resizeBundle(View &v, cv::Size size)
{
    if (v.rgb.size() != size)
    {
        cv::resize(v.rgb, v.rgb, size, 0, 0, cv::INTER_LINEAR);
        cv::resize(v.mask, v.mask, size, 0, 0, cv::INTER_NEAREST);
        cv::resize(v.depth, v.depth, size, 0, 0, cv::INTER_LINEAR);
    }
    verifyBundle(v);
}

static Mat autoMask(const Mat &bgr)
{
    Mat hsv, white, mask;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    // Find the colored/dark core, then retain its exterior silhouette. Gray glass,
    // chrome and white paint inside that outline are materials, not empty space.
    // A supplied mask/alpha remains authoritative and keeps genuine through-holes.
    cv::inRange(hsv, cv::Scalar(0, 0, 160), cv::Scalar(180, 80, 255), white);
    cv::bitwise_not(white, mask);
    // Small closing only. Do not fill external contours or open away thin parts.
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
        throw std::runtime_error("Auto mask found no foreground; supply a *_mask.png");
    auto main = std::max_element(contours.begin(), contours.end(), [](const auto &a, const auto &b) {
        return cv::contourArea(a) < cv::contourArea(b);
    });
    mask = Mat::zeros(bgr.size(), CV_8U);
    cv::drawContours(mask, contours, (int)std::distance(contours.begin(), main), cv::Scalar(255), cv::FILLED);
    return mask;
}

class DepthModel
{
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "V5Depth"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::string inputName, outputName;
    int fixedH = 0, fixedW = 0;
    int longSide;

  public:
    explicit DepthModel(const Config &cfg) : longSide(cfg.inputLongSide)
    {
        options.SetIntraOpNumThreads(cfg.threads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(env, cfg.model.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        inputName = session->GetInputNameAllocated(0, allocator).get();
        outputName = session->GetOutputNameAllocated(0, allocator).get();
        // Keep the TypeInfo owner alive while inspecting its tensor shape.
        auto info = session->GetInputTypeInfo(0);
        auto tensorInfo = info.GetTensorTypeAndShapeInfo();
        const auto shape = tensorInfo.GetShape();
        if (shape.size() != 4 || (shape[1] > 0 && shape[1] != 3) ||
            tensorInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Expected one float NCHW RGB model input");
        if ((shape[2] > 0) != (shape[3] > 0))
            throw std::runtime_error("Partially fixed model dimensions are unsupported");
        if (shape[2] > 0)
        {
            fixedH = (int)shape[2];
            fixedW = (int)shape[3];
        }
        std::cout << "ONNX: " << inputName << " -> " << outputName << ", "
                  << (fixedH ? "fixed input / letterbox" : "dynamic input / aspect ratio")
                  << ", external ImageNet normalization" << std::endl;
    }
    Mat infer(const Mat &bgr)
    {
        Mat resized;
        cv::Rect valid;
        if (fixedH)
        {
            const double s = std::min((double)fixedW / bgr.cols, (double)fixedH / bgr.rows);
            const cv::Size size(std::max(1, (int)std::round(bgr.cols * s)),
                                std::max(1, (int)std::round(bgr.rows * s)));
            Mat tmp;
            cv::resize(bgr, tmp, size, 0, 0, cv::INTER_CUBIC);
            resized = Mat(fixedH, fixedW, CV_8UC3, cv::Scalar(124, 116, 104));
            valid = cv::Rect((fixedW - size.width) / 2, (fixedH - size.height) / 2, size.width, size.height);
            tmp.copyTo(resized(valid));
        }
        else
        {
            const double s = (double)longSide / std::max(bgr.cols, bgr.rows);
            auto dim = [s](int d) { return std::max(14, (int)std::round(d * s / 14) * 14); };
            cv::resize(bgr, resized, cv::Size(dim(bgr.cols), dim(bgr.rows)), 0, 0, cv::INTER_CUBIC);
            valid = cv::Rect(0, 0, resized.cols, resized.rows);
        }
        const int h = resized.rows, w = resized.cols;
        const float mean[] = {0.485f, 0.456f, 0.406f};
        const float stddev[] = {0.229f, 0.224f, 0.225f};
        std::vector<float> data((size_t)3 * h * w);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                    data[((size_t)c * h + y) * w + x] =
                        (resized.at<Vec3b>(y, x)[2 - c] / 255.f - mean[c]) / stddev[c];
        std::array<int64_t, 4> shape{1, 3, h, w};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), 4);
        const char *inputs[] = {inputName.c_str()};
        const char *outputs[] = {outputName.c_str()};
        auto result = session->Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
        auto info = result[0].GetTensorTypeAndShapeInfo();
        auto dims = info.GetShape();
        if (dims.size() < 2 || info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Expected float depth output");
        const int oh = (int)dims[dims.size() - 2], ow = (int)dims.back();
        if (oh <= 0 || ow <= 0 || info.GetElementCount() != (size_t)oh * ow)
            throw std::runtime_error("Expected one single-channel depth map");
        Mat raw(oh, ow, CV_32F, result[0].GetTensorMutableData<float>());
        Mat own = raw.clone();
        if (own.size() != resized.size())
            cv::resize(own, own, resized.size());
        own = own(valid).clone();
        cv::resize(own, own, bgr.size(), 0, 0, cv::INTER_LINEAR);
        return own;
    }
};

static std::string depthCacheKey(const fs::path &input, const Config &cfg)
{
    auto stamp = [](const fs::path &p) {
        return fs::absolute(p).lexically_normal().string() + ":" + std::to_string(fs::file_size(p)) + ":" +
               std::to_string(fs::last_write_time(p).time_since_epoch().count());
    };
    return "aspect-imagenet-v1:" + std::to_string(cfg.inputLongSide) + ":" + stamp(input) + ":" +
           stamp(cfg.model);
}
static View loadView(ViewId id, const Config &cfg, std::unique_ptr<DepthModel> &model)
{
    View v;
    v.id = id;
    const std::array<fs::path, 5> filenames{cfg.frontFile, cfg.backFile, cfg.leftFile, cfg.rightFile,
                                            cfg.topFile};
    const fs::path filename = filenames[(int)id];
    const bool required = id == ViewId::Front || id == ViewId::Left || id == ViewId::Top;
    if (filename.empty())
    {
        if (required)
            throw std::runtime_error(std::string("Required input filename is empty: ") + viewName(id));
        return v;
    }
    const fs::path path = cfg.inputDir / filename;
    if (!fs::exists(path))
    {
        if (required)
            throw std::runtime_error(std::string("Missing required ") + viewName(id) +
                                     " image: " + path.string());
        std::cout << "Optional " << viewName(id) << " image not found: " << path.string() << std::endl;
        return v;
    }
    std::cout << "Input " << viewName(id) << ": " << path.string() << std::endl;
    Mat src = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (src.empty())
        throw std::runtime_error("Cannot read " + path.string());
    Mat raw;
    if (src.channels() == 4)
        cv::cvtColor(src, raw, cv::COLOR_BGRA2BGR);
    else if (src.channels() == 1)
        cv::cvtColor(src, raw, cv::COLOR_GRAY2BGR);
    else
        raw = src;
    if (raw.type() != CV_8UC3)
        throw std::runtime_error("Input must be an 8-bit image: " + path.string());
    Mat mask;
    const fs::path maskPath = path.parent_path() / (path.stem().string() + "_mask.png");
    if (fs::exists(maskPath))
    {
        v.authoritativeMask = true;
        mask = cv::imread(maskPath.string(), cv::IMREAD_GRAYSCALE);
        if (mask.empty() || mask.size() != raw.size())
            throw std::runtime_error("External mask must match its original RGB image: " + maskPath.string());
        cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
        std::cout << "  external mask: " << maskPath.string() << std::endl;
    }
    else if (src.channels() == 4)
    {
        cv::extractChannel(src, mask, 3);
        if (cv::countNonZero(mask < 255))
        {
            v.authoritativeMask = true;
            cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
        }
        else
            mask = autoMask(raw);
    }
    else
        mask = autoMask(raw);
    std::vector<cv::Point> fg;
    cv::findNonZero(mask, fg);
    if (fg.empty())
        throw std::runtime_error("Empty foreground: " + path.string());
    v.crop = cv::boundingRect(fg);
    std::cout << "Infer " << viewName(id) << " from original " << raw.cols << 'x' << raw.rows << std::endl;
    Mat fullDepth;
    if (cfg.silhouetteOnly)
        fullDepth = Mat(raw.size(), CV_32F, cv::Scalar(0.5f));
    else
    {
        const auto cachePath = cfg.cacheDir / (std::string(viewName(id)) + ".yml.gz");
        const std::string key = depthCacheKey(path, cfg);
        if (cfg.useCache && fs::exists(cachePath))
        {
            try
            {
                cv::FileStorage cache(cachePath.string(), cv::FileStorage::READ);
                std::string oldKey;
                cache["key"] >> oldKey;
                if (oldKey == key)
                    cache["raw_depth"] >> fullDepth;
            }
            catch (const cv::Exception &)
            {
                fullDepth.release();
            }
            if (fullDepth.size() != raw.size() || fullDepth.type() != CV_32F)
                fullDepth.release();
            if (!fullDepth.empty())
                std::cout << "  reused verified raw depth cache" << std::endl;
        }
        if (fullDepth.empty())
        {
            if (!model)
                model = std::make_unique<DepthModel>(cfg);
            fullDepth = model->infer(raw);
            if (cfg.useCache)
            {
                fs::create_directories(cfg.cacheDir);
                cv::FileStorage cache(cachePath.string(), cv::FileStorage::WRITE | cv::FileStorage::BASE64);
                if (cache.isOpened())
                    cache << "key" << key << "raw_depth" << fullDepth;
            }
        }
    }
    v.rgb = raw(v.crop).clone();
    v.mask = mask(v.crop).clone();
    v.depth = fullDepth(v.crop).clone();
    verifyBundle(v);
    return v;
}

static std::vector<View> alignViews(std::array<View, 5> &loaded)
{
    View &front = loaded[(int)ViewId::Front];
    View &left = loaded[(int)ViewId::Left];
    View &top = loaded[(int)ViewId::Top];
    if (front.rgb.empty() || left.rgb.empty() || top.rgb.empty())
        throw std::runtime_error(
            "Need valid front, left and top images; check Config filenames and inputDir");
    const int w = front.rgb.cols, h = front.rgb.rows;
    const int d = std::max(1, (int)std::round((double)left.rgb.cols * h / left.rgb.rows));
    for (auto &v : loaded)
    {
        if (v.rgb.empty())
            continue;
        if (v.id == ViewId::Left || v.id == ViewId::Back)
            flipBundle(v);
        if (v.id == ViewId::Top)
            rotateBundle(v);
        resizeBundle(v, v.id == ViewId::Top                             ? cv::Size(w, d)
                        : v.id == ViewId::Left || v.id == ViewId::Right ? cv::Size(d, h)
                                                                        : cv::Size(w, h));
    }
    // Mirror in camera space then undo the opposite-view horizontal orientation:
    // in the unified world projection the aligned arrays are an exact clone.
    for (auto pair : {std::pair<ViewId, ViewId>{ViewId::Back, ViewId::Front}, {ViewId::Right, ViewId::Left}})
    {
        View &dst = loaded[(int)pair.first];
        if (!dst.rgb.empty())
            continue;
        const View &src = loaded[(int)pair.second];
        dst = src;
        dst.id = pair.first;
        dst.real = false;
        dst.rgb = src.rgb.clone();
        dst.mask = src.mask.clone();
        dst.depth = src.depth.clone();
        std::cout << "Missing " << viewName(dst.id) << ": weak mirrored prior, not an independent vote"
                  << std::endl;
    }
    return std::vector<View>(loaded.begin(), loaded.end());
}

static void normalizeDepth(View &v, const Config &cfg)
{
    cv::distanceTransform(v.mask, v.edgeDistance, cv::DIST_L2, 3);
    std::vector<float> interior, all;
    // Quantiles are measured on real foreground, avoiding a large temporary list.
    const int stride = std::max(1, (int)std::sqrt((double)v.depth.total() / 200000));
    for (int y = 0; y < v.depth.rows; y += stride)
        for (int x = 0; x < v.depth.cols; x += stride)
        {
            const float d = v.depth.at<float>(y, x);
            if (!v.mask.at<uchar>(y, x) || !std::isfinite(d))
                continue;
            all.push_back(d);
            if (v.edgeDistance.at<float>(y, x) >= 3.f)
                interior.push_back(d);
        }
    const auto &samples = interior.size() >= 32 ? interior : all;
    v.depthReliability = 0.f;
    if (!samples.empty())
    {
        v.rawLow = quantile(samples, cfg.depthLowQuantile);
        v.rawHigh = quantile(samples, cfg.depthHighQuantile);
        if (!cfg.silhouetteOnly &&
            v.rawHigh - v.rawLow > 1e-5f * std::max({1.f, std::abs(v.rawLow), std::abs(v.rawHigh)}))
            v.depthReliability = 1.f;
    }
    const float span = std::max(1e-6f, v.rawHigh - v.rawLow);
    for (int y = 0; y < v.depth.rows; ++y)
        for (int x = 0; x < v.depth.cols; ++x)
        {
            float &d = v.depth.at<float>(y, x);
            if (!v.mask.at<uchar>(y, x) || !std::isfinite(d))
            {
                d = std::numeric_limits<float>::quiet_NaN();
            }
            else
            {
                d = v.depthReliability > 0 ? clamp01((d - v.rawLow) / span) : 0.5f;
                d = clamp01((d * cfg.depthScale + cfg.depthOffset) * cfg.viewScale[(int)v.id] +
                            cfg.viewOffset[(int)v.id]);
            }
        }
    std::cout << "  " << viewName(v.id) << " robust raw range [" << v.rawLow << ", " << v.rawHigh
              << "], depth reliability " << v.depthReliability << std::endl;
}

enum class CellState
{
    Unknown,
    Foreground,
    Background
};
struct Cell
{
    CellState state = CellState::Unknown;
    float depth = 0.5f;
    float confidence = 0.f;
    float gradient = 0.f;
    float coverage = 0.f;
    Vec3f color{0, 0, 0}; // RGB, not BGR.
    float anchoredBound = std::numeric_limits<float>::quiet_NaN();
};
struct Projection
{
    ViewId id = ViewId::Front;
    bool real = true;
    int aSize = 0, bSize = 0;
    std::vector<Cell> cells;
    const Cell &at(int a, int b) const
    {
        return cells[(size_t)a * bSize + b];
    }
};
struct Grid
{
    int x = 0, y = 0, z = 0;
    size_t count() const
    {
        return (size_t)x * y * z;
    }
    size_t index(int a, int b, int c) const
    {
        return ((size_t)a * y + b) * z + c;
    }
    bool contains(int a, int b, int c) const
    {
        return a >= 0 && b >= 0 && c >= 0 && a < x && b < y && c < z;
    }
    int axisSize(int axis) const
    {
        return axis == 0 ? x : axis == 1 ? y : z;
    }
};
static Grid makeGrid(const std::vector<View> &views, const Config &cfg)
{
    const auto &front = views[(int)ViewId::Front];
    const auto &left = views[(int)ViewId::Left];
    const double s =
        std::min(1., (double)cfg.maxGrid / std::max({front.rgb.cols, front.rgb.rows, left.rgb.cols}));
    return {std::max(1, (int)std::round(front.rgb.cols * s)),
            std::max(1, (int)std::round(front.rgb.rows * s)),
            std::max(1, (int)std::round(left.rgb.cols * s))};
}

struct Sample
{
    float d;
    Vec3f color;
};
static Cell resolveCell(const std::vector<Sample> &samples, int fgCount, int total, float reliability)
{
    Cell cell;
    if (!fgCount)
        return cell;
    cell.state = CellState::Foreground;
    cell.coverage = (float)fgCount / std::max(1, total);
    if (samples.empty())
        return cell; // Valid silhouette, unknown depth.
    std::vector<float> values;
    for (const auto &s : samples)
        if (std::isfinite(s.d))
            values.push_back(s.d);
    if (values.empty())
    {
        for (const auto &s : samples)
            cell.color += s.color;
        cell.color *= 1.f / samples.size();
        return cell;
    }
    cell.depth = quantile(values, 0.5f);
    const float spread = quantile(values, 0.9f) - quantile(values, 0.1f);
    float weight = 0.f;
    for (const auto &s : samples)
    {
        if (!std::isfinite(s.d))
            continue;
        float w = std::exp(-std::abs(s.d - cell.depth) / 0.025f);
        cell.color += s.color * w;
        weight += w;
    }
    cell.color *= 1.f / std::max(weight, 1e-6f);
    // Single samples remain valid: resolution increases must not erase the model.
    const float countConfidence = values.size() == 1 ? 0.65f : values.size() < 4 ? 0.8f : 1.f;
    cell.confidence = reliability * countConfidence * std::sqrt(cell.coverage) *
                      clamp01(1.f - spread / 0.25f) * ((float)values.size() / samples.size());
    return cell;
}

static Projection buildProjection(const View &v, const Grid &grid, const Config &cfg)
{
    Projection p;
    p.id = v.id;
    p.real = v.real;
    p.aSize = v.id == ViewId::Left || v.id == ViewId::Right ? grid.z : grid.x;
    p.bSize = v.id == ViewId::Top ? grid.z : grid.y;
    const size_t n = (size_t)p.aSize * p.bSize;
    p.cells.resize(n);
    std::vector<std::vector<Sample>> samples(n);
    std::vector<int> fg(n, 0), total(n, 0);
    // Bounded oversampling. Masks and depths use nearest-neighbor sampling;
    // invalid/background depth never blends into foreground cells.
    const int w = std::min(v.rgb.cols, p.aSize * cfg.projectionSamples);
    const int h = std::min(v.rgb.rows, p.bSize * cfg.projectionSamples);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            const int px = std::min(v.rgb.cols - 1, (int)((x + 0.5) * v.rgb.cols / w));
            const int py = std::min(v.rgb.rows - 1, (int)((y + 0.5) * v.rgb.rows / h));
            const int a = x * p.aSize / w;
            const int b = v.id == ViewId::Top ? y * p.bSize / h : (h - y - 1) * p.bSize / h;
            const size_t i = (size_t)a * p.bSize + b;
            ++total[i];
            if (!v.mask.at<uchar>(py, px))
                continue;
            ++fg[i];
            const float d = v.depth.at<float>(py, px);
            const Vec3b c = v.rgb.at<Vec3b>(py, px);
            samples[i].push_back({d, Vec3f(c[2] / 255.f, c[1] / 255.f, c[0] / 255.f)});
        }
    Mat foreground(p.aSize, p.bSize, CV_8U, cv::Scalar(0));
    for (int a = 0; a < p.aSize; ++a)
        for (int b = 0; b < p.bSize; ++b)
        {
            size_t i = (size_t)a * p.bSize + b;
            p.cells[i] = resolveCell(samples[i], fg[i], total[i], v.depthReliability);
            if (fg[i])
                foreground.at<uchar>(a, b) = 255;
        }
    Mat outside, distance;
    cv::bitwise_not(foreground, outside);
    cv::distanceTransform(outside, distance, cv::DIST_L2, 3);
    int reliable = 0, unknown = 0;
    for (int a = 0; a < p.aSize; ++a)
        for (int b = 0; b < p.bSize; ++b)
        {
            Cell &c = p.cells[(size_t)a * p.bSize + b];
            if (c.state != CellState::Foreground)
            {
                // One-cell silhouette boundary is uncertain, not negative evidence.
                c.state = total[(size_t)a * p.bSize + b] > 0 && distance.at<float>(a, b) > 1.4f
                              ? CellState::Background
                              : CellState::Unknown;
                continue;
            }
            float gradient = 0;
            for (auto d : {cv::Point(-1, 0), cv::Point(1, 0), cv::Point(0, -1), cv::Point(0, 1)})
            {
                int na = a + d.x, nb = b + d.y;
                if (na < 0 || nb < 0 || na >= p.aSize || nb >= p.bSize)
                    continue;
                const Cell &neighbor = p.at(na, nb);
                if (neighbor.state == CellState::Foreground && neighbor.confidence > 0)
                    gradient = std::max(gradient, std::abs(neighbor.depth - c.depth));
            }
            c.gradient = gradient;
            c.confidence *= 1.f - 0.6f * clamp01(gradient / 0.1f);
            c.confidence > 0.05f ? ++reliable : ++unknown;
        }
    std::cout << "Projection " << viewName(v.id) << ": " << reliable << " trusted, " << unknown
              << " uncertain foreground cells" << (v.real ? " (real)" : " (prior)") << std::endl;
    return p;
}

static const Cell &projectCell(const Projection &p, int x, int y, int z)
{
    if (p.id == ViewId::Left || p.id == ViewId::Right)
        return p.at(z, y);
    return p.id == ViewId::Top ? p.at(x, z) : p.at(x, y);
}
static float surfaceBound(const Projection &p, const Cell &c)
{
    if (std::isfinite(c.anchoredBound))
        return c.anchoredBound;
    return positiveSide(p.id) ? c.depth : 1.f - c.depth;
}
static float axisCoordinate(ViewId id, const Grid &g, int x, int y, int z)
{
    const int axis = depthAxis(id);
    const int k = axis == 0 ? x : axis == 1 ? y : z;
    return (k + 0.5f) / g.axisSize(axis);
}

struct Evidence
{
    float support = 0.f, reject = 0.f;
    uint8_t silhouetteVotes = 0, depthVotes = 0, sources = 0;
    bool hull = true;
};
static Evidence evaluate(const std::vector<Projection> &projections, const Grid &g, const Config &cfg, int x,
                         int y, int z)
{
    // Fixed view order for floating-point sums, even when caller order changes.
    std::array<const Projection *, 5> sorted{};
    for (const auto &p : projections)
        sorted[(int)p.id] = &p;
    Evidence e;
    for (const Projection *ptr : sorted)
    {
        if (!ptr)
            continue;
        const Projection &p = *ptr;
        const Cell &c = projectCell(p, x, y, z);
        if (c.state == CellState::Background)
        {
            if (p.real)
                e.hull = false;
            continue;
        }
        if (c.state == CellState::Unknown)
            continue;
        if (p.real)
            ++e.silhouetteVotes;
        if (c.confidence <= 0.05f || cfg.silhouetteOnly)
            continue;
        const float weight = c.confidence * (p.real ? 1.f : cfg.priorWeight);
        const float dim = (float)g.axisSize(depthAxis(p.id));
        const float edge = clamp01(c.gradient / 0.1f);
        const float tolerance = (cfg.anchorDepth ? 0.25f + 0.5f * edge + 0.5f * (1.f - c.coverage)
                                                 : 1.25f + 2.f * edge + 1.f - c.coverage) /
                                dim;
        const float t = axisCoordinate(p.id, g, x, y, z);
        const float freeSpace = positiveSide(p.id) ? t - surfaceBound(p, c) : surfaceBound(p, c) - t;
        if (freeSpace <= tolerance)
        {
            e.support += weight;
            if (p.real)
            {
                ++e.depthVotes;
                e.sources |= (uint8_t)(1 << (int)p.id);
            }
        }
        else
        {
            const float edgeWeight = 1.f - 0.65f * edge;
            e.reject += weight * edgeWeight * clamp01((freeSpace - tolerance) / std::max(tolerance, 1e-6f));
        }
    }
    e.hull = e.hull && e.silhouetteVotes >= 2;
    return e;
}

struct Volume
{
    Grid grid;
    std::vector<uint8_t> occupied, hull, votes, sources;
    std::vector<float> support, reject;
};
static int neighborCount(const Volume &v, const std::vector<uint8_t> &occ, int x, int y, int z)
{
    int n = 0;
    for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz)
                if ((dx || dy || dz) && v.grid.contains(x + dx, y + dy, z + dz))
                    n += occ[v.grid.index(x + dx, y + dy, z + dz)] != 0;
    return n;
}
static void anchorDepthToHull(std::vector<Projection> &projections, const Grid &g,
                              const std::vector<uint8_t> &hull, const Config &cfg)
{
    for (auto &p : projections)
    {
        const size_t cells = p.cells.size();
        const int dim = g.axisSize(depthAxis(p.id));
        std::vector<int> lo(cells, dim), hi(cells, -1);
        for (int x = 0; x < g.x; ++x)
            for (int y = 0; y < g.y; ++y)
                for (int z = 0; z < g.z; ++z)
                {
                    if (!hull[g.index(x, y, z)])
                        continue;
                    const size_t idx = p.id == ViewId::Left || p.id == ViewId::Right ? (size_t)z * p.bSize + y
                                       : p.id == ViewId::Top                         ? (size_t)x * p.bSize + z
                                                             : (size_t)x * p.bSize + y;
                    const int k = depthAxis(p.id) == 0 ? x : depthAxis(p.id) == 1 ? y : z;
                    lo[idx] = std::min(lo[idx], k);
                    hi[idx] = std::max(hi[idx], k);
                }
        // Low-frequency, mask-aware relative depth: it refines a known surface,
        // never invents an absolute camera-space position from a min/max value.
        Mat values(p.aSize, p.bSize, CV_32F, cv::Scalar(0));
        Mat weights(p.aSize, p.bSize, CV_32F, cv::Scalar(0));
        for (int a = 0; a < p.aSize; ++a)
            for (int b = 0; b < p.bSize; ++b)
            {
                const auto &c = p.at(a, b);
                if (c.state != CellState::Foreground)
                    continue;
                const float w = std::max(0.05f, c.confidence);
                weights.at<float>(a, b) = w;
                values.at<float>(a, b) = c.depth * w;
            }
        cv::GaussianBlur(values, values, cv::Size(0, 0), 1.2);
        cv::GaussianBlur(weights, weights, cv::Size(0, 0), 1.2);
        for (int a = 0; a < p.aSize; ++a)
            for (int b = 0; b < p.bSize; ++b)
            {
                size_t i = (size_t)a * p.bSize + b;
                auto &c = p.cells[i];
                if (hi[i] < lo[i])
                    continue;
                const float smoothed = weights.at<float>(a, b) > 1e-6f
                                           ? values.at<float>(a, b) / weights.at<float>(a, b)
                                           : c.depth;
                const float maximum = std::min(cfg.depthRelief, 0.10f * (hi[i] - lo[i] + 1));
                const float inset =
                    (1.f - smoothed) * maximum * c.confidence * (p.real ? 1.f : cfg.priorWeight);
                c.anchoredBound = positiveSide(p.id) ? (hi[i] + 1.f - inset) / dim : (lo[i] + inset) / dim;
            }
    }
}
static Volume reconstruct(std::vector<Projection> &projections, const Grid &g, const Config &cfg)
{
    Volume v;
    v.grid = g;
    const size_t n = g.count();
    v.occupied.resize(n);
    v.hull.resize(n);
    v.votes.resize(n);
    v.sources.resize(n);
    v.support.resize(n);
    v.reject.resize(n);
    Config hullCfg = cfg;
    hullCfg.silhouetteOnly = true;
    for (int x = 0; x < g.x; ++x)
        for (int y = 0; y < g.y; ++y)
            for (int z = 0; z < g.z; ++z)
                v.hull[g.index(x, y, z)] = evaluate(projections, g, hullCfg, x, y, z).hull;
    if (cfg.anchorDepth && !cfg.silhouetteOnly)
        anchorDepthToHull(projections, g, v.hull, cfg);
    for (int x = 0; x < g.x; ++x)
        for (int y = 0; y < g.y; ++y)
            for (int z = 0; z < g.z; ++z)
            {
                const size_t i = g.index(x, y, z);
                const Evidence e = evaluate(projections, g, cfg, x, y, z);
                v.hull[i] = e.hull;
                v.votes[i] = e.depthVotes;
                v.sources[i] = e.sources;
                v.support[i] = e.support;
                v.reject[i] = e.reject;
                // No trusted depth: retain the silhouette candidate, not an empty ray.
                v.occupied[i] = e.hull && (e.reject <= 1e-6f || e.support >= cfg.rejectStrength * e.reject);
            }
    std::vector<uint8_t> restored = v.occupied;
    for (int x = 0; x < g.x; ++x)
        for (int y = 0; y < g.y; ++y)
            for (int z = 0; z < g.z; ++z)
            {
                const size_t i = g.index(x, y, z);
                if (!v.occupied[i] && v.hull[i] && v.votes[i] >= 2 &&
                    v.support[i] >= cfg.rejectStrength * v.reject[i] * 0.85f &&
                    neighborCount(v, v.occupied, x, y, z) >= 10)
                    restored[i] = 1;
            }
    v.occupied.swap(restored);
    std::vector<uint8_t> pruned = v.occupied;
    for (int x = 0; x < g.x; ++x)
        for (int y = 0; y < g.y; ++y)
            for (int z = 0; z < g.z; ++z)
            {
                const size_t i = g.index(x, y, z);
                if (v.occupied[i] && v.votes[i] < 2 && neighborCount(v, v.occupied, x, y, z) == 0)
                    pruned[i] = 0;
            }
    v.occupied.swap(pruned);
    return v;
}

struct MeshVertex
{
    Vec3f position{0, 0, 0}, normal{0, 0, 0}, color{0.1f, 0.1f, 0.1f};
    std::array<float, 5> textureWeight{};
};
struct Mesh
{
    std::vector<MeshVertex> vertices;
    std::vector<std::array<uint32_t, 3>> triangles;
    size_t boundaryEdges = 0, nonmanifoldEdges = 0, inconsistentOrientedEdges = 0;
    int components = 0;
};
static std::array<cv::Vec4f, 5> photoBounds{};
static std::array<float, 5> photoPitch{{0.f, 0.f, 0.f, 0.f, 0.f}};
static bool fittedPhotoProjection = false;
static cv::Vec2f photoPlane(ViewId id, const Vec3f &p)
{
    if (id == ViewId::Top)
        return {p[0], -p[2]};
    float depth = depthAxis(id) == 0 ? p[0] : p[2];
    if (!positiveSide(id))
        depth = -depth;
    const float a = photoPitch[(int)id] * 0.01745329252f;
    return {depthAxis(id) == 0 ? p[2] : p[0], p[1] * std::cos(a) - depth * std::sin(a)};
}
static void fitPhotoProjection(const Mesh &mesh, const Config &cfg)
{
    photoPitch = {(float)cfg.frontPhotoPitch, (float)cfg.backPhotoPitch, 0.f, 0.f, 0.f};
    for (int id = 0; id < 5; ++id)
    {
        cv::Vec4f b{FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX};
        for (const auto &v : mesh.vertices)
        {
            auto p = photoPlane((ViewId)id, v.position);
            b[0] = std::min(b[0], p[0]);
            b[1] = std::max(b[1], p[0]);
            b[2] = std::min(b[2], p[1]);
            b[3] = std::max(b[3], p[1]);
        }
        photoBounds[id] = b;
    }
    fittedPhotoProjection = true;
}
static uint64_t edgeKey(uint32_t a, uint32_t b)
{
    if (a > b)
        std::swap(a, b);
    return ((uint64_t)a << 32) | b;
}
static cv::Vec2f textureUV(ViewId id, const Vec3f &p, const Grid &g)
{
    if (fittedPhotoProjection)
    {
        const auto q = photoPlane(id, p);
        const auto b = photoBounds[(int)id];
        return {(q[0] - b[0]) / std::max(1e-5f, b[1] - b[0]), (q[1] - b[2]) / std::max(1e-5f, b[3] - b[2])};
    }
    const float x = (p[0] + g.x * 0.5f) / g.x;
    const float y = (p[1] + g.y * 0.5f) / g.y;
    const float z = (p[2] + g.z * 0.5f) / g.z;
    return id == ViewId::Left || id == ViewId::Right ? cv::Vec2f(z, y)
           : id == ViewId::Top                       ? cv::Vec2f(x, 1.f - z)
                                                     : cv::Vec2f(x, y);
}
static Vec3f bilinearColor(const Mat &image, cv::Vec2f uv)
{
    float x = std::clamp(uv[0] * image.cols - 0.5f, 0.f, (float)image.cols - 1);
    float y = std::clamp((1.f - uv[1]) * image.rows - 0.5f, 0.f, (float)image.rows - 1);
    int ix = (int)x, iy = (int)y;
    int jx = std::min(ix + 1, image.cols - 1), jy = std::min(iy + 1, image.rows - 1);
    auto rgb = [&](int a, int b) {
        Vec3b c = image.at<Vec3b>(b, a);
        return Vec3f(c[2], c[1], c[0]) / 255.f;
    };
    return rgb(ix, iy) * (1.f - (x - ix)) * (1.f - (y - iy)) + rgb(jx, iy) * (x - ix) * (1.f - (y - iy)) +
           rgb(ix, jy) * (1.f - (x - ix)) * (y - iy) + rgb(jx, jy) * (x - ix) * (y - iy);
}
static void meshNormals(Mesh &mesh)
{
    for (auto &v : mesh.vertices)
        v.normal = Vec3f(0, 0, 0);
    for (const auto &tri : mesh.triangles)
    {
        const Vec3f n = (mesh.vertices[tri[1]].position - mesh.vertices[tri[0]].position)
                            .cross(mesh.vertices[tri[2]].position - mesh.vertices[tri[0]].position);
        for (uint32_t i : tri)
            mesh.vertices[i].normal += n;
    }
    for (auto &v : mesh.vertices)
    {
        float length = (float)cv::norm(v.normal);
        if (length > 1e-10f)
            v.normal *= 1.f / length;
    }
}
static void smoothMesh(Mesh &mesh, int iterations)
{
    std::vector<std::vector<uint32_t>> neighbors(mesh.vertices.size());
    for (const auto &tri : mesh.triangles)
        for (int i = 0; i < 3; ++i)
        {
            neighbors[tri[i]].push_back(tri[(i + 1) % 3]);
            neighbors[tri[i]].push_back(tri[(i + 2) % 3]);
        }
    for (auto &ns : neighbors)
    {
        std::sort(ns.begin(), ns.end());
        ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
    }
    // Taubin alternating steps suppress triangulation staircases without the
    // systematic contraction of repeated one-sided Laplacian smoothing.
    std::vector<Vec3f> next(mesh.vertices.size());
    for (int k = 0; k < iterations; ++k)
        for (float strength : {0.32f, -0.34f})
        {
            for (size_t i = 0; i < mesh.vertices.size(); ++i)
            {
                Vec3f average{0, 0, 0};
                for (uint32_t j : neighbors[i])
                    average += mesh.vertices[j].position;
                next[i] = mesh.vertices[i].position;
                if (!neighbors[i].empty())
                    next[i] += (average * (1.f / neighbors[i].size()) - next[i]) * strength;
            }
            for (size_t i = 0; i < mesh.vertices.size(); ++i)
                mesh.vertices[i].position = next[i];
        }
    meshNormals(mesh);
}
static void meshAudit(Mesh &mesh)
{
    struct EdgeAudit
    {
        int count = 0, direction = 0;
    };
    std::unordered_map<uint64_t, EdgeAudit> edges;
    std::vector<std::vector<uint32_t>> neighbors(mesh.vertices.size());
    for (const auto &tri : mesh.triangles)
        for (int j = 0; j < 3; ++j)
        {
            uint32_t a = tri[j], b = tri[(j + 1) % 3];
            auto &edge = edges[edgeKey(a, b)];
            ++edge.count;
            edge.direction += a < b ? 1 : -1;
            neighbors[a].push_back(b);
            neighbors[b].push_back(a);
        }
    mesh.boundaryEdges = mesh.nonmanifoldEdges = mesh.inconsistentOrientedEdges = 0;
    for (const auto &e : edges)
    {
        if (e.second.count == 1)
            ++mesh.boundaryEdges;
        if (e.second.count > 2)
            ++mesh.nonmanifoldEdges;
        if (e.second.count == 2 && e.second.direction != 0)
            ++mesh.inconsistentOrientedEdges;
    }
    std::vector<uint8_t> visited(mesh.vertices.size(), 0);
    mesh.components = 0;
    for (uint32_t i = 0; i < mesh.vertices.size(); ++i)
    {
        if (visited[i] || neighbors[i].empty())
            continue;
        ++mesh.components;
        std::queue<uint32_t> todo;
        todo.push(i);
        visited[i] = 1;
        while (!todo.empty())
        {
            uint32_t v = todo.front();
            todo.pop();
            for (uint32_t j : neighbors[v])
                if (!visited[j])
                {
                    visited[j] = 1;
                    todo.push(j);
                }
        }
    }
}
static float sampleScalar(const Mat &image, float x, float y)
{
    float cx = std::clamp(x, 0.f, (float)image.cols - 1), cy = std::clamp(y, 0.f, (float)image.rows - 1);
    int a = (int)cx, b = (int)cy, c = std::min(a + 1, image.cols - 1), d = std::min(b + 1, image.rows - 1);
    return image.at<float>(b, a) * (1 - (cx - a)) * (1 - (cy - b)) +
           image.at<float>(b, c) * (cx - a) * (1 - (cy - b)) +
           image.at<float>(d, a) * (1 - (cx - a)) * (cy - b) + image.at<float>(d, c) * (cx - a) * (cy - b);
}
static void continuousHullField(std::vector<float> &field, const Grid &nodes, const Grid &g, int pad,
                                const std::vector<View> &views, const std::vector<Projection> &projections,
                                const Config &cfg)
{
    std::array<Mat, 5> distances, depths, confidences;
    for (const auto &v : views)
    {
        const int id = (int)v.id;
        const int w = depthAxis(v.id) == 0 ? g.z : g.x, h = v.id == ViewId::Top ? g.z : g.y;
        if (v.real)
        {
            Mat padded, in, out;
            cv::copyMakeBorder(v.mask, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::distanceTransform(padded, in, cv::DIST_L2, 5);
            cv::distanceTransform(255 - padded, out, cv::DIST_L2, 5);
            Mat sdf = (in - out) * ((float)w / v.mask.cols + (float)h / v.mask.rows) * 0.5f;
            cv::resize(sdf(cv::Rect(1, 1, v.mask.cols, v.mask.rows)), distances[id], cv::Size(w, h), 0, 0,
                       cv::INTER_LINEAR);
        }
    }
    for (const auto &p : projections)
    {
        const int id = (int)p.id;
        Mat weighted(p.bSize, p.aSize, CV_32F, cv::Scalar(0)),
            weights(weighted.size(), CV_32F, cv::Scalar(0));
        for (int a = 0; a < p.aSize; ++a)
            for (int b = 0; b < p.bSize; ++b)
            {
                const auto &c = p.at(a, b);
                const int row = p.id == ViewId::Top ? b : p.bSize - 1 - b;
                const float w =
                    c.state == CellState::Foreground ? c.confidence * (p.real ? 1.f : cfg.priorWeight) : 0.f;
                weighted.at<float>(row, a) = c.depth * w;
                weights.at<float>(row, a) = w;
            }
        cv::GaussianBlur(weighted, weighted, cv::Size(0, 0), 2.0);
        cv::GaussianBlur(weights, weights, cv::Size(0, 0), 2.0);
        depths[id] = Mat(weighted.size(), CV_32F, cv::Scalar(0.5f));
        confidences[id] = weights;
        for (int y = 0; y < weights.rows; ++y)
            for (int x = 0; x < weights.cols; ++x)
                if (weights.at<float>(y, x) > 1e-6f)
                    depths[id].at<float>(y, x) = weighted.at<float>(y, x) / weights.at<float>(y, x);
    }
    auto coordinates = [&](int id, float x, float y, float z) {
        return id == (int)ViewId::Top       ? cv::Vec2f(x - 0.5f, z - 0.5f)
               : depthAxis((ViewId)id) == 0 ? cv::Vec2f(z - 0.5f, g.y - 0.5f - y)
                                            : cv::Vec2f(x - 0.5f, g.y - 0.5f - y);
    };
    for (int x = 0; x < nodes.x; ++x)
        for (int y = 0; y < nodes.y; ++y)
            for (int z = 0; z < nodes.z; ++z)
            {
                const float a = x - pad + 0.5f, b = y - pad + 0.5f, c = z - pad + 0.5f;
                bool first = true;
                float combined = 0;
                for (int id = 0; id < 5; ++id)
                {
                    if (distances[id].empty())
                        continue;
                    auto uv = coordinates(id, a, b, c);
                    const auto &sdf = distances[id];
                    const float dx = uv[0] - std::clamp(uv[0], 0.f, (float)sdf.cols - 1),
                                dy = uv[1] - std::clamp(uv[1], 0.f, (float)sdf.rows - 1);
                    const float value =
                        sampleScalar(sdf, uv[0], uv[1]) - std::sqrt(dx * dx + dy * dy) + cfg.surfaceMargin;
                    if (first)
                    {
                        combined = value;
                        first = false;
                    }
                    else
                    {
                        const float blend = clamp01(0.5f + 0.5f * (value - combined) / 0.75f);
                        combined = value * (1 - blend) + combined * blend - 0.75f * blend * (1 - blend);
                    }
                }
                field[nodes.index(x, y, z)] = combined;
            }
    if (!cfg.silhouetteOnly)
    {
        std::vector<float> refined = field;
        const std::array<Vec3f, 5> directions{{{0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}}};
        for (int x = 1; x < nodes.x - 1; ++x)
            for (int y = 1; y < nodes.y - 1; ++y)
                for (int z = 1; z < nodes.z - 1; ++z)
                {
                    const size_t i = nodes.index(x, y, z);
                    if (std::abs(field[i]) > 3)
                        continue;
                    Vec3f outward{field[nodes.index(x - 1, y, z)] - field[nodes.index(x + 1, y, z)],
                                  field[nodes.index(x, y - 1, z)] - field[nodes.index(x, y + 1, z)],
                                  field[nodes.index(x, y, z - 1)] - field[nodes.index(x, y, z + 1)]};
                    const float length = (float)cv::norm(outward);
                    if (length < 1e-6f)
                        continue;
                    outward *= 1.f / length;
                    float weights = 0, inset = 0;
                    for (int id = 0; id < 5; ++id)
                    {
                        if (depths[id].empty())
                            continue;
                        const float facing = std::max(0.f, outward.dot(directions[id]));
                        if (facing < 0.01f)
                            continue;
                        auto uv = coordinates(id, x - pad + 0.5f, y - pad + 0.5f, z - pad + 0.5f);
                        const float w = facing * facing * sampleScalar(confidences[id], uv[0], uv[1]);
                        inset += w * (1 - sampleScalar(depths[id], uv[0], uv[1]));
                        weights += w;
                    }
                    if (weights > 1e-6f)
                        refined[i] -= 0.2f * cfg.depthRelief * inset / weights;
                }
        field.swap(refined);
    }
    std::cout << "Continuous silhouette distance field: sub-voxel contours, bounded depth relief"
              << std::endl;
}
static void discardNumericalIslands(Mesh &mesh)
{
    std::vector<std::vector<uint32_t>> graph(mesh.vertices.size());
    for (const auto &t : mesh.triangles)
        for (int i = 0; i < 3; ++i)
        {
            graph[t[i]].push_back(t[(i + 1) % 3]);
            graph[t[(i + 1) % 3]].push_back(t[i]);
        }
    std::vector<int> component(mesh.vertices.size(), -1), sizes;
    for (uint32_t start = 0; start < mesh.vertices.size(); ++start)
    {
        if (component[start] >= 0 || graph[start].empty())
            continue;
        const int id = (int)sizes.size();
        int size = 0;
        std::queue<uint32_t> queue;
        queue.push(start);
        component[start] = id;
        while (!queue.empty())
        {
            uint32_t v = queue.front();
            queue.pop();
            ++size;
            for (uint32_t n : graph[v])
                if (component[n] < 0)
                {
                    component[n] = id;
                    queue.push(n);
                }
        }
        sizes.push_back(size);
    }
    if (sizes.size() < 2)
        return;
    const int cutoff = std::max(64, (int)(*std::max_element(sizes.begin(), sizes.end()) * 0.0002));
    std::vector<uint32_t> remap(mesh.vertices.size(), UINT32_MAX);
    Mesh kept;
    for (const auto &t : mesh.triangles)
    {
        if (sizes[component[t[0]]] < cutoff)
            continue;
        std::array<uint32_t, 3> copy;
        for (int i = 0; i < 3; ++i)
        {
            if (remap[t[i]] == UINT32_MAX)
            {
                remap[t[i]] = (uint32_t)kept.vertices.size();
                kept.vertices.push_back(mesh.vertices[t[i]]);
            }
            copy[i] = remap[t[i]];
        }
        kept.triangles.push_back(copy);
    }
    std::cout << "Discarded " << mesh.vertices.size() - kept.vertices.size()
              << " numerical-island vertices (automatic masks only)" << std::endl;
    mesh = std::move(kept);
}
static Mesh extractTetrahedraMesh(const std::vector<float> &field, const Grid &fieldGrid, const Grid &g,
                                  int pad)
{
    auto gradient = [&](int x, int y, int z) {
        auto f = [&](int a, int b, int c) {
            return field[fieldGrid.index(std::clamp(a, 0, fieldGrid.x - 1), std::clamp(b, 0, fieldGrid.y - 1),
                                         std::clamp(c, 0, fieldGrid.z - 1))];
        };
        return Vec3f(f(x + 1, y, z) - f(x - 1, y, z), f(x, y + 1, z) - f(x, y - 1, z),
                     f(x, y, z + 1) - f(x, y, z - 1));
    };
    const int xyz[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                           {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    const int tetra[6][4] = {{0, 5, 1, 6}, {0, 1, 2, 6}, {0, 2, 3, 6},
                             {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}};
    Mesh mesh;
    std::unordered_map<uint64_t, uint32_t> crossings;
    for (int x = 0; x < fieldGrid.x - 1; ++x)
        for (int y = 0; y < fieldGrid.y - 1; ++y)
            for (int z = 0; z < fieldGrid.z - 1; ++z)
            {
                float values[8];
                uint32_t node[8];
                Vec3f positions[8], normals[8];
                bool inside = false, outside = false;
                for (int i = 0; i < 8; ++i)
                {
                    int a = x + xyz[i][0], b = y + xyz[i][1], c = z + xyz[i][2];
                    node[i] = (uint32_t)fieldGrid.index(a, b, c);
                    values[i] = field[node[i]];
                    inside |= values[i] > 0;
                    outside |= values[i] <= 0;
                    positions[i] = Vec3f(a - pad + 0.5f - g.x * 0.5f, b - pad + 0.5f - g.y * 0.5f,
                                         c - pad + 0.5f - g.z * 0.5f);
                }
                if (!inside || !outside)
                    continue;
                for (int i = 0; i < 8; ++i)
                    normals[i] = -gradient(x + xyz[i][0], y + xyz[i][1], z + xyz[i][2]);
                auto crossing = [&](int a, int b) {
                    // One vertex per lattice edge, in a canonical endpoint order.
                    // Endpoint snapping can merge some crossings but not their
                    // neighbors, creating holes when tiny triangles are removed.
                    if (node[a] > node[b])
                        std::swap(a, b);
                    const uint64_t key = edgeKey(node[a], node[b]);
                    auto found = crossings.find(key);
                    if (found != crossings.end())
                        return found->second;
                    const double t =
                        std::clamp((double)values[a] / ((double)values[a] - values[b]), 1e-4, 1.0 - 1e-4);
                    MeshVertex vertex;
                    for (int axis = 0; axis < 3; ++axis)
                        vertex.position[axis] =
                            (float)((double)positions[a][axis] +
                                    ((double)positions[b][axis] - positions[a][axis]) * t);
                    vertex.normal = normals[a] * (float)(1.0 - t) + normals[b] * (float)t;
                    uint32_t id = (uint32_t)mesh.vertices.size();
                    mesh.vertices.push_back(vertex);
                    crossings[key] = id;
                    return id;
                };
                auto triangle = [&](uint32_t a, uint32_t b, uint32_t c, const Vec3f &outward) {
                    if (a == b || b == c || a == c)
                        return;
                    Vec3f n = (mesh.vertices[b].position - mesh.vertices[a].position)
                                  .cross(mesh.vertices[c].position - mesh.vertices[a].position);
                    if (cv::norm(n) < 1e-10)
                        return;
                    if (n.dot(outward) < 0)
                        std::swap(b, c);
                    mesh.triangles.push_back({a, b, c});
                };
                for (const auto &t : tetra)
                {
                    int in[4], out[4], ni = 0, no = 0;
                    for (int i : t)
                    {
                        if (values[i] > 0)
                            in[ni++] = i;
                        else
                            out[no++] = i;
                    }
                    if (ni == 0 || no == 0)
                        continue;
                    // The isosurface is planar within each tetrahedron. Orient it
                    // toward its negative-valued corners; averaged lattice gradients
                    // can point across a sharp crease and reverse individual faces.
                    Vec3f insideCenter{0, 0, 0}, outsideCenter{0, 0, 0};
                    for (int i = 0; i < ni; ++i)
                        insideCenter += positions[in[i]];
                    for (int i = 0; i < no; ++i)
                        outsideCenter += positions[out[i]];
                    const Vec3f outward = outsideCenter * (1.f / no) - insideCenter * (1.f / ni);
                    if (ni == 1)
                        triangle(crossing(in[0], out[0]), crossing(in[0], out[1]), crossing(in[0], out[2]),
                                 outward);
                    else if (ni == 3)
                        triangle(crossing(out[0], in[0]), crossing(out[0], in[1]), crossing(out[0], in[2]),
                                 outward);
                    else if (ni == 2)
                    {
                        auto a = crossing(in[0], out[0]), b = crossing(in[0], out[1]),
                             c = crossing(in[1], out[0]), d = crossing(in[1], out[1]);
                        triangle(a, b, c, outward);
                        triangle(b, d, c, outward);
                    }
                }
            }
    return mesh;
}
static Mesh buildSmoothMesh(const Volume &volume, const Config &cfg, const std::vector<View> *views = nullptr,
                            const std::vector<Projection> *projections = nullptr)
{
    const Grid &g = volume.grid;
    const int pad = 2;
    const Grid fieldGrid{g.x + 2 * pad, g.y + 2 * pad, g.z + 2 * pad};
    std::vector<float> field(fieldGrid.count(), 0), tmp(field.size());
    for (int x = 0; x < g.x; ++x)
        for (int y = 0; y < g.y; ++y)
            for (int z = 0; z < g.z; ++z)
                field[fieldGrid.index(x + pad, y + pad, z + pad)] =
                    volume.occupied[g.index(x, y, z)] ? 1.f : 0.f;
    // Separable compact Gaussian. Padding closes every outer boundary of the mesh.
    for (int axis = 0; axis < 3; ++axis)
    {
        for (int x = 0; x < fieldGrid.x; ++x)
            for (int y = 0; y < fieldGrid.y; ++y)
                for (int z = 0; z < fieldGrid.z; ++z)
                {
                    float sum = 0;
                    for (int d = -1; d <= 1; ++d)
                    {
                        int a = x + (axis == 0 ? d : 0), b = y + (axis == 1 ? d : 0),
                            c = z + (axis == 2 ? d : 0);
                        if (fieldGrid.contains(a, b, c))
                            sum += field[fieldGrid.index(a, b, c)] * (d ? 0.2f : 0.6f);
                    }
                    tmp[fieldGrid.index(x, y, z)] = sum;
                }
        field.swap(tmp);
    }
    for (float &v : field)
        v -= 0.5f;
    if (views && projections && cfg.continuousSurface && (cfg.anchorDepth || cfg.silhouetteOnly))
        continuousHullField(field, fieldGrid, g, pad, *views, *projections, cfg);
    Mesh mesh = extractTetrahedraMesh(field, fieldGrid, g, pad);
    if (views &&
        std::none_of(views->begin(), views->end(), [](const auto &v) { return v.authoritativeMask; }))
        discardNumericalIslands(mesh);
    smoothMesh(mesh, cfg.smoothIterations);
    meshAudit(mesh);
    std::cout << "Smooth surface: " << mesh.vertices.size() << " welded vertices, " << mesh.triangles.size()
              << " triangles, " << mesh.components << " components, " << mesh.boundaryEdges << " open edges"
              << ", " << mesh.inconsistentOrientedEdges << " inconsistent oriented edges" << std::endl;
    return mesh;
}

static std::array<Mat, 5> textureImages;
static std::array<Mat, 5> textureMasks, textureMaskDistance;
static std::array<bool, 5> textureReal{{true, true, true, false, true}};
static void prepareTextures(const std::vector<View> &views)
{
    for (const auto &view : views)
    {
        const int id = (int)view.id;
        const double scale = std::min(1., 2048. / std::max(view.rgb.cols, view.rgb.rows));
        Mat image, mask;
        cv::resize(view.rgb, image, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(view.mask, mask, image.size(), 0, 0, cv::INTER_NEAREST);
        Mat labels, distance;
        // Extend material colors beyond the silhouette so antialiasing and smooth
        // boundary positions cannot sample white background into the car paint.
        cv::distanceTransform(255 - mask, distance, labels, cv::DIST_L2, 5, cv::DIST_LABEL_PIXEL);
        double largestLabel;
        cv::minMaxLoc(labels, nullptr, &largestLabel);
        std::vector<Vec3b> nearest((size_t)largestLabel + 1);
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x)
                if (mask.at<uchar>(y, x))
                    nearest[labels.at<int>(y, x)] = image.at<Vec3b>(y, x);
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x)
                if (!mask.at<uchar>(y, x))
                    image.at<Vec3b>(y, x) = nearest[labels.at<int>(y, x)];
        textureImages[id] = image;
        mask.convertTo(textureMasks[id], CV_32F, 1. / 255.);
        textureMaskDistance[id] = distance;
        textureReal[id] = view.real;
    }
}
static std::array<Mat, 5> textureVisibilityDepth(const Mesh &mesh, const Grid &grid,
                                                 const std::array<Vec3f, 5> &directions)
{
    std::array<Mat, 5> result;
    for (int id = 0; id < 5; ++id)
    {
        if (textureImages[id].empty())
            continue;
        const int w = std::max(128, (depthAxis((ViewId)id) == 0 ? grid.z : grid.x) * 4);
        const int h = std::max(128, (id == (int)ViewId::Top ? grid.z : grid.y) * 4);
        Mat depth(h, w, CV_32F, cv::Scalar(-1e6f));
        for (const auto &t : mesh.triangles)
        {
            if ((mesh.vertices[t[0]].normal + mesh.vertices[t[1]].normal + mesh.vertices[t[2]].normal)
                    .dot(directions[id]) <= 0)
                continue;
            cv::Vec2f p[3];
            float z[3];
            for (int k = 0; k < 3; ++k)
            {
                auto uv = textureUV((ViewId)id, mesh.vertices[t[k]].position, grid);
                p[k] = {uv[0] * (w - 1), (1 - uv[1]) * (h - 1)};
                z[k] = mesh.vertices[t[k]].position.dot(directions[id]);
            }
            const float area =
                (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
            if (std::abs(area) < 1e-7f)
                continue;
            int x0 = std::clamp((int)std::floor(std::min({p[0][0], p[1][0], p[2][0]})), 0, w - 1);
            int x1 = std::clamp((int)std::ceil(std::max({p[0][0], p[1][0], p[2][0]})), 0, w - 1);
            int y0 = std::clamp((int)std::floor(std::min({p[0][1], p[1][1], p[2][1]})), 0, h - 1);
            int y1 = std::clamp((int)std::ceil(std::max({p[0][1], p[1][1], p[2][1]})), 0, h - 1);
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x)
                {
                    const float b =
                        ((x - p[0][0]) * (p[2][1] - p[0][1]) - (y - p[0][1]) * (p[2][0] - p[0][0])) / area;
                    const float c =
                        ((p[1][0] - p[0][0]) * (y - p[0][1]) - (p[1][1] - p[0][1]) * (x - p[0][0])) / area;
                    const float a = 1 - b - c;
                    if (a >= -1e-5f && b >= -1e-5f && c >= -1e-5f)
                        depth.at<float>(y, x) =
                            std::max(depth.at<float>(y, x), a * z[0] + b * z[1] + c * z[2]);
                }
        }
        // A one-pixel conservative expansion avoids cracks at projected vertices.
        cv::dilate(depth, result[id], Mat::ones(3, 3, CV_8U));
    }
    return result;
}
static void colorMesh(Mesh &mesh, const Grid &grid, const Config &cfg)
{
    fitPhotoProjection(mesh, cfg);
    const float front = cfg.frontPhotoPitch * 0.01745329252f, back = cfg.backPhotoPitch * 0.01745329252f;
    const std::array<Vec3f, 5> directions{{{0, std::sin(front), std::cos(front)},
                                           {0, std::sin(back), -std::cos(back)},
                                           {1, 0, 0},
                                           {-1, 0, 0},
                                           {0, 1, 0}}};
    const auto visibilityDepth = textureVisibilityDepth(mesh, grid, directions);
    for (auto &v : mesh.vertices)
    {
        float sum = 0;
        for (int id = 0; id < 5; ++id)
        {
            if (textureImages[id].empty())
                continue;
            float facing = std::max(0.f, v.normal.dot(directions[id]));
            if (facing < 0.2f)
                continue;
            const auto uv = textureUV((ViewId)id, v.position, grid);
            const auto &mask = textureMasks[id];
            float visibility = 0;
            if (uv[0] >= 0 && uv[0] <= 1 && uv[1] >= 0 && uv[1] <= 1)
            {
                const float x = uv[0] * mask.cols - 0.5f, y = (1.f - uv[1]) * mask.rows - 0.5f;
                visibility = std::max(sampleScalar(mask, x, y),
                                      clamp01(1.f - sampleScalar(textureMaskDistance[id], x, y) / 12.f));
                const auto &depth = visibilityDepth[id];
                const float closest =
                    sampleScalar(depth, uv[0] * (depth.cols - 1), (1 - uv[1]) * (depth.rows - 1));
                const float hidden = closest - v.position.dot(directions[id]);
                visibility *= clamp01((1.5f - hidden) / 1.0f);
            }
            float weight = std::pow(facing, 6.f) * (textureReal[id] ? 1.f : 0.65f) * visibility;
            v.textureWeight[id] = weight;
            sum += weight;
        }
        v.color = Vec3f(0.07f, 0.075f, 0.08f);
        if (sum < 1e-8f)
            continue; // Unobserved underside stays neutral rather than borrowing roof paint.
        v.color = Vec3f(0, 0, 0);
        for (int id = 0; id < 5; ++id)
        {
            v.textureWeight[id] /= sum;
            if (v.textureWeight[id] > 0)
                v.color += bilinearColor(textureImages[id], textureUV((ViewId)id, v.position, grid)) *
                           v.textureWeight[id];
        }
    }
}

struct Face
{
    std::array<Vec3f, 4> corners;
    Vec3f normal, color;
};
static const std::array<std::array<int, 3>, 6> faceDirections{
    {{{1, 0, 0}}, {{-1, 0, 0}}, {{0, 1, 0}}, {{0, -1, 0}}, {{0, 0, 1}}, {{0, 0, -1}}}};
static Vec3f surfaceColor(const std::vector<Projection> &projections, const Grid &g, int x, int y, int z,
                          int face)
{
    const ViewId preferred = face == 0   ? ViewId::Left
                             : face == 1 ? ViewId::Right
                             : face == 2 ? ViewId::Top
                             : face == 4 ? ViewId::Front
                                         : ViewId::Back;
    Vec3f sum{0, 0, 0};
    float weights = 0.f;
    for (int id = 0; id < 5; ++id)
        for (const auto &p : projections)
        {
            if ((int)p.id != id)
                continue;
            const Cell &c = projectCell(p, x, y, z);
            if (c.state != CellState::Foreground)
                continue;
            // A real view facing this surface owns its appearance. Bottom has no view.
            if (p.real && p.id == preferred && face != 3 && c.coverage >= 0.5f)
                return c.color;
            const float dist = std::abs(axisCoordinate(p.id, g, x, y, z) - surfaceBound(p, c));
            const float weight = (p.real ? 1.f : 0.1f) * (0.2f + c.confidence) / (0.04f + dist);
            sum += c.color * weight;
            weights += weight;
        }
    return weights > 0 ? sum * (1.f / weights) : Vec3f(0.7f, 0.7f, 0.7f);
}
static std::vector<Face> buildFaces(const Volume &v, const std::vector<Projection> &projections)
{
    std::vector<Face> result;
    const Grid &g = v.grid;
    for (int x = 0; x < g.x; ++x)
        for (int y = 0; y < g.y; ++y)
            for (int z = 0; z < g.z; ++z)
            {
                if (!v.occupied[g.index(x, y, z)])
                    continue;
                const float a = (float)x - g.x * 0.5f, b = (float)y - g.y * 0.5f, c = (float)z - g.z * 0.5f;
                const std::array<Vec3f, 8> pts{{{a, b, c},
                                                {a + 1, b, c},
                                                {a + 1, b + 1, c},
                                                {a, b + 1, c},
                                                {a, b, c + 1},
                                                {a + 1, b, c + 1},
                                                {a + 1, b + 1, c + 1},
                                                {a, b + 1, c + 1}}};
                const int indices[6][4] = {{1, 2, 6, 5}, {4, 7, 3, 0}, {3, 7, 6, 2},
                                           {0, 1, 5, 4}, {4, 5, 6, 7}, {0, 3, 2, 1}};
                for (int f = 0; f < 6; ++f)
                {
                    const auto &d = faceDirections[f];
                    const int nx = x + d[0], ny = y + d[1], nz = z + d[2];
                    if (g.contains(nx, ny, nz) && v.occupied[g.index(nx, ny, nz)])
                        continue;
                    Face face;
                    for (int i = 0; i < 4; ++i)
                        face.corners[i] = pts[indices[f][i]];
                    face.normal = Vec3f((float)d[0], (float)d[1], (float)d[2]);
                    face.color = surfaceColor(projections, g, x, y, z, f);
                    result.push_back(face);
                }
            }
    return result;
}

// Reload our own exported mesh without spending another inference pass.
static std::pair<Grid, std::vector<Face>> loadPreview(const fs::path &path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot read preview PLY: " + path.string());
    std::string line;
    std::getline(in, line);
    if (line != "ply")
        throw std::runtime_error("Preview needs a V5 ASCII PLY");
    bool format = false, ownFormat = false, header = false;
    size_t vertices = 0, triangles = 0;
    while (std::getline(in, line))
    {
        if (line == "format ascii 1.0")
            format = true;
        if (line == "comment V5 exposed voxel faces; unit = one voxel")
            ownFormat = true;
        if (line.rfind("element vertex ", 0) == 0)
            vertices = std::stoull(line.substr(15));
        if (line.rfind("element face ", 0) == 0)
            triangles = std::stoull(line.substr(13));
        if (line == "end_header")
        {
            header = true;
            break;
        }
    }
    if (!format || !ownFormat || !header || vertices == 0 || vertices % 4 || triangles != vertices / 2)
        throw std::runtime_error("Preview only supports nonempty surface.ply exported by V5");
    std::vector<Face> faces(vertices / 4);
    Vec3f extent{0, 0, 0};
    for (auto &f : faces)
        for (int k = 0; k < 4; ++k)
        {
            Vec3f pos, normal, color;
            if (!(in >> pos[0] >> pos[1] >> pos[2] >> normal[0] >> normal[1] >> normal[2] >> color[0] >>
                  color[1] >> color[2]))
                throw std::runtime_error("Incomplete preview vertex data");
            for (int c = 0; c < 3; ++c)
            {
                if (!std::isfinite(pos[c]) || !std::isfinite(normal[c]) || !std::isfinite(color[c]))
                    throw std::runtime_error("Nonfinite preview vertex");
                extent[c] = std::max(extent[c], std::abs(pos[c]));
            }
            f.corners[k] = pos;
            if (!k)
            {
                f.normal = normal;
                f.color = color * (1.f / 255.f);
            }
        }
    for (size_t i = 0; i < triangles; ++i)
    {
        int count;
        size_t a, b, c;
        if (!(in >> count >> a >> b >> c) || count != 3 || a >= vertices || b >= vertices || c >= vertices)
            throw std::runtime_error("Invalid preview triangle indices");
    }
    Grid grid{std::max(1, (int)std::ceil(2 * extent[0])), std::max(1, (int)std::ceil(2 * extent[1])),
              std::max(1, (int)std::ceil(2 * extent[2]))};
    return {grid, std::move(faces)};
}

static std::pair<Grid, Mesh> loadSmoothPreview(const fs::path &path, const Config &cfg)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot read preview PLY: " + path.string());
    std::string line;
    std::getline(in, line);
    if (line != "ply")
        throw std::runtime_error("Preview needs a V5 ASCII PLY");
    bool format = false, own = false, header = false;
    size_t vertices = 0, triangles = 0;
    while (std::getline(in, line))
    {
        if (line == "format ascii 1.0")
            format = true;
        if (line.rfind("comment V5 ", 0) == 0)
            own = true;
        if (line.rfind("element vertex ", 0) == 0)
            vertices = std::stoull(line.substr(15));
        if (line.rfind("element face ", 0) == 0)
            triangles = std::stoull(line.substr(13));
        if (line == "end_header")
        {
            header = true;
            break;
        }
    }
    if (!format || !own || !header || !vertices || !triangles)
        throw std::runtime_error("Unsupported V5 PLY");
    Mesh mesh;
    mesh.vertices.resize(vertices);
    mesh.triangles.resize(triangles);
    Vec3f extent{0, 0, 0};
    for (auto &v : mesh.vertices)
    {
        if (!(in >> v.position[0] >> v.position[1] >> v.position[2] >> v.normal[0] >> v.normal[1] >>
              v.normal[2] >> v.color[0] >> v.color[1] >> v.color[2]))
            throw std::runtime_error("Incomplete mesh vertex data");
        for (int c = 0; c < 3; ++c)
        {
            if (!std::isfinite(v.position[c]) || !std::isfinite(v.normal[c]) || !std::isfinite(v.color[c]))
                throw std::runtime_error("Nonfinite mesh vertex");
            extent[c] = std::max(extent[c], std::abs(v.position[c]));
        }
        v.color *= 1.f / 255.f;
    }
    for (auto &t : mesh.triangles)
    {
        int n;
        if (!(in >> n >> t[0] >> t[1] >> t[2]) || n != 3 || t[0] >= vertices || t[1] >= vertices ||
            t[2] >= vertices)
            throw std::runtime_error("Invalid mesh triangle");
    }
    Grid grid{std::max(1, (int)std::ceil(extent[0] * 2)), std::max(1, (int)std::ceil(extent[1] * 2)),
              std::max(1, (int)std::ceil(extent[2] * 2))};
    // The report retains the original grid used by the image projections.
    std::ifstream report(path.parent_path() / "report.txt");
    while (std::getline(report, line))
        if (line.rfind("grid=", 0) == 0)
        {
            std::string dimensions = line.substr(5);
            std::replace(dimensions.begin(), dimensions.end(), 'x', ' ');
            std::istringstream dims(dimensions);
            dims >> grid.x >> grid.y >> grid.z;
        }
    for (int id = 0; id < 5; ++id)
    {
        const auto stem = path.parent_path() / viewName((ViewId)id);
        View view;
        view.id = (ViewId)id;
        view.real = id != (int)ViewId::Right;
        view.rgb = cv::imread(stem.string() + "_aligned.png");
        view.mask = cv::imread(stem.string() + "_mask.png", cv::IMREAD_GRAYSCALE);
        if (view.rgb.empty() || view.mask.empty())
            continue;
        prepareTextures({view});
    }
    if (!textureImages[0].empty())
        colorMesh(mesh, grid, cfg);
    return {grid, std::move(mesh)};
}

static void saveDebug(const std::vector<View> &views, const Config &cfg)
{
    for (const auto &v : views)
    {
        const std::string stem = (cfg.outputDir / viewName(v.id)).string();
        Mat depth8(v.depth.size(), CV_8U, cv::Scalar(0));
        for (int y = 0; y < v.depth.rows; ++y)
            for (int x = 0; x < v.depth.cols; ++x)
                if (v.mask.at<uchar>(y, x) && std::isfinite(v.depth.at<float>(y, x)))
                    depth8.at<uchar>(y, x) = (uchar)(1 + std::round(clamp01(v.depth.at<float>(y, x)) * 254));
        if (!cv::imwrite(stem + "_aligned.png", v.rgb) || !cv::imwrite(stem + "_mask.png", v.mask) ||
            !cv::imwrite(stem + "_depth.png", depth8))
            throw std::runtime_error("Cannot save debug images: " + stem);
    }
}
static void saveProjectionDebug(const std::vector<Projection> &projections, const Config &cfg)
{
    for (const auto &p : projections)
    {
        Mat states(p.bSize, p.aSize, CV_8UC3, cv::Scalar(128, 128, 128));
        Mat confidence(p.bSize, p.aSize, CV_8U, cv::Scalar(0));
        for (int a = 0; a < p.aSize; ++a)
            for (int b = 0; b < p.bSize; ++b)
            {
                const Cell &c = p.at(a, b);
                const int row = p.id == ViewId::Top ? b : p.bSize - b - 1;
                states.at<Vec3b>(row, a) = c.state == CellState::Foreground   ? Vec3b(255, 255, 255)
                                           : c.state == CellState::Background ? Vec3b(0, 0, 0)
                                                                              : Vec3b(128, 128, 128);
                confidence.at<uchar>(row, a) = (uchar)std::round(clamp01(c.confidence) * 255);
            }
        const std::string stem = (cfg.outputDir / viewName(p.id)).string();
        if (!cv::imwrite(stem + "_states.png", states) || !cv::imwrite(stem + "_confidence.png", confidence))
            throw std::runtime_error("Cannot save projection diagnostics");
    }
}
static void exportResult(const Volume &v, const Mesh &mesh, const std::vector<Projection> &projections,
                         const Config &cfg)
{
    std::ofstream ply(cfg.outputDir / "surface.ply");
    if (!ply)
        throw std::runtime_error("Cannot write surface.ply");
    ply << "ply\nformat ascii 1.0\ncomment V5 welded smooth surface; unit = one voxel\n"
        << "element vertex " << mesh.vertices.size()
        << "\nproperty float x\nproperty float y\nproperty float z\n"
        << "property float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar "
           "green\n"
        << "property uchar blue\nelement face " << mesh.triangles.size()
        << "\nproperty list uchar int vertex_indices\nend_header\n";
    ply << std::setprecision(8);
    for (const auto &vertex : mesh.vertices)
    {
        const auto &p = vertex.position;
        const auto &n = vertex.normal;
        const auto &c = vertex.color;
        ply << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << n[0] << ' ' << n[1] << ' ' << n[2] << ' '
            << (int)std::round(clamp01(c[0]) * 255) << ' ' << (int)std::round(clamp01(c[1]) * 255) << ' '
            << (int)std::round(clamp01(c[2]) * 255) << '\n';
    }
    for (const auto &t : mesh.triangles)
        ply << "3 " << t[0] << ' ' << t[1] << ' ' << t[2] << '\n';
    if (!ply)
        throw std::runtime_error("Failed to finish PLY export");
    std::ofstream report(cfg.outputDir / "report.txt");
    if (!report)
        throw std::runtime_error("Cannot write report.txt");
    report << "grid=" << v.grid.x << 'x' << v.grid.y << 'x' << v.grid.z << '\n'
           << "real_views="
           << std::count_if(projections.begin(), projections.end(), [](const auto &p) { return p.real; })
           << '\n'
           << "mirrored_priors="
           << std::count_if(projections.begin(), projections.end(), [](const auto &p) { return !p.real; })
           << '\n'
           << "candidate_voxels=" << std::count(v.hull.begin(), v.hull.end(), (uint8_t)1) << '\n'
           << "occupied_voxels=" << std::count(v.occupied.begin(), v.occupied.end(), (uint8_t)1) << '\n'
           << "vertices=" << mesh.vertices.size() << "\ntriangles=" << mesh.triangles.size() << '\n'
           << "mesh_components=" << mesh.components << "\nboundary_edges=" << mesh.boundaryEdges
           << "\nnonmanifold_edges=" << mesh.nonmanifoldEdges << '\n'
           << "inconsistent_oriented_edges=" << mesh.inconsistentOrientedEdges << '\n'
           << "depth_mode=" << (cfg.silhouetteOnly ? "silhouette" : "relative_depth_soft_constraints") << '\n'
           << "input_long_side=" << cfg.inputLongSide << "\nreject_strength=" << cfg.rejectStrength << '\n'
           << "depth_mapping=" << (cfg.anchorDepth ? "silhouette_anchored_relief" : "legacy_absolute_range")
           << '\n'
           << "maximum_depth_relief_voxels=" << cfg.depthRelief << "\nsurface=marching_tetrahedra_taubin\n"
           << "camera_model=axis_aligned_orthographic_assumption\n"
           << "metric_scale=not_calibrated\n";
    report << "continuous_surface=" << cfg.continuousSurface
           << "\nfront_photo_pitch_deg=" << cfg.frontPhotoPitch
           << "\nback_photo_pitch_deg=" << cfg.backPhotoPitch << '\n';
    for (const auto &p : projections)
        if (p.real)
        {
            std::vector<uint8_t> hit(p.cells.size(), 0);
            for (int x = 0; x < v.grid.x; ++x)
                for (int y = 0; y < v.grid.y; ++y)
                    for (int z = 0; z < v.grid.z; ++z)
                    {
                        if (!v.occupied[v.grid.index(x, y, z)])
                            continue;
                        const size_t i = p.id == ViewId::Left || p.id == ViewId::Right
                                             ? (size_t)z * p.bSize + y
                                         : p.id == ViewId::Top ? (size_t)x * p.bSize + z
                                                               : (size_t)x * p.bSize + y;
                        hit[i] = 1;
                    }
            int expected = 0, covered = 0, united = 0;
            for (size_t i = 0; i < hit.size(); ++i)
            {
                bool fg = p.cells[i].state == CellState::Foreground;
                expected += fg;
                covered += fg && hit[i];
                united += fg || hit[i];
            }
            report << viewName(p.id) << "_foreground_coverage=" << (expected ? (double)covered / expected : 0)
                   << '\n'
                   << viewName(p.id) << "_silhouette_iou=" << (united ? (double)covered / united : 0) << '\n';
            Mat projected(p.bSize, p.aSize, CV_8U, cv::Scalar(0));
            for (const auto &t : mesh.triangles)
            {
                cv::Point points[3];
                for (int k = 0; k < 3; ++k)
                {
                    const auto &vtx = mesh.vertices[t[k]].position;
                    const float a =
                        depthAxis(p.id) == 0 ? vtx[2] + v.grid.z * 0.5f : vtx[0] + v.grid.x * 0.5f;
                    const float b = p.id == ViewId::Top ? vtx[2] + v.grid.z * 0.5f : v.grid.y * 0.5f - vtx[1];
                    points[k] =
                        cv::Point((int)std::round((a - 0.5f) * 256), (int)std::round((b - 0.5f) * 256));
                }
                cv::fillConvexPoly(projected, points, 3, cv::Scalar(255), cv::LINE_8, 8);
            }
            int meshIntersection = 0, meshUnion = 0;
            for (int a = 0; a < p.aSize; ++a)
                for (int b = 0; b < p.bSize; ++b)
                {
                    const bool foreground = p.at(a, b).state == CellState::Foreground;
                    const int row = p.id == ViewId::Top ? b : p.bSize - 1 - b;
                    const bool seen = projected.at<uchar>(row, a) > 0;
                    meshIntersection += foreground && seen;
                    meshUnion += foreground || seen;
                }
            report << viewName(p.id)
                   << "_mesh_foreground_coverage=" << (expected ? (double)meshIntersection / expected : 0)
                   << '\n'
                   << viewName(p.id)
                   << "_mesh_silhouette_iou=" << (meshUnion ? (double)meshIntersection / meshUnion : 0)
                   << '\n';
        }
}

static Mesh displayMesh;
static std::array<GLuint, 6> displayLists{};
static std::array<GLuint, 5> textureIds{};
static float angleX = 25, angleY = 55, cameraDistance = 600, modelSize = 256;
static bool dragging = false, lighting = true, textured = true;
static int mouseX = 0, mouseY = 0;
static fs::path snapshotPath;

static float fitCamera(float yaw, float pitch)
{
    const float y = yaw * 0.01745329252f, x = pitch * 0.01745329252f;
    float distance = 10.f;
    for (const auto &vertex : displayMesh.vertices)
    {
        const auto &p = vertex.position;
        float ax = std::cos(y) * p[0] + std::sin(y) * p[2], az = -std::sin(y) * p[0] + std::cos(y) * p[2];
        float ay = std::cos(x) * p[1] - std::sin(x) * az, bz = std::sin(x) * p[1] + std::cos(x) * az;
        distance = std::max(
            distance, bz + 1.15f * std::max(std::abs(ay) / 0.52057f, std::abs(ax) / (0.52057f * 1000 / 760)));
    }
    return distance;
}
static void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    const GLfloat light[] = {-0.4f, 0.7f, 1.f, 0.f};
    glLightfv(GL_LIGHT0, GL_POSITION, light);
    glTranslatef(0, 0, -cameraDistance);
    glRotatef(angleX, 1, 0, 0);
    glRotatef(angleY, 0, 1, 0);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
    if (textured)
    {
        glDisable(GL_LIGHTING);
        glCallList(displayLists[0]);
        if (lighting)
            glEnable(GL_LIGHTING);
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glDepthMask(GL_FALSE);
        glDepthFunc(GL_EQUAL);
        for (int id = 0; id < 5; ++id)
            if (textureIds[id])
            {
                glBindTexture(GL_TEXTURE_2D, textureIds[id]);
                glCallList(displayLists[id + 1]);
            }
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D);
    }
    else
    {
        if (lighting)
            glEnable(GL_LIGHTING);
        else
            glDisable(GL_LIGHTING);
        glColor3f(0.65f, 0.67f, 0.70f);
        glBegin(GL_TRIANGLES);
        for (const auto &triangle : displayMesh.triangles)
            for (uint32_t i : triangle)
            {
                const auto &vertex = displayMesh.vertices[i];
                glNormal3fv(vertex.normal.val);
                glVertex3fv(vertex.position.val);
            }
        glEnd();
    }
    if (!snapshotPath.empty())
    {
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        Mat pixels(viewport[3], viewport[2], CV_8UC3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_BACK);
        glReadPixels(0, 0, viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, pixels.data);
        cv::flip(pixels, pixels, 0);
        cv::cvtColor(pixels, pixels, cv::COLOR_RGB2BGR);
        if (!cv::imwrite(snapshotPath.string(), pixels))
            throw std::runtime_error("Cannot save viewer snapshot");
        std::cout << "Viewer snapshot: " << snapshotPath.string() << std::endl;
        snapshotPath.clear();
    }
    glutSwapBuffers();
}
static void reshape(int w, int h)
{
    h = std::max(1, h);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(55., (double)w / h, 0.1, modelSize * 30.);
    glMatrixMode(GL_MODELVIEW);
}
static void mouse(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON)
    {
        dragging = state == GLUT_DOWN;
        mouseX = x;
        mouseY = y;
    }
    if (state == GLUT_DOWN)
    {
        if (button == 3)
            cameraDistance *= 0.9f;
        if (button == 4)
            cameraDistance *= 1.1f;
        cameraDistance = std::clamp(cameraDistance, modelSize * 0.2f, modelSize * 15.f);
    }
    glutPostRedisplay();
}
static void motion(int x, int y)
{
    if (!dragging)
        return;
    angleY += (x - mouseX) * 0.5f;
    angleX += (y - mouseY) * 0.5f;
    mouseX = x;
    mouseY = y;
    glutPostRedisplay();
}
static void keyboard(unsigned char key, int, int)
{
    if (key == 27)
        glutLeaveMainLoop();
    if (key == 'l' || key == 'L')
        lighting = !lighting;
    if (key == 't' || key == 'T')
        textured = !textured;
    if (key == 'r' || key == 'R')
    {
        angleX = 25;
        angleY = 55;
        cameraDistance = fitCamera(angleY, angleX);
    }
    glutPostRedisplay();
}
static void showViewer(int argc, char **argv, const Grid &grid, Mesh mesh, const Config &cfg)
{
    displayMesh = std::move(mesh);
    modelSize = (float)std::max({grid.x, grid.y, grid.z});
    angleX = cfg.snapshotPitch;
    angleY = cfg.snapshotYaw;
    cameraDistance = fitCamera(angleY, angleX);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1000, 760);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutCreateWindow("V5 complete smooth reconstruction | L: light  T: texture  R: reset  Esc: exit");
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    const GLfloat ambient[] = {0.4f, 0.4f, 0.4f, 1.f};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
    glClearColor(0.12f, 0.13f, 0.15f, 1.f);
    glShadeModel(GL_SMOOTH);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    for (int id = 0; id < 5; ++id)
        if (!textureImages[id].empty())
        {
            Mat rgb;
            cv::cvtColor(textureImages[id], rgb, cv::COLOR_BGR2RGB);
            cv::flip(rgb, rgb, 0);
            glGenTextures(1, &textureIds[id]);
            glBindTexture(GL_TEXTURE_2D, textureIds[id]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        }
    for (int pass = 0; pass < 6; ++pass)
    {
        displayLists[pass] = glGenLists(1);
        glNewList(displayLists[pass], GL_COMPILE);
        glBegin(GL_TRIANGLES);
        for (const auto &triangle : displayMesh.triangles)
        {
            if (pass > 0 && std::max({displayMesh.vertices[triangle[0]].textureWeight[pass - 1],
                                      displayMesh.vertices[triangle[1]].textureWeight[pass - 1],
                                      displayMesh.vertices[triangle[2]].textureWeight[pass - 1]}) < 0.001f)
                continue;
            for (uint32_t i : triangle)
            {
                const auto &vertex = displayMesh.vertices[i];
                if (!pass)
                {
                    const float sum =
                        std::accumulate(vertex.textureWeight.begin(), vertex.textureWeight.end(), 0.f);
                    if (sum > 0)
                        glColor3f(0, 0, 0);
                    else
                        glColor3fv(vertex.color.val);
                }
                else
                {
                    const float weight = vertex.textureWeight[pass - 1];
                    glColor3f(weight, weight, weight);
                    auto uv = textureUV((ViewId)(pass - 1), vertex.position, grid);
                    glTexCoord2fv(uv.val);
                }
                glNormal3fv(vertex.normal.val);
                glVertex3fv(vertex.position.val);
            }
        }
        glEnd();
        glEndList();
    }
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    if (cfg.snapshot)
    {
        reshape(1000, 760);
        snapshotPath = cfg.outputDir / "preview.png";
        display();
        if (cfg.snapshotViews)
        {
            const std::array<std::pair<const char *, cv::Vec2f>, 5> angles{{{"front", {0, 0}},
                                                                            {"left", {-90, 0}},
                                                                            {"right", {90, 0}},
                                                                            {"back", {180, 0}},
                                                                            {"top", {0, 90}}}};
            for (const auto &angle : angles)
            {
                angleY = angle.second[0];
                angleX = angle.second[1];
                cameraDistance = fitCamera(angleY, angleX);
                snapshotPath = cfg.outputDir / (std::string("preview_") + angle.first + ".png");
                display();
            }
            angleY = cfg.snapshotYaw;
            angleX = cfg.snapshotPitch;
            cameraDistance = fitCamera(angleY, angleX);
            textured = false;
            snapshotPath = cfg.outputDir / "preview_geometry.png";
            display();
            textured = true;
        }
        return;
    }
    glutMainLoop();
}

static void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(std::string("Self-test failed: ") + message);
}
static Projection testProjection(ViewId id, const Grid &g, float confidence = 1.f)
{
    Projection p;
    p.id = id;
    p.aSize = depthAxis(id) == 0 ? g.z : g.x;
    p.bSize = id == ViewId::Top ? g.z : g.y;
    p.cells.resize((size_t)p.aSize * p.bSize);
    for (auto &c : p.cells)
    {
        c.state = CellState::Foreground;
        c.depth = 1.f;
        c.coverage = 1.f;
        c.confidence = confidence;
        c.color = Vec3f(0.2f, 0.4f, 0.8f);
    }
    return p;
}
static void runSelfTests()
{
    View bundle;
    bundle.rgb = Mat(3, 4, CV_8UC3);
    bundle.mask = Mat(3, 4, CV_8U);
    bundle.depth = Mat(3, 4, CV_32F);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x)
        {
            uchar k = (uchar)(y * 4 + x + 1);
            bundle.rgb.at<Vec3b>(y, x) = Vec3b(k, k, k);
            bundle.mask.at<uchar>(y, x) = k;
            bundle.depth.at<float>(y, x) = (float)k;
        }
    flipBundle(bundle);
    rotateBundle(bundle);
    verifyBundle(bundle);
    for (int y = 0; y < bundle.rgb.rows; ++y)
        for (int x = 0; x < bundle.rgb.cols; ++x)
            require(bundle.rgb.at<Vec3b>(y, x)[0] == bundle.mask.at<uchar>(y, x) &&
                        bundle.mask.at<uchar>(y, x) == bundle.depth.at<float>(y, x),
                    "bundle transform correspondence");
    Config cfg;
    cfg.anchorDepth = false; // Exercise the legacy evidence equations separately.
    Mat maskSample(40, 40, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::rectangle(maskSample, cv::Rect(5, 5, 30, 30), cv::Scalar(0, 0, 180), cv::FILLED);
    cv::rectangle(maskSample, cv::Rect(15, 15, 10, 10), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(maskSample, cv::Rect(8, 8, 4, 4), cv::Scalar(220, 220, 220), cv::FILLED);
    const Mat extracted = autoMask(maskSample);
    require(extracted.at<uchar>(20, 20) == 255 && extracted.at<uchar>(9, 9) == 255,
            "automatic material holes are filled inside the silhouette");
    Cell one = resolveCell({{0.7f, Vec3f(1, 0, 0)}}, 1, 1, 1.f);
    require(one.confidence > 0 && one.state == CellState::Foreground, "one-pixel cell is preserved");
    std::vector<float> depths(100, 1.f);
    depths.push_back(1000.f);
    require(quantile(depths, 0.98f) == 1.f, "robust range ignores isolated outlier");
    Grid g{4, 4, 4};
    std::vector<Projection> p{testProjection(ViewId::Front, g), testProjection(ViewId::Left, g),
                              testProjection(ViewId::Top, g)};
    auto volume = reconstruct(p, g, cfg);
    require(std::count(volume.occupied.begin(), volume.occupied.end(), (uint8_t)1) == 64,
            "solid cube baseline");
    require(buildFaces(volume, p).size() == 96, "only exposed faces are exported");
    auto smooth = buildSmoothMesh(volume, cfg);
    require(!smooth.triangles.empty() && smooth.boundaryEdges == 0 && smooth.nonmanifoldEdges == 0 &&
                smooth.components == 1 && smooth.inconsistentOrientedEdges == 0,
            "smooth cube is welded, connected and closed");
    std::swap(smooth.triangles[0][1], smooth.triangles[0][2]);
    meshAudit(smooth);
    require(smooth.inconsistentOrientedEdges == 3 && smooth.boundaryEdges == 0,
            "winding audit distinguishes a reversed face from an open boundary");
    // Regression: very small/zero field values used to collapse crossings to
    // lattice endpoints, then drop faces and leave open edges at sharp tips.
    const Grid testNodes{6, 6, 6}, largeCoordinates{232, 256, 239};
    for (float value : {1e-8f, 1e-6f, 1.f})
    {
        std::vector<float> field(testNodes.count(), -1.f);
        field[testNodes.index(3, 3, 3)] = value;
        field[testNodes.index(2, 3, 3)] = -0.01f;
        if (value == 1.f)
            field[testNodes.index(3, 2, 3)] = 0.f;
        auto nearNode = extractTetrahedraMesh(field, testNodes, largeCoordinates, 2);
        meshAudit(nearNode);
        require(!nearNode.triangles.empty() && nearNode.components == 1 && nearNode.boundaryEdges == 0 &&
                    nearNode.nonmanifoldEdges == 0 && nearNode.inconsistentOrientedEdges == 0,
                "near-zero crossings remain closed at large float coordinates");
    }
    p[0].cells[0].confidence = 0;
    p[0].cells[0].depth = 0.f;
    auto unknown = reconstruct(p, g, cfg);
    require(unknown.occupied[g.index(0, 0, 3)] != 0, "unknown depth cannot remove a valid silhouette voxel");
    require(unknown.reject[g.index(0, 0, 3)] == 0, "unknown depth casts no rejection");
    std::reverse(p.begin(), p.end());
    auto reversed = reconstruct(p, g, cfg);
    require(unknown.occupied == reversed.occupied && unknown.support == reversed.support &&
                unknown.reject == reversed.reject,
            "view order independence");
    Projection prior = testProjection(ViewId::Right, g);
    prior.real = false;
    auto before = evaluate(p, g, cfg, 1, 1, 1);
    p.push_back(prior);
    auto after = evaluate(p, g, cfg, 1, 1, 1);
    require(before.silhouetteVotes == after.silhouetteVotes && before.depthVotes == after.depthVotes,
            "mirrored prior cannot cast an independent vote");
    p[0].cells[0].state = CellState::Background;
    require(!evaluate(p, g, cfg, 0, 0, 0).hull, "trusted background bounds the hull");
    p[0].cells[0].state = CellState::Unknown;
    require(evaluate(p, g, cfg, 0, 0, 0).hull, "unknown silhouette border is not background");
    // Compare opposite surfaces: reliable free-space evidence must actually carve.
    p.clear();
    for (auto id : {ViewId::Front, ViewId::Back, ViewId::Left, ViewId::Right, ViewId::Top})
        p.push_back(testProjection(id, g));
    for (auto &c : p[0].cells)
        c.depth = 0.1f;
    for (auto &c : p[2].cells)
        c.depth = 0.1f;
    auto carved = reconstruct(p, g, cfg);
    require(!carved.occupied[g.index(3, 3, 3)], "two reliable free-space observations carve exterior");
    cfg.anchorDepth = true;
    auto protectedBody = reconstruct(p, g, cfg);
    require(protectedBody.occupied[g.index(1, 1, 1)] != 0,
            "relative depth cannot destroy the silhouette interior");
    // A lower, upward-facing surface is hidden by a roof with the same UVs.
    // Normal-facing and mask checks alone must not let it borrow roof paint.
    for (auto &image : textureImages)
        image.release();
    for (auto &mask : textureMasks)
        mask.release();
    for (auto &distance : textureMaskDistance)
        distance.release();
    textureImages[4] = Mat(8, 8, CV_8UC3, cv::Scalar(0, 0, 255));
    textureMasks[4] = Mat(8, 8, CV_32F, cv::Scalar(1));
    textureMaskDistance[4] = Mat(8, 8, CV_32F, cv::Scalar(0));
    Mesh layers;
    for (float height : {3.f, 0.f})
        for (auto point : {cv::Vec2f(-1, -1), cv::Vec2f(1, -1), cv::Vec2f(1, 1), cv::Vec2f(-1, 1)})
        {
            MeshVertex vertex;
            vertex.position = {point[0], height, point[1]};
            vertex.normal = {0, 1, 0};
            layers.vertices.push_back(vertex);
        }
    layers.triangles = {{{0, 2, 1}}, {{0, 3, 2}}, {{4, 6, 5}}, {{4, 7, 6}}};
    colorMesh(layers, Grid{4, 4, 4}, cfg);
    require(layers.vertices[0].textureWeight[4] > 0.9f && layers.vertices[4].textureWeight[4] == 0,
            "hidden underside cannot borrow roof texture");
    for (auto &image : textureImages)
        image.release();
    for (auto &mask : textureMasks)
        mask.release();
    for (auto &distance : textureMaskDistance)
        distance.release();
    fittedPhotoProjection = false;
    std::cout << "SELF-TEST PASS: transforms, robust range, sparse cells, unknown evidence, "
                 "view order, prior votes, hull boundaries, free-space carving, closed surface, near-zero "
                 "crossings, texture occlusion"
              << std::endl;
}

static Config parseArgs(int argc, char **argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i)
    {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc)
                throw std::runtime_error("Missing value for " + key);
            return argv[i];
        };
        if (key == "--headless")
            cfg.headless = true;
        else if (key == "--snapshot")
            cfg.snapshot = true;
        else if (key == "--snapshot-views")
        {
            cfg.snapshot = true;
            cfg.snapshotViews = true;
        }
        else if (key == "--no-cache")
            cfg.useCache = false;
        else if (key == "--legacy-depth")
            cfg.anchorDepth = false;
        else if (key == "--voxel-surface")
            cfg.continuousSurface = false;
        else if (key == "--surface-margin")
            cfg.surfaceMargin = std::stof(value());
        else if (key == "--front-photo-pitch")
            cfg.frontPhotoPitch = std::stof(value());
        else if (key == "--back-photo-pitch")
            cfg.backPhotoPitch = std::stof(value());
        else if (key == "--depth-relief")
            cfg.depthRelief = std::stof(value());
        else if (key == "--smooth-iterations")
            cfg.smoothIterations = std::stoi(value());
        else if (key == "--yaw")
            cfg.snapshotYaw = std::stof(value());
        else if (key == "--pitch")
            cfg.snapshotPitch = std::stof(value());
        else if (key == "--self-test")
            cfg.selfTest = true;
        else if (key == "--silhouette-only")
            cfg.silhouetteOnly = true;
        else if (key == "--grid")
            cfg.maxGrid = std::stoi(value());
        else if (key == "--input-size")
            cfg.inputLongSide = std::stoi(value());
        else if (key == "--threads")
            cfg.threads = std::stoi(value());
        else if (key == "--depth-scale")
            cfg.depthScale = std::stof(value());
        else if (key == "--depth-offset")
            cfg.depthOffset = std::stof(value());
        else if (key == "--reject-strength")
            cfg.rejectStrength = std::stof(value());
        else if (key == "--input-dir")
            cfg.inputDir = value();
        else if (key == "--output-dir")
            cfg.outputDir = value();
        else if (key == "--model")
            cfg.model = value();
        else if (key == "--preview-ply")
            cfg.previewPly = value();
        else
        {
            bool matched = false;
            for (int j = 0; j < 5; ++j)
            {
                if (key == std::string("--") + viewName((ViewId)j) + "-scale")
                {
                    cfg.viewScale[j] = std::stof(value());
                    matched = true;
                    break;
                }
                if (key == std::string("--") + viewName((ViewId)j) + "-offset")
                {
                    cfg.viewOffset[j] = std::stof(value());
                    matched = true;
                    break;
                }
            }
            if (!matched)
                throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (cfg.maxGrid < 4 || cfg.maxGrid > 384 || cfg.inputLongSide < 140 || cfg.inputLongSide > 1036 ||
        cfg.threads < 1 || cfg.threads > 16 || !std::isfinite(cfg.depthScale) || cfg.depthScale <= 0 ||
        !std::isfinite(cfg.depthOffset) || !std::isfinite(cfg.rejectStrength) || cfg.rejectStrength <= 0)
        throw std::runtime_error("Invalid grid/input-size/thread/depth settings; see docs/usage.md");
    for (int j = 0; j < 5; ++j)
        if (!std::isfinite(cfg.viewScale[j]) || cfg.viewScale[j] <= 0 || !std::isfinite(cfg.viewOffset[j]))
            throw std::runtime_error("Invalid per-view scale/offset");
    if (cfg.headless && cfg.snapshot)
        throw std::runtime_error("Use --snapshot or --headless, not both");
    if (!std::isfinite(cfg.depthRelief) || cfg.depthRelief < 0 || cfg.depthRelief > 6 ||
        cfg.smoothIterations < 0 || cfg.smoothIterations > 10 || !std::isfinite(cfg.snapshotYaw) ||
        !std::isfinite(cfg.snapshotPitch))
        throw std::runtime_error("Invalid surface/preview settings");
    if (!std::isfinite(cfg.surfaceMargin) || cfg.surfaceMargin < 0 || cfg.surfaceMargin > 2 ||
        !std::isfinite(cfg.frontPhotoPitch) || std::abs(cfg.frontPhotoPitch) > 30 ||
        !std::isfinite(cfg.backPhotoPitch) || std::abs(cfg.backPhotoPitch) > 30)
        throw std::runtime_error("Invalid continuous surface or photo projection settings");
    return cfg;
}

int main(int argc, char **argv)
{
    try
    {
        const Config cfg = parseArgs(argc, argv);
        if (cfg.selfTest)
        {
            runSelfTests();
            return 0;
        }
        fs::create_directories(cfg.outputDir);
        if (!cfg.previewPly.empty())
        {
            if (cfg.headless)
                throw std::runtime_error("--preview-ply needs viewer or --snapshot mode");
            auto loadedMesh = loadSmoothPreview(cfg.previewPly, cfg);
            int viewerArgc = 1;
            showViewer(viewerArgc, argv, loadedMesh.first, std::move(loadedMesh.second), cfg);
            return 0;
        }
        std::unique_ptr<DepthModel> model;
        std::array<View, 5> loaded;
        for (int i = 0; i < 5; ++i)
            loaded[i] = loadView((ViewId)i, cfg, model);
        model.reset(); // Keep only detached float depth; release ORT buffers before meshing/rendering.
        std::vector<View> views = alignViews(loaded);
        const Grid g = makeGrid(views, cfg);
        std::cout << "Grid " << g.x << 'x' << g.y << 'x' << g.z << std::endl;
        for (auto &v : views)
            normalizeDepth(v, cfg);
        saveDebug(views, cfg);
        std::vector<Projection> projections;
        for (const auto &v : views)
            projections.push_back(buildProjection(v, g, cfg));
        saveProjectionDebug(projections, cfg);
        Volume volume = reconstruct(projections, g, cfg);
        prepareTextures(views);
        Mesh mesh = buildSmoothMesh(volume, cfg, &views, &projections);
        if (mesh.boundaryEdges || mesh.nonmanifoldEdges || mesh.inconsistentOrientedEdges)
            throw std::runtime_error(
                "Extracted surface failed topology checks: " + std::to_string(mesh.boundaryEdges) +
                " open edges, " + std::to_string(mesh.nonmanifoldEdges) + " nonmanifold edges, " +
                std::to_string(mesh.inconsistentOrientedEdges) + " inconsistent oriented edges");
        colorMesh(mesh, g, cfg);
        exportResult(volume, mesh, projections, cfg);
        const auto occupied = std::count(volume.occupied.begin(), volume.occupied.end(), (uint8_t)1);
        std::cout << "Voxel Build Done: " << occupied << " voxels, " << mesh.triangles.size()
                  << " smooth triangles. Export: " << (cfg.outputDir / "surface.ply").string() << std::endl;
        if (!occupied)
            throw std::runtime_error(
                "Empty reconstruction; inspect masks/states and compare --silhouette-only");
        if (!cfg.headless)
        {
            // FreeGLUT must not receive application-specific CLI arguments.
            int viewerArgc = 1;
            showViewer(viewerArgc, argv, g, std::move(mesh), cfg);
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "V5 error: " << error.what() << std::endl;
        return 1;
    }
}
