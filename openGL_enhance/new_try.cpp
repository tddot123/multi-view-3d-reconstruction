#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

////////////////////////////////////////////////////////
// 图像
////////////////////////////////////////////////////////

Mat frontImg, backImg;
Mat leftImg, rightImg;
Mat topImg;

Mat frontImgRaw, backImgRaw;
Mat leftImgRaw, rightImgRaw;
Mat topImgRaw;

Mat maskFront, maskBack;
Mat maskLeft, maskRight;
Mat maskTop;

Rect boxFront, boxBack;
Rect boxLeft, boxRight;
Rect boxTop;

Mat depthFront;
Mat depthBack;
Mat depthLeft;
Mat depthRight;
Mat depthTop;

// raw float depth (shared scale before normalization)
Mat depthFrontF;
Mat depthBackF;
Mat depthLeftF;
Mat depthRightF;
Mat depthTopF;

bool hasRealBack = false;
bool hasRealRight = false;

////////////////////////////////////////////////////////
// ⭐比例 GRID（只新增）
////////////////////////////////////////////////////////

//const int BASE_GRID = 128;

int GRID_X;
int GRID_Y;
int GRID_Z;

static vector<uint8_t> voxelOcc;   // 0/1 occupancy (final after carving)
static vector<Vec3b> voxelCol;     // per-voxel color
static vector<uint8_t> voxelVote;  // supporting view count (0..255)
static vector<uint8_t> voxelSrc;   // bitmask: which view(s) wrote this voxel
static vector<float> voxelSupport; // accumulated support weight
static vector<float> voxelReject;  // accumulated weighted rejection

// per-view scratch before soft carving (v4 full-ray fill)
static vector<uint8_t> voxelPossible;
static vector<Vec3b> voxelPossibleCol;
static bool carveFirstView = true;
static int carvedViewCount = 0;

// depth fine-tune before voxelization
static const int DEPTH_CELL_MIN_FG = 2;
static const int DEPTH_CELL_GOOD_FG = 4;
static const int DEPTH_CELL_MAX_SPREAD = 65;
static const int DEPTH_CELL_SOFT_SPREAD = 35;
static const int DEPTH_MEDIAN_COLOR_BAND = 4;

// selective soft carving on v4 solid visual hull
static const float GRADIENT_HIGH_THRESHOLD = 12.0f;
static const float GRADIENT_LOW_THRESHOLD = 5.0f;
static const float REJECT_WEIGHT_FLAT = 1.0f;
static const float REJECT_WEIGHT_EDGE = 0.22f;
static const int TOL_UCHAR_FLAT = 2;
static const int TOL_UCHAR_EDGE = 8;
static const float SURVIVAL_REJECT_RATIO = 1.15f;
static const float CONTINUITY_MIN_SUPPORT = 2.0f;
static const int CONTINUITY_MIN_NEIGHBORS_26 = 8;
static const int ISOLATE_MAX_NEIGHBORS_26 = 2;

float cubeSize = 2.0f;

enum class ViewId
{
    Front,
    Back,
    Left,
    Right,
    Top,
};

static const uint8_t SRC_FRONT = 1 << 0;
static const uint8_t SRC_BACK = 1 << 1;
static const uint8_t SRC_LEFT = 1 << 2;
static const uint8_t SRC_RIGHT = 1 << 3;
static const uint8_t SRC_TOP = 1 << 4;

static inline int voxelIndex(int x, int y, int z)
{
    return (x * GRID_Y + y) * GRID_Z + z;
}

static inline bool inBounds(int x, int y, int z)
{
    return (x >= 0 && y >= 0 && z >= 0 &&
        x < GRID_X && y < GRID_Y && z < GRID_Z);
}

static void buildView(ViewId id);
static void safeSetColor(int x, int y, int z, Vec3b c, uint8_t srcBit);
static void resetPossibleVolume();
static void markPossibleVoxel(int x, int y, int z, const Vec3b& c);
static void applySoftVoxelCarving(ViewId id, uint8_t srcBit,
    const vector<float>& cellRejectWeight, int aSize, int bSize);
static void fillVolumeZ(int vx, int vy, int zBound, bool fromMinZ, int tolVox, const Vec3b& c);
static void fillVolumeX(int vy, int vz, int xBound, bool fromMinX, int tolVox, const Vec3b& c);
static void fillVolumeY(int vx, int vz, int yBound, bool fromMinY, int tolVox, const Vec3b& c);
static void finalizeSelectiveSoftCarving();
struct DepthSample;
struct ProjCellMeta;
static bool resolveStableProjCell(
    const vector<DepthSample>& samples,
    ProjCellMeta& outMeta);
static float computeProjCellGradient(
    const vector<vector<DepthSample>>& cells,
    int aSize, int bSize, int a, int b, uchar centerDepth);
static float rejectWeightFromGradient(float gradient);
static int adaptiveTolVoxels(float gradient, int gridDim);
static int projCellIndex(ViewId id, int x, int y, int z, int bSize);
static float survivalRejectRatio(int votes);
static void applyContinuity26();
static void pruneIsolatedVoxels26();
static int countOccupiedNeighbors26(int x, int y, int z, const vector<uint8_t>& occ);
void finalizeVoxelByVote(unsigned char minVote);
static int mapLeftDepthToX(uchar u);
static int mapRightDepthToX(uchar u);
static int mapTopDepthToY(uchar u);

////////////////////////////////////////////////////////
// Camera（完全保留）
////////////////////////////////////////////////////////

float angleX = 20;
float angleY = 30;
float distanceZ = 1200;

int lastX, lastY;
bool isDragging = false;

////////////////////////////////////////////////////////
// removeWhite（完全保留）
////////////////////////////////////////////////////////

Mat extractForegroundCrop(const Mat& src, Mat& outMask, Rect& outBox)
{
    if (src.empty()) return src;

    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    Mat mask;

    inRange(
        hsv,
        Scalar(0, 0, 160),
        Scalar(180, 80, 255),
        mask);

    bitwise_not(mask, mask);

    Mat kernel =
        getStructuringElement(
            MORPH_ELLIPSE,
            Size(9, 9));

    morphologyEx(
        mask,
        mask,
        MORPH_CLOSE,
        kernel);

    morphologyEx(
        mask,
        mask,
        MORPH_OPEN,
        kernel);

    vector<vector<Point>> contours;

    findContours(
        mask,
        contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        outMask = Mat::ones(src.size(), CV_8U) * 255;
        outBox = Rect(0, 0, src.cols, src.rows);
        return src;
    }

    int maxIndex = 0;
    double maxArea = 0;

    for (int i = 0; i < contours.size(); i++)
    {
        double area =
            contourArea(contours[i]);

        if (area > maxArea)
        {
            maxArea = area;
            maxIndex = i;
        }
    }

    Mat fullMask = Mat::zeros(src.size(), CV_8U);
    drawContours(
        fullMask,
        contours,
        maxIndex,
        Scalar(255),
        FILLED);

    outBox = boundingRect(contours[maxIndex]);
    outMask = fullMask(outBox).clone();
    return src(outBox).clone();
}

static Mat cropDepthByBox(const Mat& depth, const Rect& box)
{
    if (depth.empty()) return Mat();
    Rect valid = box & Rect(0, 0, depth.cols, depth.rows);
    if (valid.width <= 0 || valid.height <= 0)
        return depth.clone();
    return depth(valid).clone();
}

////////////////////////////////////////////////////////
// alignImages（只增加 GRID 比例）
////////////////////////////////////////////////////////

void alignImages()
{
    frontImg =
        extractForegroundCrop(frontImgRaw, maskFront, boxFront);

    leftImg =
        extractForegroundCrop(leftImgRaw, maskLeft, boxLeft);

    topImg =
        extractForegroundCrop(topImgRaw, maskTop, boxTop);

    if (!backImgRaw.empty())
        backImg =
        extractForegroundCrop(backImgRaw, maskBack, boxBack);

    if (!rightImgRaw.empty())
        rightImg =
        extractForegroundCrop(rightImgRaw, maskRight, boxRight);

    int W = frontImg.cols;
    int H = frontImg.rows;

    float scale =
        (float)H / leftImg.rows;

    resize(
        leftImg,
        leftImg,
        Size(),
        scale,
        scale);

    resize(
        maskLeft,
        maskLeft,
        leftImg.size());
    flip(leftImg, leftImg, 1);
    flip(maskLeft, maskLeft, 1);
    int D = leftImg.cols;

    // 顶视图坐标系修正：先旋转再 resize，避免最终尺寸变成 D×W（长宽互换）
    rotate(topImg, topImg, ROTATE_90_COUNTERCLOCKWISE);
    rotate(maskTop, maskTop, ROTATE_90_COUNTERCLOCKWISE);

    resize(
        topImg,
        topImg,
        Size(W, D));

    resize(
        maskTop,
        maskTop,
        topImg.size());

    if (backImg.empty())
    {
        flip(frontImg, backImg, 1);
        flip(maskFront, maskBack, 1);
        boxBack = boxFront;
    }

    if (rightImg.empty())
    {
        // 左视图已做过一次左右翻转，这里直接拷贝给右视图，避免再次反向
        rightImg = leftImg.clone();
        maskRight = maskLeft.clone();
        boxRight = boxLeft;
    }

    resize(backImg, backImg, Size(W, H));
    resize(maskBack, maskBack, backImg.size());

    resize(rightImg, rightImg, Size(D, H));
    resize(maskRight, maskRight, rightImg.size());

    //////////////////////////////////////////////////////
    // ⭐新增：比例 GRID 计算（唯一核心新增）
    //////////////////////////////////////////////////////

    //////////////////////////////////////////////////////
// ⭐真正按图分辨率 + 自动缩放（最终推荐）
//////////////////////////////////////////////////////

    const int MAX_GRID = 256;
    // 体素数据结构已改为动态分配，可直接调大 MAX_GRID 提升精度

    int maxDim =
        max({ W,H,D });

    float nscale =
        min(1.0f,
            (float)MAX_GRID /
            maxDim);

    GRID_X = int(W * nscale);
    GRID_Y = int(H * nscale);
    GRID_Z = int(D * nscale);

    // 防止极端情况下某轴变成 0
    GRID_X = max(1, GRID_X);
    GRID_Y = max(1, GRID_Y);
    GRID_Z = max(1, GRID_Z);

    cout << endl;
    cout << "=== Image Geometry ===" << endl;
    cout << "W = " << W << endl;
    cout << "H = " << H << endl;
    cout << "D = " << D << endl;

    cout << endl;
    cout << "=== Voxel Resolution ===" << endl;
    cout << "GX = " << GRID_X << endl;
    cout << "GY = " << GRID_Y << endl;
    cout << "GZ = " << GRID_Z << endl;
}

////////////////////////////////////////////////////////
// Depth（完全保留）
////////////////////////////////////////////////////////

Mat runDepthAnything(Mat& inputImg)
{
    if (inputImg.empty())
        return Mat();

    static Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "Depth");

    static Ort::SessionOptions opts;
    static bool optsInit = false;
    if (!optsInit)
    {
        opts.SetIntraOpNumThreads(1);
        optsInit = true;
    }

    static Ort::Session session(
        env,
        L"model/depth_anything_v2_large.onnx",
        opts);

    Mat img;

    resize(inputImg, img, Size(384, 384));

    cvtColor(img, img, COLOR_BGR2RGB);

    img.convertTo(img, CV_32F, 1.0 / 255.0);

    vector<float> tensor(
        1 * 3 * 384 * 384);

    int idx = 0;

    for (int c = 0; c < 3; c++)
        for (int y = 0; y < 384; y++)
            for (int x = 0; x < 384; x++)
                tensor[idx++] =
                img.at<Vec3f>(y, x)[c];

    array<int64_t, 4> shape =
    { 1,3,384,384 };

    auto mem =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);

    auto inputTensor =
        Ort::Value::CreateTensor<float>(
            mem,
            tensor.data(),
            tensor.size(),
            shape.data(),
            shape.size());

    const char* inNames[] = { "pixel_values" };
    const char* outNames[] = { "predicted_depth" };

    auto output =
        session.Run(
            Ort::RunOptions{ nullptr },
            inNames,
            &inputTensor,
            1,
            outNames,
            1);

    float* data =
        output[0]
        .GetTensorMutableData<float>();

    auto outShape =
        output[0]
        .GetTensorTypeAndShapeInfo()
        .GetShape();

    Mat depth(
        (int)outShape[1],
        (int)outShape[2],
        CV_32F,
        data);

    Mat safeDepth = depth.clone(); // detach from ORT buffer

    // return raw float depth resized to original image size (global normalization happens later)
    resize(
        safeDepth,
        safeDepth,
        inputImg.size());

    return safeDepth;
}

static void normalizeDepthsByAxis(
    const Mat& dFrontF, const Mat& mFront,
    const Mat& dBackF, const Mat& mBack,
    const Mat& dLeftF, const Mat& mLeft,
    const Mat& dRightF, const Mat& mRight,
    const Mat& dTopF, const Mat& mTop,
    Mat& outFront8U, Mat& outBack8U, Mat& outLeft8U, Mat& outRight8U, Mat& outTop8U)
{
    // 不同视角的 depth 表示的是不同轴（Z/X/Y），不能混在一个 gmin/gmax 里统一归一化
    // - front/back: Z 轴
    // - left/right: X 轴
    // - top: Y 轴

    auto computeMinMax = [&](const Mat& dA, const Mat& mA, const Mat& dB, const Mat& mB, float& mn, float& mx)
        {
            mn = FLT_MAX;
            mx = -FLT_MAX;
            auto scan = [&](const Mat& d, const Mat& m)
                {
                    if (d.empty() || m.empty()) return;
                    for (int y = 0; y < d.rows; y++)
                        for (int x = 0; x < d.cols; x++)
                        {
                            if (m.at<uchar>(y, x) == 0) continue;
                            float v = d.at<float>(y, x);
                            if (!std::isfinite(v)) continue;
                            mn = min(mn, v);
                            mx = max(mx, v);
                        }
                };
            scan(dA, mA);
            scan(dB, mB);
        };

    auto toU8Range = [&](const Mat& d, float mn, float mx, Mat& out)
        {
            if (d.empty()) { out = Mat(); return; }
            if (!(mx > mn))
            {
                Mat tmp8;
                normalize(d, tmp8, 0, 255, NORM_MINMAX);
                tmp8.convertTo(out, CV_8U);
                return;
            }

            Mat tmp(d.size(), CV_8U);
            const float scale = 255.0f / (mx - mn);
            for (int y = 0; y < d.rows; y++)
                for (int x = 0; x < d.cols; x++)
                {
                    float v = d.at<float>(y, x);
                    if (!std::isfinite(v))
                        tmp.at<uchar>(y, x) = 0;
                    else
                    {
                        float n = (v - mn) * scale;
                        n = max(0.0f, min(255.0f, n));
                        tmp.at<uchar>(y, x) = (uchar)(n);
                    }
                }
            out = tmp;
        };

    // Z group
    float zMin, zMax;
    computeMinMax(dFrontF, mFront, dBackF, mBack, zMin, zMax);
    toU8Range(dFrontF, zMin, zMax, outFront8U);
    toU8Range(dBackF, zMin, zMax, outBack8U);

    // X group
    float xMin, xMax;
    computeMinMax(dLeftF, mLeft, dRightF, mRight, xMin, xMax);
    toU8Range(dLeftF, xMin, xMax, outLeft8U);
    toU8Range(dRightF, xMin, xMax, outRight8U);

    // Y group (top only)
    float yMin = FLT_MAX, yMax = -FLT_MAX;
    computeMinMax(dTopF, mTop, Mat(), Mat(), yMin, yMax);
    toU8Range(dTopF, yMin, yMax, outTop8U);
}

static void saveDepthDebug(const char* filename, const Mat& depth8U)
{
    if (depth8U.empty()) return;
    imwrite(filename, depth8U);
}

static std::string makeDepthDebugPath(const std::string& inputImagePath, const char* dirName)
{
    // Avoid std::filesystem (requires C++17); do simple path/stem parse.
    std::string name = inputImagePath;
    const size_t slash = name.find_last_of("/\\");
    if (slash != std::string::npos)
        name = name.substr(slash + 1);

    std::string stem = name;
    const size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos)
        stem = stem.substr(0, dot);

    return std::string("output/") + stem + "_depth_" + std::string(dirName) + ".png";
}

struct MaskedDepthRange
{
    float mn = FLT_MAX;
    float mx = -FLT_MAX;
    bool valid = false;
};

static MaskedDepthRange computeMaskedDepthRange(const Mat& d, const Mat& m)
{
    MaskedDepthRange r;
    if (d.empty() || m.empty() || d.size() != m.size())
        return r;

    for (int y = 0; y < d.rows; y++)
        for (int x = 0; x < d.cols; x++)
        {
            if (m.at<uchar>(y, x) == 0) continue;
            float v = d.at<float>(y, x);
            if (!std::isfinite(v)) continue;
            r.mn = min(r.mn, v);
            r.mx = max(r.mx, v);
            r.valid = true;
        }
    return r;
}

static void normalizeDepthPerView(const Mat& dF, const Mat& m, Mat& out8U)
{
    out8U = Mat();
    if (dF.empty()) return;

    const MaskedDepthRange r = computeMaskedDepthRange(dF, m);
    if (!r.valid)
    {
        Mat tmp8;
        normalize(dF, tmp8, 0, 255, NORM_MINMAX);
        tmp8.convertTo(out8U, CV_8U);
        return;
    }

    const float span = max(r.mx - r.mn, 1e-6f);
    out8U = Mat::zeros(dF.size(), CV_8U);
    for (int y = 0; y < dF.rows; y++)
        for (int x = 0; x < dF.cols; x++)
        {
            if (!m.empty() && m.at<uchar>(y, x) == 0) continue;
            float v = dF.at<float>(y, x);
            if (!std::isfinite(v)) continue;
            const float n = (v - r.mn) / span;
            const int u = (int)(n * 254.0f + 1.5f);
            out8U.at<uchar>(y, x) = (uchar)max(1, min(255, u));
        }
}

static void normalizeDepthsIndependent(
    const Mat& dFrontF, const Mat& mFront,
    const Mat& dBackF, const Mat& mBack,
    const Mat& dLeftF, const Mat& mLeft,
    const Mat& dRightF, const Mat& mRight,
    const Mat& dTopF, const Mat& mTop,
    Mat& outFront8U, Mat& outBack8U, Mat& outLeft8U, Mat& outRight8U, Mat& outTop8U)
{
    normalizeDepthPerView(dFrontF, mFront, outFront8U);
    normalizeDepthPerView(dBackF, mBack, outBack8U);
    normalizeDepthPerView(dLeftF, mLeft, outLeft8U);
    normalizeDepthPerView(dRightF, mRight, outRight8U);
    normalizeDepthPerView(dTopF, mTop, outTop8U);
}

// 同一条 (x,y) 射线上 front+back depth 等比放大，使管壁在 Z 方向尽量铺满
static void scaleFrontBackDepthPair(Mat& front8U, Mat& back8U, const Mat& mFront, const Mat& mBack)
{
    if (front8U.empty() || back8U.empty()) return;

    vector<int> pairSums;
    pairSums.reserve(4096);
    for (int y = 0; y < front8U.rows; y++)
        for (int x = 0; x < front8U.cols; x++)
        {
            if (!mFront.empty() && mFront.at<uchar>(y, x) == 0) continue;
            if (!mBack.empty() && mBack.at<uchar>(y, x) == 0) continue;
            const uchar uf = front8U.at<uchar>(y, x);
            const uchar ub = back8U.at<uchar>(y, x);
            if (uf == 0 || ub == 0) continue;
            pairSums.push_back((int)uf + (int)ub);
        }

    if (pairSums.empty()) return;

    sort(pairSums.begin(), pairSums.end());
    const size_t iP95 = min(pairSums.size() - 1, (size_t)(pairSums.size() * 0.95f));
    const int p95Sum = pairSums[iP95];
    const float targetSum = 510.0f;
    if (p95Sum <= 0 || p95Sum >= (int)targetSum) return;

    const float k = min(2.0f, targetSum / (float)p95Sum);
    cout << "  front/back pair scale: P95(uF+uB)=" << p95Sum << " k=" << k << endl;

    for (int y = 0; y < front8U.rows; y++)
        for (int x = 0; x < front8U.cols; x++)
        {
            if (!mFront.empty() && mFront.at<uchar>(y, x) == 0) continue;
            if (!mBack.empty() && mBack.at<uchar>(y, x) == 0) continue;
            uchar& uf = front8U.at<uchar>(y, x);
            uchar& ub = back8U.at<uchar>(y, x);
            if (uf == 0 || ub == 0) continue;
            uf = (uchar)max(1, min(255, (int)(uf * k + 0.5f)));
            ub = (uchar)max(1, min(255, (int)(ub * k + 0.5f)));
        }
}

static int mapFrontDepthToZ(uchar u)
{
    if (u == 0) return -1;
    return (int)(((float)u - 1.0f) * (GRID_Z - 1) / 254.0f + 0.5f);
}

static int mapBackDepthToZ(uchar u)
{
    if (u == 0) return -1;
    return (int)((GRID_Z - 1) - ((float)u - 1.0f) * (GRID_Z - 1) / 254.0f + 0.5f);
}

static int mapLeftDepthToX(uchar u)
{
    if (u == 0) return -1;
    return (int)(((float)u - 1.0f) * (GRID_X - 1) / 254.0f + 0.5f);
}

static int mapRightDepthToX(uchar u)
{
    if (u == 0) return -1;
    return (int)((GRID_X - 1) - ((float)u - 1.0f) * (GRID_X - 1) / 254.0f + 0.5f);
}

static int mapTopDepthToY(uchar u)
{
    if (u == 0) return -1;
    return (int)(((float)u - 1.0f) * (GRID_Y - 1) / 254.0f + 0.5f);
}

////////////////////////////////////////////////////////
// v4 solid full-ray carving + selective soft constraints
////////////////////////////////////////////////////////

struct DepthSample
{
    uchar d = 0;
    Vec3b c;
};

struct ProjCellMeta
{
    uchar depth = 0;
    Vec3b color;
    int confidence = 0;
    int spread = 0;
    float gradient = 0.f;
};

static float rejectWeightFromGradient(float gradient)
{
    if (gradient <= GRADIENT_LOW_THRESHOLD)
        return REJECT_WEIGHT_FLAT;
    if (gradient >= GRADIENT_HIGH_THRESHOLD)
        return REJECT_WEIGHT_EDGE;
    const float t = (gradient - GRADIENT_LOW_THRESHOLD) /
        (GRADIENT_HIGH_THRESHOLD - GRADIENT_LOW_THRESHOLD);
    return REJECT_WEIGHT_FLAT - t * (REJECT_WEIGHT_FLAT - REJECT_WEIGHT_EDGE);
}

static int adaptiveTolVoxels(float gradient, int gridDim)
{
    const int flatTol = std::max(1, (int)((float)TOL_UCHAR_FLAT * (gridDim - 1) / 254.0f + 0.5f));
    const int edgeTol = std::max(flatTol + 1, (int)((float)TOL_UCHAR_EDGE * (gridDim - 1) / 254.0f + 0.5f));
    if (gradient <= GRADIENT_LOW_THRESHOLD)
        return flatTol;
    if (gradient >= GRADIENT_HIGH_THRESHOLD)
        return edgeTol;
    const float t = (gradient - GRADIENT_LOW_THRESHOLD) /
        (GRADIENT_HIGH_THRESHOLD - GRADIENT_LOW_THRESHOLD);
    return (int)(flatTol + t * (edgeTol - flatTol) + 0.5f);
}

static float survivalRejectRatio(int votes)
{
    if (votes >= 4) return SURVIVAL_REJECT_RATIO * 0.72f;
    if (votes == 3) return SURVIVAL_REJECT_RATIO * 0.85f;
    if (votes == 2) return SURVIVAL_REJECT_RATIO;
    return SURVIVAL_REJECT_RATIO * 1.25f;
}

static int projCellIndex(ViewId id, int x, int y, int z, int bSize)
{
    switch (id)
    {
    case ViewId::Front:
    case ViewId::Back:
        return x * bSize + y;
    case ViewId::Left:
    case ViewId::Right:
        return z * bSize + y;
    case ViewId::Top:
        return x * bSize + z;
    default:
        return -1;
    }
}

static uchar medianDepthOfSamples(const vector<DepthSample>& samples)
{
    vector<uchar> depths;
    depths.reserve(samples.size());
    for (const auto& s : samples)
        depths.push_back(s.d);
    sort(depths.begin(), depths.end());
    return depths[depths.size() / 2];
}

static float computeProjCellGradient(
    const vector<vector<DepthSample>>& cells,
    int aSize, int bSize, int a, int b, uchar centerDepth)
{
    float maxGrad = 0.f;
    const int da[] = { -1, 1, 0, 0 };
    const int db[] = { 0, 0, -1, 1 };
    for (int k = 0; k < 4; k++)
    {
        const int na = a + da[k];
        const int nb = b + db[k];
        if (na < 0 || nb < 0 || na >= aSize || nb >= bSize)
            continue;
        const vector<DepthSample>& ns = cells[(size_t)(na * bSize + nb)];
        if (ns.empty())
            continue;
        const uchar nd = medianDepthOfSamples(ns);
        maxGrad = std::max(maxGrad, (float)abs((int)nd - (int)centerDepth));
    }
    return maxGrad;
}

static bool resolveStableProjCell(
    const vector<DepthSample>& samples,
    ProjCellMeta& outMeta)
{
    const int n = (int)samples.size();
    if (n < DEPTH_CELL_MIN_FG)
        return false;

    vector<uchar> depths;
    depths.reserve((size_t)n);
    for (const auto& s : samples)
        depths.push_back(s.d);
    sort(depths.begin(), depths.end());

    const int spread = (int)depths.back() - (int)depths.front();
    if (spread > DEPTH_CELL_MAX_SPREAD)
        return false;

    outMeta.depth = depths[n / 2];
    outMeta.spread = spread;

    int bSum = 0, gSum = 0, rSum = 0, hit = 0;
    for (const auto& s : samples)
    {
        if (abs((int)s.d - (int)outMeta.depth) > DEPTH_MEDIAN_COLOR_BAND)
            continue;
        bSum += (int)s.c[0];
        gSum += (int)s.c[1];
        rSum += (int)s.c[2];
        hit++;
    }
    if (hit <= 0)
        outMeta.color = samples[n / 2].c;
    else
        outMeta.color = Vec3b((uchar)(bSum / hit), (uchar)(gSum / hit), (uchar)(rSum / hit));

    outMeta.confidence = 255;
    if (spread > DEPTH_CELL_SOFT_SPREAD)
        outMeta.confidence = 140;
    if (n < DEPTH_CELL_GOOD_FG)
        outMeta.confidence = min(outMeta.confidence, 160);
    return true;
}

static void resetPossibleVolume()
{
    fill(voxelPossible.begin(), voxelPossible.end(), 0);
}

static void markPossibleVoxel(int x, int y, int z, const Vec3b& c)
{
    if (!inBounds(x, y, z)) return;
    const int idx = voxelIndex(x, y, z);
    if (!voxelPossible[idx])
    {
        voxelPossible[idx] = 1;
        voxelPossibleCol[idx] = c;
    }
    else
    {
        const Vec3b old = voxelPossibleCol[idx];
        voxelPossibleCol[idx] = Vec3b(
            (uchar)(((int)old[0] + (int)c[0]) / 2),
            (uchar)(((int)old[1] + (int)c[1]) / 2),
            (uchar)(((int)old[2] + (int)c[2]) / 2)
        );
    }
}

static void fillVolumeZ(int vx, int vy, int zBound, bool fromMinZ, int tolVox, const Vec3b& c)
{
    zBound = min(GRID_Z - 1, max(0, zBound));
    if (fromMinZ)
        zBound = min(GRID_Z - 1, zBound + tolVox);
    else
        zBound = max(0, zBound - tolVox);

    if (fromMinZ)
    {
        for (int z = 0; z <= zBound; z++)
            markPossibleVoxel(vx, vy, z, c);
    }
    else
    {
        for (int z = zBound; z < GRID_Z; z++)
            markPossibleVoxel(vx, vy, z, c);
    }
}

static void fillVolumeX(int vy, int vz, int xBound, bool fromMinX, int tolVox, const Vec3b& c)
{
    xBound = min(GRID_X - 1, max(0, xBound));
    if (fromMinX)
        xBound = min(GRID_X - 1, xBound + tolVox);
    else
        xBound = max(0, xBound - tolVox);

    if (fromMinX)
    {
        for (int x = 0; x <= xBound; x++)
            markPossibleVoxel(x, vy, vz, c);
    }
    else
    {
        for (int x = xBound; x < GRID_X; x++)
            markPossibleVoxel(x, vy, vz, c);
    }
}

static void fillVolumeY(int vx, int vz, int yBound, bool fromMinY, int tolVox, const Vec3b& c)
{
    yBound = min(GRID_Y - 1, max(0, yBound));
    if (fromMinY)
        yBound = min(GRID_Y - 1, yBound + tolVox);
    else
        yBound = max(0, yBound - tolVox);

    if (fromMinY)
    {
        for (int y = 0; y <= yBound; y++)
            markPossibleVoxel(vx, y, vz, c);
    }
    else
    {
        for (int y = yBound; y < GRID_Y; y++)
            markPossibleVoxel(vx, y, vz, c);
    }
}

static void applySoftVoxelCarving(ViewId id, uint8_t srcBit,
    const vector<float>& cellRejectWeight, int aSize, int bSize)
{
    const size_t n = voxelOcc.size();
    if (carveFirstView)
    {
        for (size_t i = 0; i < n; i++)
        {
            voxelOcc[i] = voxelPossible[i];
            if (voxelOcc[i])
            {
                voxelCol[i] = voxelPossibleCol[i];
                voxelSrc[i] = srcBit;
                voxelVote[i] = 1;
                voxelSupport[i] = 1.0f;
                voxelReject[i] = 0.0f;
            }
        }
        carveFirstView = false;
        return;
    }

    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                const int idx = voxelIndex(x, y, z);
                const int pidx = projCellIndex(id, x, y, z, bSize);
                const float rejW = (pidx >= 0 && pidx < (int)cellRejectWeight.size())
                    ? cellRejectWeight[(size_t)pidx] : REJECT_WEIGHT_FLAT;

                if (voxelPossible[idx])
                {
                    if (!voxelOcc[idx])
                        continue;

                    if (!(voxelSrc[idx] & srcBit))
                    {
                        if (voxelVote[idx] < 255)
                            voxelVote[idx]++;
                        voxelSrc[idx] |= srcBit;
                    }
                    voxelSupport[idx] += 1.0f;

                    const Vec3b neu = voxelPossibleCol[idx];
                    const Vec3b old = voxelCol[idx];
                    voxelCol[idx] = Vec3b(
                        (uchar)(((int)old[0] + (int)neu[0]) / 2),
                        (uchar)(((int)old[1] + (int)neu[1]) / 2),
                        (uchar)(((int)old[2] + (int)neu[2]) / 2)
                    );
                }
                else if (voxelOcc[idx])
                {
                    voxelReject[idx] += rejW;
                }
            }
}

static int countOccupiedNeighbors26(int x, int y, int z, const vector<uint8_t>& occ)
{
    int count = 0;
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
            {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                const int nx = x + dx, ny = y + dy, nz = z + dz;
                if (!inBounds(nx, ny, nz))
                    continue;
                if (occ[voxelIndex(nx, ny, nz)])
                    count++;
            }
    return count;
}

static void applyContinuity26()
{
    vector<uint8_t> restored = voxelOcc;

    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                const int idx = voxelIndex(x, y, z);
                if (voxelOcc[idx])
                    continue;
                if (voxelSupport[idx] < CONTINUITY_MIN_SUPPORT)
                    continue;

                const int neighbors = countOccupiedNeighbors26(x, y, z, voxelOcc);
                if (neighbors >= CONTINUITY_MIN_NEIGHBORS_26)
                    restored[idx] = 1;
            }

    voxelOcc = restored;
}

static void pruneIsolatedVoxels26()
{
    vector<uint8_t> pruned = voxelOcc;

    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                const int idx = voxelIndex(x, y, z);
                if (!voxelOcc[idx])
                    continue;

                const int neighbors = countOccupiedNeighbors26(x, y, z, voxelOcc);
                if (neighbors <= ISOLATE_MAX_NEIGHBORS_26 && voxelVote[idx] < 3)
                    pruned[idx] = 0;
            }

    voxelOcc = pruned;
}

static void finalizeSelectiveSoftCarving()
{
    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                const int idx = voxelIndex(x, y, z);
                if (!voxelOcc[idx])
                    continue;

                const float support = voxelSupport[idx];
                const float reject = voxelReject[idx];
                const int votes = (int)voxelVote[idx];
                const float ratio = survivalRejectRatio(votes);

                bool keep = support > 0.f && support >= reject * ratio;
                if (!keep && votes >= 3 && support >= reject * ratio * 0.8f)
                    keep = true;

                voxelOcc[idx] = keep ? 1 : 0;
            }

    applyContinuity26();
    pruneIsolatedVoxels26();
}

////////////////////////////////////////////////////////
// voxel
////////////////////////////////////////////////////////

void initVoxel()
{
    const size_t n = (size_t)GRID_X * (size_t)GRID_Y * (size_t)GRID_Z;
    voxelOcc.assign(n, 0);
    voxelCol.assign(n, Vec3b(0, 0, 0));
    voxelVote.assign(n, 0);
    voxelSrc.assign(n, 0);
    voxelSupport.assign(n, 0.f);
    voxelReject.assign(n, 0.f);
    voxelPossible.assign(n, 0);
    voxelPossibleCol.assign(n, Vec3b(0, 0, 0));
    carveFirstView = true;
    carvedViewCount = 0;
}

void safeSetColor(int x, int y, int z, Vec3b c, uint8_t srcBit)
{
    if (!inBounds(x, y, z)) return;
    const int idx = voxelIndex(x, y, z);

    if (!voxelOcc[idx])
    {
        voxelOcc[idx] = 1;
        voxelCol[idx] = c;
    }
    else
    {
        Vec3b old = voxelCol[idx];
        voxelCol[idx] = Vec3b(
            (uchar)(((int)old[0] + (int)c[0]) / 2),
            (uchar)(((int)old[1] + (int)c[1]) / 2),
            (uchar)(((int)old[2] + (int)c[2]) / 2)
        );
    }

    if (voxelVote[idx] < 255)
        voxelVote[idx]++;

    voxelSrc[idx] |= srcBit;
}

void finalizeVoxelByVote(unsigned char minVote)
{
    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                const int idx = voxelIndex(x, y, z);
                if (voxelOcc[idx] && voxelVote[idx] < minVote)
                    voxelOcc[idx] = 0;
            }
}

////////////////////////////////////////////////////////
// buildView: v4 full-ray fill + selective soft carving
////////////////////////////////////////////////////////

static void buildView(ViewId id)
{
    const Mat* depth = nullptr;
    const Mat* img = nullptr;
    const Mat* mask = nullptr;

    int aSize = 0;
    int bSize = 0;

    switch (id)
    {
    case ViewId::Front:
        depth = &depthFront; img = &frontImg; mask = &maskFront;
        aSize = GRID_X; bSize = GRID_Y;
        break;
    case ViewId::Back:
        depth = &depthBack; img = &backImg; mask = &maskBack;
        aSize = GRID_X; bSize = GRID_Y;
        break;
    case ViewId::Left:
        depth = &depthLeft; img = &leftImg; mask = &maskLeft;
        aSize = GRID_Z; bSize = GRID_Y;
        break;
    case ViewId::Right:
        depth = &depthRight; img = &rightImg; mask = &maskRight;
        aSize = GRID_Z; bSize = GRID_Y;
        break;
    case ViewId::Top:
        depth = &depthTop; img = &topImg; mask = &maskTop;
        aSize = GRID_X; bSize = GRID_Z;
        break;
    }

    if (!depth || depth->empty() || !img || img->empty() || !mask || mask->empty())
        return;

    const int w = depth->cols;
    const int h = depth->rows;

    const int projN = aSize * bSize;
    vector<vector<DepthSample>> cells((size_t)projN);

    // Step 1: mask-only samples into projection cells (no background averaging)
    for (int py = 0; py < h; py++)
        for (int px = 0; px < w; px++)
        {
            if (mask->at<uchar>(py, px) == 0)
                continue;

            const uchar d = depth->at<uchar>(py, px);
            if (d == 0)
                continue;

            int a = 0;
            int b = 0;
            switch (id)
            {
            case ViewId::Front:
            case ViewId::Back:
                a = px * GRID_X / w;
                b = (h - py - 1) * GRID_Y / h;
                break;
            case ViewId::Left:
            case ViewId::Right:
                a = px * GRID_Z / w;
                b = (h - py - 1) * GRID_Y / h;
                break;
            case ViewId::Top:
                a = px * GRID_X / w;
                b = py * GRID_Z / h;
                break;
            }

            if (a < 0 || b < 0 || a >= aSize || b >= bSize)
                continue;

            const int pidx = a * bSize + b;
            DepthSample s;
            s.d = d;
            s.c = img->at<Vec3b>(py, px);
            cells[(size_t)pidx].push_back(s);
        }

    uint8_t srcBit = 0;
    switch (id)
    {
    case ViewId::Front: srcBit = SRC_FRONT; break;
    case ViewId::Back:  srcBit = SRC_BACK; break;
    case ViewId::Left:  srcBit = SRC_LEFT; break;
    case ViewId::Right: srcBit = SRC_RIGHT; break;
    case ViewId::Top:   srcBit = SRC_TOP; break;
    }

    resetPossibleVolume();

    int filledCells = 0;
    int skippedCells = 0;

    vector<float> cellRejectWeight((size_t)projN, REJECT_WEIGHT_FLAT);

    // Step 2: median depth + edge-aware full-ray fill
    for (int a = 0; a < aSize; a++)
        for (int b = 0; b < bSize; b++)
        {
            const int pidx = a * bSize + b;
            const vector<DepthSample>& samples = cells[(size_t)pidx];
            if (samples.empty())
                continue;

            ProjCellMeta meta;
            if (!resolveStableProjCell(samples, meta))
            {
                skippedCells++;
                continue;
            }

            meta.gradient = computeProjCellGradient(cells, aSize, bSize, a, b, meta.depth);
            cellRejectWeight[(size_t)pidx] = rejectWeightFromGradient(meta.gradient);

            switch (id)
            {
            case ViewId::Front:
            {
                const int z = mapFrontDepthToZ(meta.depth);
                if (z < 0) continue;
                const int tol = adaptiveTolVoxels(meta.gradient, GRID_Z);
                fillVolumeZ(a, b, z, true, tol, meta.color);
                filledCells++;
                break;
            }
            case ViewId::Back:
            {
                const int z = mapBackDepthToZ(meta.depth);
                if (z < 0) continue;
                const int tol = adaptiveTolVoxels(meta.gradient, GRID_Z);
                fillVolumeZ(a, b, z, false, tol, meta.color);
                filledCells++;
                break;
            }
            case ViewId::Left:
            {
                const int x = mapLeftDepthToX(meta.depth);
                if (x < 0) continue;
                const int tol = adaptiveTolVoxels(meta.gradient, GRID_X);
                fillVolumeX(b, a, x, true, tol, meta.color);
                filledCells++;
                break;
            }
            case ViewId::Right:
            {
                const int x = mapRightDepthToX(meta.depth);
                if (x < 0) continue;
                const int tol = adaptiveTolVoxels(meta.gradient, GRID_X);
                fillVolumeX(b, a, x, false, tol, meta.color);
                filledCells++;
                break;
            }
            case ViewId::Top:
            {
                const int y = mapTopDepthToY(meta.depth);
                if (y < 0) continue;
                const int tol = adaptiveTolVoxels(meta.gradient, GRID_Y);
                fillVolumeY(a, b, y, true, tol, meta.color);
                filledCells++;
                break;
            }
            default:
                break;
            }
        }

    if (filledCells <= 0)
        return;

    applySoftVoxelCarving(id, srcBit, cellRejectWeight, aSize, bSize);
    carvedViewCount++;

    const char* viewName = "?";
    switch (id)
    {
    case ViewId::Front: viewName = "Front"; break;
    case ViewId::Back:  viewName = "Back"; break;
    case ViewId::Left:  viewName = "Left"; break;
    case ViewId::Right: viewName = "Right"; break;
    case ViewId::Top:   viewName = "Top"; break;
    }
    cout << "  carve " << viewName << ": filled " << filledCells
        << ", skipped(discontinuity/fg) " << skippedCells << endl;
}

////////////////////////////////////////////////////////
// OpenGL（完全保留）
////////////////////////////////////////////////////////

void drawCube(float x, float y, float z, float size) {
    float s = size / 2.0f;

    glBegin(GL_QUADS);

    // Front
    glVertex3f(x - s, y - s, z + s);
    glVertex3f(x + s, y - s, z + s);
    glVertex3f(x + s, y + s, z + s);
    glVertex3f(x - s, y + s, z + s);

    // Back
    glVertex3f(x - s, y - s, z - s);
    glVertex3f(x + s, y - s, z - s);
    glVertex3f(x + s, y + s, z - s);
    glVertex3f(x - s, y + s, z - s);

    // Left
    glVertex3f(x - s, y - s, z - s);
    glVertex3f(x - s, y - s, z + s);
    glVertex3f(x - s, y + s, z + s);
    glVertex3f(x - s, y + s, z - s);

    // Right
    glVertex3f(x + s, y - s, z - s);
    glVertex3f(x + s, y - s, z + s);
    glVertex3f(x + s, y + s, z + s);
    glVertex3f(x + s, y + s, z - s);

    // Top
    glVertex3f(x - s, y + s, z - s);
    glVertex3f(x - s, y + s, z + s);
    glVertex3f(x + s, y + s, z + s);
    glVertex3f(x + s, y + s, z - s);

    // Bottom
    glVertex3f(x - s, y - s, z - s);
    glVertex3f(x - s, y - s, z + s);
    glVertex3f(x + s, y - s, z + s);
    glVertex3f(x + s, y - s, z - s);

    glEnd();
}

void drawVoxel()
{
    float offX = GRID_X / 2.0f;
    float offY = GRID_Y / 2.0f;
    //float offZ = GRID_Z / 2.0f;
    float offZ = 0;

    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                const int idx = voxelIndex(x, y, z);
                if (!voxelOcc[idx]) continue;

                Vec3b c = voxelCol[idx];

                glColor3f(
                    c[2] / 255.0f,
                    c[1] / 255.0f,
                    c[0] / 255.0f);

                drawCube(
                    (x - offX) * cubeSize,
                    (y - offY) * cubeSize,
                    (z - offZ) * cubeSize,
                    cubeSize);
            }
}

void display()
{
    glClear(
        GL_COLOR_BUFFER_BIT |
        GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();

    glTranslatef(
        0, 0,
        -distanceZ);

    glRotatef(angleX, 1, 0, 0);
    glRotatef(angleY, 0, 1, 0);

    drawVoxel();

    glutSwapBuffers();
}

void reshape(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(
        60,
        (float)w / h,
        1,
        5000);

    glMatrixMode(GL_MODELVIEW);
}

////////////////////////////////////////////////////////
// 鼠标（完全保留）
////////////////////////////////////////////////////////

void mouse(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON)
    {
        isDragging =
            (state == GLUT_DOWN);

        lastX = x;
        lastY = y;
    }

    if (button == 3)
        distanceZ *= 0.9f;

    if (button == 4)
        distanceZ *= 1.1f;

    glutPostRedisplay();
}

void motion(int x, int y)
{
    if (isDragging)
    {
        angleY += (x - lastX) * 0.5f;
        angleX += (y - lastY) * 0.5f;

        lastX = x;
        lastY = y;

        glutPostRedisplay();
    }
}

////////////////////////////////////////////////////////
// MAIN（完全保留）
////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    const std::string frontPath = "input/front.png";
    const std::string backPath = "input/back.png";
    const std::string leftPath = "input/left.png";
    const std::string rightPath = "input/right.png";
    const std::string topPath = "input/top.png";

    frontImgRaw = imread(frontPath);
    backImgRaw = imread(backPath);
    hasRealBack = !backImgRaw.empty();

    leftImgRaw = imread(leftPath);
    rightImgRaw = imread(rightPath);
    hasRealRight = !rightImgRaw.empty();

    topImgRaw = imread(topPath);

    if (frontImgRaw.empty() ||
        leftImgRaw.empty() ||
        topImgRaw.empty())
    {
        cout << "Need front left top" << endl;
        return -1;
    }

    if (!hasRealBack)
        flip(frontImgRaw, backImgRaw, 1);

    if (!hasRealRight)
        flip(leftImgRaw, rightImgRaw, 1);

    alignImages();

    depthFrontF = cropDepthByBox(runDepthAnything(frontImgRaw), boxFront);
    depthBackF = cropDepthByBox(runDepthAnything(backImgRaw), boxBack);
    depthLeftF = cropDepthByBox(runDepthAnything(leftImgRaw), boxLeft);
    depthRightF = cropDepthByBox(runDepthAnything(rightImgRaw), boxRight);
    depthTopF = cropDepthByBox(runDepthAnything(topImgRaw), boxTop);

    if (!depthFrontF.empty() && depthFrontF.size() != frontImg.size())
        resize(depthFrontF, depthFrontF, frontImg.size());

    if (!depthLeftF.empty())
    {
        float scale = (float)frontImg.rows / depthLeftF.rows;
        resize(depthLeftF, depthLeftF, Size(), scale, scale);
        flip(depthLeftF, depthLeftF, 1);
        if (depthLeftF.size() != leftImg.size())
            resize(depthLeftF, depthLeftF, leftImg.size());
    }

    if (!depthTopF.empty())
    {
        rotate(depthTopF, depthTopF, ROTATE_90_COUNTERCLOCKWISE);
        if (depthTopF.size() != topImg.size())
            resize(depthTopF, depthTopF, topImg.size());
    }

    if (!depthBackF.empty() && depthBackF.size() != backImg.size())
        resize(depthBackF, depthBackF, backImg.size());

    if (!hasRealBack)
        flip(depthFrontF, depthBackF, 1);
    else if (!depthBackF.empty())
        flip(depthBackF, depthBackF, 1);

    if (!depthRightF.empty())
    {
        if (!hasRealRight)
        {
            depthRightF = depthLeftF.clone();
        }
        else if (depthRightF.size() != rightImg.size())
        {
            resize(depthRightF, depthRightF, rightImg.size());
        }
    }

    // 各视图独立 depth 归一化 → 配对缩放 → 体素雕刻
    cout << "=== Depth: per-view normalize + pair scale ===" << endl;
    normalizeDepthsIndependent(
        depthFrontF, maskFront,
        depthBackF, maskBack,
        depthLeftF, maskLeft,
        depthRightF, maskRight,
        depthTopF, maskTop,
        depthFront, depthBack, depthLeft, depthRight, depthTop);

    scaleFrontBackDepthPair(depthFront, depthBack, maskFront, maskBack);

    // 保存深度图到 output/（便于观察）
    const std::string outFront = makeDepthDebugPath(frontPath, "front");
    const std::string outBack = makeDepthDebugPath(backPath, "back");
    const std::string outLeft = makeDepthDebugPath(leftPath, "left");
    const std::string outRight = makeDepthDebugPath(rightPath, "right");
    const std::string outTop = makeDepthDebugPath(topPath, "top");

    saveDepthDebug(outFront.c_str(), depthFront);
    saveDepthDebug(outBack.c_str(), depthBack);
    saveDepthDebug(outLeft.c_str(), depthLeft);
    saveDepthDebug(outRight.c_str(), depthRight);
    saveDepthDebug(outTop.c_str(), depthTop);

    initVoxel();

    cout << "=== v4 selective soft carve ===" << endl;
    buildView(ViewId::Front);
    buildView(ViewId::Back);
    buildView(ViewId::Left);
    buildView(ViewId::Right);
    buildView(ViewId::Top);

    if (carvedViewCount >= 2)
    {
        finalizeSelectiveSoftCarving();
        finalizeVoxelByVote(2);
    }

    cout << "Voxel Build Done" << endl;

    glutInit(&argc, argv);

    glutInitDisplayMode(
        GLUT_DOUBLE |
        GLUT_RGB |
        GLUT_DEPTH);

    glutInitWindowSize(900, 700);

    glutCreateWindow(
        "Multi View Depth Model");

    glEnable(GL_DEPTH_TEST);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);

    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    glutIdleFunc(display);

    glutMainLoop();

    return 0;
}