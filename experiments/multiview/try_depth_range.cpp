/*#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <climits>
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

static vector<uint8_t> voxelOcc;   // 0/1 occupancy
static vector<Vec3b> voxelCol;     // per-voxel color
static vector<uint8_t> voxelVote;  // vote count (0..255)
static vector<uint8_t> voxelSrc;   // bitmask: which view(s) wrote this voxel

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

static void alignDepthSingleToGrid(const Mat& dF, const Mat& m, int gridExtent, Mat& out8U);
static void buildView(ViewId id);
static void removeIsolatedVoxels(int minNeighborCount);
static void keepConnectedComponentsBySize(int minKeepSize, float keepRatioOfLargest);

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

    // 真实 back/right：对立相机需水平翻转，使图左↔世界右与对侧视角一致
    // （合成 back/right 已在上面 flip 或 clone 左图，不再翻）
    if (hasRealBack)
    {
        flip(backImg, backImg, 1);
        flip(maskBack, maskBack, 1);
    }
    if (hasRealRight)
    {
        flip(rightImg, rightImg, 1);
        flip(maskRight, maskRight, 1);
    }

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
        L"models/depth_anything_v2_large.onnx",
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
        outShape[1],
        outShape[2],
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

// 去掉 mask 内深度两端离群（噪声常拉高 gmin/gmax），Pass2 用此算 n
static MaskedDepthRange computeMaskedDepthRangeTrimmed(
    const Mat& d, const Mat& m, float lowQ = 0.05f, float highQ = 0.95f)
{
    vector<float> vals;
    vals.reserve((size_t)d.rows * (size_t)d.cols / 4);

    for (int y = 0; y < d.rows; y++)
        for (int x = 0; x < d.cols; x++)
        {
            if (m.at<uchar>(y, x) == 0) continue;
            float v = d.at<float>(y, x);
            if (!std::isfinite(v)) continue;
            vals.push_back(v);
        }

    MaskedDepthRange r;
    if (vals.empty()) return r;

    sort(vals.begin(), vals.end());
    const size_t n = vals.size();
    const size_t iLo = (size_t)min((float)(n - 1), floor(lowQ * (float)(n - 1)));
    const size_t iHi = (size_t)min((float)(n - 1), ceil(highQ * (float)(n - 1)));
    r.mn = vals[iLo];
    r.mx = vals[max(iLo, iHi)];
    if (r.mx <= r.mn)
    {
        r.mn = vals.front();
        r.mx = vals.back();
    }
    r.valid = true;
    return r;
}

// uchar=0 保留为「无 depth」；有效 coord∈[0,maxIdx] 映射到 uchar∈[1,255]，避免最浅点被当成空
static const uchar DEPTH_U8_EMPTY = 0;
static const uchar DEPTH_U8_MIN = 1;
static const uchar DEPTH_U8_MAX = 255;

static uchar encodeDepthU8LowSide(float coord, int gridExtent)
{
    const float maxIdx = (float)max(0, gridExtent - 1);
    coord = max(0.0f, min(maxIdx, coord));
    const float u = (float)DEPTH_U8_MIN + coord * (float)(DEPTH_U8_MAX - DEPTH_U8_MIN) / max(1.0f, maxIdx);
    return (uchar)max((float)DEPTH_U8_MIN, min((float)DEPTH_U8_MAX, u));
}

// 高侧视角（back/right）：coord 越小 uchar 越大
static uchar encodeDepthU8HighSide(float coord, int gridExtent)
{
    const float maxIdx = (float)max(0, gridExtent - 1);
    coord = max(0.0f, min(maxIdx, coord));
    const float u = (float)DEPTH_U8_MIN + (maxIdx - coord) * (float)(DEPTH_U8_MAX - DEPTH_U8_MIN) / max(1.0f, maxIdx);
    return (uchar)max((float)DEPTH_U8_MIN, min((float)DEPTH_U8_MAX, u));
}

// 第一遍：各视角独立拉满 0–255，仅用于粗建模（与 front 正确版本一致）
static void normalizeDepthsProvisional(
    const Mat& dFrontF, const Mat& mFront,
    const Mat& dBackF, const Mat& mBack,
    const Mat& dLeftF, const Mat& mLeft,
    const Mat& dRightF, const Mat& mRight,
    const Mat& dTopF, const Mat& mTop,
    Mat& outFront8U, Mat& outBack8U, Mat& outLeft8U, Mat& outRight8U, Mat& outTop8U)
{
    alignDepthSingleToGrid(dFrontF, mFront, GRID_Z, outFront8U);
    alignDepthSingleToGrid(dBackF, mBack, GRID_Z, outBack8U);
    alignDepthSingleToGrid(dLeftF, mLeft, GRID_X, outLeft8U);
    alignDepthSingleToGrid(dRightF, mRight, GRID_X, outRight8U);
    alignDepthSingleToGrid(dTopF, mTop, GRID_Y, outTop8U);
}

// 去噪体素在厚度轴上的 [mn,mx]；用分位数去掉贴边噪声（如 mx=255）
static bool measureVoxelExtentPercentile(
    uint8_t srcBit, int thicknessAxis, int& outMn, int& outMx,
    float loQ = 0.05f, float hiQ = 0.95f)
{
    vector<int> coords;
    coords.reserve(4096);

    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                const int idx = voxelIndex(x, y, z);
                if (!voxelOcc[idx]) continue;
                if ((voxelSrc[idx] & srcBit) == 0) continue;

                int c = x;
                if (thicknessAxis == 1) c = y;
                else if (thicknessAxis == 2) c = z;
                coords.push_back(c);
            }

    if (coords.empty()) return false;

    sort(coords.begin(), coords.end());
    const size_t n = coords.size();
    const size_t iLo = (size_t)min((float)(n - 1), floor(loQ * (float)(n - 1)));
    const size_t iHi = (size_t)min((float)(n - 1), ceil(hiQ * (float)(n - 1)));
    outMn = coords[iLo];
    outMx = coords[max(iLo, iHi)];
    if (outMx <= outMn) { outMn = coords.front(); outMx = coords.back(); }
    return true;
}

static bool measureVoxelExtent(uint8_t srcBit, int thicknessAxis, int& outMn, int& outMx)
{
    return measureVoxelExtentPercentile(srcBit, thicknessAxis, outMn, outMx);
}

static float measureVoxelThickness(uint8_t srcBit, int thicknessAxis)
{
    int mn = 0, mx = 0;
    if (!measureVoxelExtent(srcBit, thicknessAxis, mn, mx))
        return -1.0f;
    return (float)(mx - mn + 1);
}

// meet 接缝比例：去噪 depth 壳层厚度
static void thicknessForMeetPair(
    float voxelTLow, float voxelTHigh,
    float& outLow, float& outHigh)
{
    outLow = (voxelTLow > 0.0f) ? voxelTLow : 1.0f;
    outHigh = (voxelTHigh > 0.0f) ? voxelTHigh : 1.0f;

    const float avg = (outLow + outHigh) * 0.5f;
    const float relDiff = fabs(outLow - outHigh) / max(avg, 1e-6f);

    if (relDiff < 0.35f)
    {
        cout << "    meet pair denoised depth L=" << outLow << " R=" << outHigh
            << " -> symmetric " << avg << endl;
        outLow = outHigh = avg;
    }
    else
    {
        cout << "    meet pair denoised depth L=" << outLow << " R=" << outHigh << endl;
    }
}

// Pass1 uchar -> 体素轴坐标（与 buildView 解码一致）；u=0 表示无效
static float provToAxisCoord(uchar u, int gridExtent, bool highSideCodec)
{
    if (u == DEPTH_U8_EMPTY)
        return -1.0f;

    const float maxIdx = (float)max(0, gridExtent - 1);
    const float t = ((float)u - (float)DEPTH_U8_MIN) * maxIdx / (float)(DEPTH_U8_MAX - DEPTH_U8_MIN);
    if (highSideCodec)
        return maxIdx - t;
    return t;
}

static void encodeAxisCoord(float coord, int gridExtent, bool highSideCodec, uchar& outU)
{
    if (highSideCodec)
        outU = encodeDepthU8HighSide(coord, gridExtent);
    else
        outU = encodeDepthU8LowSide(coord, gridExtent);
}

// 仿射伸缩；输入=去噪 depth；extent 外像素已在 extract 阶段置 0
static void stretchProvView(
    const Mat& prov8U, const Mat& m,
    int gridExtent,
    int voxMn, int voxMx,
    float slabLo, float slabHi,
    bool highSideCodec,
    const char* dbgTag,
    Mat& out8U)
{
    out8U = Mat();
    if (prov8U.empty() || voxMx <= voxMn) return;

    const float maxIdx = (float)max(0, gridExtent - 1);
    const float srcMn = (float)voxMn;
    const float srcMx = (float)voxMx;
    const float srcSpan = max(1e-4f, srcMx - srcMn);
    const float dstSpan = slabHi - slabLo;
    const float gridMid = 0.5f * maxIdx;
    const float centroid = 0.5f * (srcMn + srcMx);
    // front/left 壳在轴高端需 invert；back/right 壳在轴低端也需 invert（内外侧才能 meet↔maxIdx）
    const bool invert = highSideCodec ? (centroid < gridMid) : (centroid > gridMid);

    int kept = 0;
    int skipped = 0;

    out8U = Mat::zeros(prov8U.size(), CV_8U);
    for (int y = 0; y < prov8U.rows; y++)
        for (int x = 0; x < prov8U.cols; x++)
        {
            if (!m.empty() && m.at<uchar>(y, x) == 0) continue;
            const uchar uIn = prov8U.at<uchar>(y, x);
            if (uIn == DEPTH_U8_EMPTY) continue;
            float coord = provToAxisCoord(uIn, gridExtent, highSideCodec);
            if (coord < 0.0f) continue;

            if (coord < srcMn || coord > srcMx)
            {
                // 略低于 P5 的最浅像素：钳到 srcMn 参与 stretch，避免起始段被 skip
                if (coord < srcMn)
                    coord = srcMn;
                else
                {
                    skipped++;
                    continue;
                }
            }

            float coordNew;
            if (invert)
                coordNew = slabHi - (coord - srcMn) * dstSpan / srcSpan;
            else
                coordNew = slabLo + (coord - srcMn) * dstSpan / srcSpan;

            coordNew = max(slabLo, min(slabHi, coordNew));
            coordNew = max(0.0f, min(maxIdx, coordNew));
            encodeAxisCoord(coordNew, gridExtent, highSideCodec, out8U.at<uchar>(y, x));
            kept++;
        }

    if (dbgTag)
        cout << "    " << dbgTag << " shell kept=" << kept << " skipped(outside denoised extent)=" << skipped << endl;
}

// 从 Pass1 粗 depth + 去噪体素 extent 提取壳层 → 去噪 depth（extent 外置 0）
static void extractDenoisedShellDepth(
    const Mat& pass1Depth8U, const Mat& mask,
    uint8_t srcBit, int thicknessAxis, bool highSideCodec,
    Mat& outDenoised8U)
{
    outDenoised8U = Mat();
    if (pass1Depth8U.empty()) return;

    int mn = 0, mx = 0;
    // loQ=0 保留最浅壳层；hiQ=0.95 去掉高端噪声（全范围会把 mx 拉到 255 导致 Pass2 过度压缩）
    if (!measureVoxelExtentPercentile(srcBit, thicknessAxis, mn, mx, 0.0f, 0.95f))
    {
        outDenoised8U = Mat::zeros(pass1Depth8U.size(), CV_8U);
        return;
    }

    const float srcMn = (float)mn;
    const float srcMx = (float)mx;
    const int gridExtent = (thicknessAxis == 2) ? GRID_Z : ((thicknessAxis == 0) ? GRID_X : GRID_Y);

    outDenoised8U = Mat::zeros(pass1Depth8U.size(), CV_8U);
    for (int y = 0; y < pass1Depth8U.rows; y++)
        for (int x = 0; x < pass1Depth8U.cols; x++)
        {
            if (!mask.empty() && mask.at<uchar>(y, x) == 0) continue;
            const uchar u = pass1Depth8U.at<uchar>(y, x);
            if (u == DEPTH_U8_EMPTY) continue;
            const float coord = provToAxisCoord(u, gridExtent, highSideCodec);
            if (coord < 0.0f) continue;
            if (coord > srcMx) continue;
            // 低于体素 extent 下限的最浅像素仍保留（起始段）
            outDenoised8U.at<uchar>(y, x) = u;
        }
}

// 去噪 depth 在轴上的 extent
// loQ=0：保留最浅点（起始部分）；hiQ=0.95：裁掉高端离群噪声，避免 stretch 被压扁
static bool measureDenoisedDepthExtent(
    const Mat& den8U, const Mat& mask,
    int gridExtent, bool highSideCodec,
    int& outMn, int& outMx,
    float loQ = 0.0f, float hiQ = 0.95f)
{
    vector<float> coords;
    if (den8U.empty()) return false;

    coords.reserve((size_t)den8U.rows * (size_t)den8U.cols / 8);
    for (int y = 0; y < den8U.rows; y++)
        for (int x = 0; x < den8U.cols; x++)
        {
            if (!mask.empty() && mask.at<uchar>(y, x) == 0) continue;
            const uchar u = den8U.at<uchar>(y, x);
            if (u == DEPTH_U8_EMPTY) continue;
            const float c = provToAxisCoord(u, gridExtent, highSideCodec);
            if (c < 0.0f) continue;
            coords.push_back(c);
        }

    if (coords.empty()) return false;

    sort(coords.begin(), coords.end());
    const size_t n = coords.size();
    const size_t iLo = (size_t)min((float)(n - 1), floor(loQ * (float)(n - 1)));
    const size_t iHi = (size_t)min((float)(n - 1), ceil(hiQ * (float)(n - 1)));
    outMn = (int)floor(coords[iLo]);
    outMx = (int)ceil(coords[max(iLo, iHi)]);
    if (outMx <= outMn)
    {
        outMn = (int)floor(coords.front());
        outMx = (int)ceil(coords.back());
    }
    return true;
}

static float measureDenoisedDepthThickness(
    const Mat& den8U, const Mat& mask,
    int gridExtent, bool highSideCodec)
{
    int mn = 0, mx = 0;
    // meet 比例仍用 P5–P95 厚度，不受 stretch 范围影响
    if (!measureDenoisedDepthExtent(den8U, mask, gridExtent, highSideCodec, mn, mx, 0.05f, 0.95f))
        return -1.0f;
    return (float)(mx - mn + 1);
}

static void stretchDepthPairFromDenoised(
    const Mat& denLow, const Mat& mLow,
    const Mat& denHigh, const Mat& mHigh,
    int gridExtent,
    bool highSideLow, bool highSideHigh,
    const char* axisName,
    Mat& outLow8U, Mat& outHigh8U)
{
    outLow8U = Mat();
    outHigh8U = Mat();

    const float tLow = measureDenoisedDepthThickness(denLow, mLow, gridExtent, highSideLow);
    const float tHigh = measureDenoisedDepthThickness(denHigh, mHigh, gridExtent, highSideHigh);
    if (tLow <= 0.0f && tHigh <= 0.0f) return;

    float tL = (tLow > 0.0f) ? tLow : tHigh;
    float tH = (tHigh > 0.0f) ? tHigh : tLow;

    float meetLow = 0.0f, meetHigh = 0.0f;
    thicknessForMeetPair(tL, tH, meetLow, meetHigh);
    (void)meetLow;

    const float total = meetLow + meetHigh;
    const float maxIdx = (float)max(0, gridExtent - 1);
    const float meet = maxIdx * (meetLow / total);

    int mnL = 0, mxL = 0, mnH = 0, mxH = 0;
    const bool hasL = measureDenoisedDepthExtent(denLow, mLow, gridExtent, highSideLow, mnL, mxL);
    const bool hasH = measureDenoisedDepthExtent(denHigh, mHigh, gridExtent, highSideHigh, mnH, mxH);

    const bool invertL = hasL && (0.5f * (float)(mnL + mxL)) > (0.5f * maxIdx);
    const bool invertH = hasH && (0.5f * (float)(mnH + mxH)) < (0.5f * maxIdx);

    cout << "  [" << axisName << "] meet=" << meet
        << " denoised-depth L[" << mnL << "," << mxL << "] H[" << mnH << "," << mxH << "]"
        << " invert L=" << (invertL ? 1 : 0) << " H=" << (invertH ? 1 : 0) << endl;

    if (hasL && mxL > mnL && !denLow.empty())
        stretchProvView(denLow, mLow, gridExtent, mnL, mxL, 0.0f, meet, highSideLow, axisName, outLow8U);

    if (hasH && mxH > mnH && !denHigh.empty())
    {
        const char* tagH = (axisName[0] == 'Z') ? "Z-back" : "X-right";
        stretchProvView(denHigh, mHigh, gridExtent, mnH, mxH, meet, maxIdx, highSideHigh, tagH, outHigh8U);
    }
}

static void denoiseVoxels()
{
    removeIsolatedVoxels(4);
    keepConnectedComponentsBySize(200, 0.03f);
}

// 单视角（top）：拉伸至整轴高度 GRID_Y
static void alignDepthSingleToGrid(
    const Mat& dF, const Mat& m,
    int gridExtent,
    Mat& out8U)
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
    const float maxIdx = (float)max(0, gridExtent - 1);

    out8U = Mat::zeros(dF.size(), CV_8U);
    for (int y = 0; y < dF.rows; y++)
        for (int x = 0; x < dF.cols; x++)
        {
            if (!m.empty() && m.at<uchar>(y, x) == 0) continue;
            float v = dF.at<float>(y, x);
            if (!std::isfinite(v)) continue;
            const float n = (v - r.mn) / span;
            const float coord = n * maxIdx;
            out8U.at<uchar>(y, x) = encodeDepthU8LowSide(coord, gridExtent);
        }
}

// Pass2：meet + 仿射伸缩；输入/范围/extent/厚度 全部来自去噪 depth
static void normalizeDepthsFromDenoisedShell(
    const Mat& denFront, const Mat& denBack,
    const Mat& denLeft, const Mat& denRight,
    const Mat& denTop,
    Mat& outFront8U, Mat& outBack8U, Mat& outLeft8U, Mat& outRight8U, Mat& outTop8U)
{
    cout << "=== Pass2: stretch denoised depth -> meet slab ===" << endl;

    const float vZF = measureDenoisedDepthThickness(denFront, maskFront, GRID_Z, false);
    const float vZB = measureDenoisedDepthThickness(denBack, maskBack, GRID_Z, true);
    const float vXL = measureDenoisedDepthThickness(denLeft, maskLeft, GRID_X, false);
    const float vXR = measureDenoisedDepthThickness(denRight, maskRight, GRID_X, true);
    cout << "  denoised depth span: Z front=" << vZF << " back=" << vZB
        << " X left=" << vXL << " right=" << vXR << endl;

    stretchDepthPairFromDenoised(
        denFront, maskFront, denBack, maskBack,
        GRID_Z, false, true, "Z",
        outFront8U, outBack8U);

    stretchDepthPairFromDenoised(
        denLeft, maskLeft, denRight, maskRight,
        GRID_X, false, true, "X",
        outLeft8U, outRight8U);

    int yMn = 0, yMx = 0;
    if (measureDenoisedDepthExtent(denTop, maskTop, GRID_Y, false, yMn, yMx) && !denTop.empty())
    {
        const float maxIdx = (float)max(0, GRID_Y - 1);
        stretchProvView(denTop, maskTop, GRID_Y, yMn, yMx, 0.0f, maxIdx, false, "Y-top", outTop8U);
    }

    cout << "=== Pass2 done ===" << endl;
}

// 粗建 + 去噪 + 按体素厚度重映射 + 精建
static void buildAllViewsForThicknessProbe()
{
    buildView(ViewId::Front);
    buildView(ViewId::Back);
    buildView(ViewId::Left);
    buildView(ViewId::Right);
    buildView(ViewId::Top);
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

    return std::string("outputs/") + stem + "_depth_" + std::string(dirName) + ".png";
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

void removeIsolatedVoxels(int minNeighborCount)
{
    vector<uint8_t> keep(voxelOcc.size(), 0);

    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                const int idx = voxelIndex(x, y, z);
                if (!voxelOcc[idx]) continue;

                int neighbors = 0;
                for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dz = -1; dz <= 1; dz++)
                        {
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            int nx = x + dx, ny = y + dy, nz = z + dz;
                            if (!inBounds(nx, ny, nz)) continue;
                            if (voxelOcc[voxelIndex(nx, ny, nz)]) neighbors++;
                        }

                if (neighbors >= minNeighborCount)
                    keep[idx] = 1;
            }

    voxelOcc.swap(keep);
}

static void keepConnectedComponentsBySize(int minKeepSize, float keepRatioOfLargest)
{
    vector<uint8_t> visited(voxelOcc.size(), 0);

    int bestSize = 0;

    // 6-neighborhood flood fill queue (store 1D indices)
    vector<int> q;
    q.reserve(4096);

    // First pass: find largest component size
    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                const int start = voxelIndex(x, y, z);
                if (!voxelOcc[start] || visited[start]) continue;

                q.clear();
                q.push_back(start);
                visited[start] = 1;

                int size = 0;
                for (size_t qi = 0; qi < q.size(); qi++)
                {
                    const int idx = q[qi];
                    size++;

                    const int cx = idx / (GRID_Y * GRID_Z);
                    const int rem = idx - cx * (GRID_Y * GRID_Z);
                    const int cy = rem / GRID_Z;
                    const int cz = rem - cy * GRID_Z;

                    for (int i = 0; i < 6; i++)
                    {
                        int xx = cx, yy = cy, zz = cz;
                        if (i == 0) xx--;
                        else if (i == 1) xx++;
                        else if (i == 2) yy--;
                        else if (i == 3) yy++;
                        else if (i == 4) zz--;
                        else zz++;

                        if (!inBounds(xx, yy, zz)) continue;
                        const int nidx = voxelIndex(xx, yy, zz);
                        if (!voxelOcc[nidx] || visited[nidx]) continue;
                        visited[nidx] = 1;
                        q.push_back(nidx);
                    }
                }

                if (size > bestSize)
                {
                    bestSize = size;
                }
            }

    if (bestSize <= 0)
        return;

    const int ratioKeep = (int)(bestSize * keepRatioOfLargest);
    const int threshold = max(1, max(minKeepSize, ratioKeep));

    vector<uint8_t> keep(voxelOcc.size(), 0);
    fill(visited.begin(), visited.end(), 0);

    // Second pass: keep any component whose size >= threshold
    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                const int start = voxelIndex(x, y, z);
                if (!voxelOcc[start] || visited[start]) continue;

                q.clear();
                q.push_back(start);
                visited[start] = 1;

                int size = 0;
                for (size_t qi = 0; qi < q.size(); qi++)
                {
                    const int idx = q[qi];
                    size++;

                    const int cx = idx / (GRID_Y * GRID_Z);
                    const int rem = idx - cx * (GRID_Y * GRID_Z);
                    const int cy = rem / GRID_Z;
                    const int cz = rem - cy * GRID_Z;

                    for (int i = 0; i < 6; i++)
                    {
                        int xx = cx, yy = cy, zz = cz;
                        if (i == 0) xx--;
                        else if (i == 1) xx++;
                        else if (i == 2) yy--;
                        else if (i == 3) yy++;
                        else if (i == 4) zz--;
                        else zz++;

                        if (!inBounds(xx, yy, zz)) continue;
                        const int nidx = voxelIndex(xx, yy, zz);
                        if (!voxelOcc[nidx] || visited[nidx]) continue;
                        visited[nidx] = 1;
                        q.push_back(nidx);
                    }
                }

                if (size >= threshold)
                {
                    for (size_t i = 0; i < q.size(); i++)
                        keep[q[i]] = 1;
                }
            }

    voxelOcc.swap(keep);
}

////////////////////////////////////////////////////////
// ⭐build 全部保留（只改 GRID→GRID_X/Y/Z）
////////////////////////////////////////////////////////

// buildFront/buildBack/buildLeft/buildRight/buildTop 已合并为 buildView(ViewId)

static void buildView(ViewId id)
{
    const Mat* depth = nullptr;
    const Mat* img = nullptr;
    const Mat* mask = nullptr;

    int aSize = 0;
    int bSize = 0;
    int thicknessAxis = 2; // 0->x,1->y,2->z
    const int thickness = 2;

    switch (id)
    {
    case ViewId::Front:
        depth = &depthFront; img = &frontImg; mask = &maskFront;
        aSize = GRID_X; bSize = GRID_Y; thicknessAxis = 2;
        break;
    case ViewId::Back:
        depth = &depthBack; img = &backImg; mask = &maskBack;
        aSize = GRID_X; bSize = GRID_Y; thicknessAxis = 2;
        break;
    case ViewId::Left:
        depth = &depthLeft; img = &leftImg; mask = &maskLeft;
        aSize = GRID_Z; bSize = GRID_Y; thicknessAxis = 0;
        break;
    case ViewId::Right:
        depth = &depthRight; img = &rightImg; mask = &maskRight;
        aSize = GRID_Z; bSize = GRID_Y; thicknessAxis = 0;
        break;
    case ViewId::Top:
        depth = &depthTop; img = &topImg; mask = &maskTop;
        aSize = GRID_X; bSize = GRID_Z; thicknessAxis = 1;
        break;
    }

    if (!depth || depth->empty() || !img || img->empty() || !mask || mask->empty())
        return;

    const int w = depth->cols;
    const int h = depth->rows;

    const int projN = aSize * bSize;
    vector<int> depthSum(projN, 0);
    vector<int> colorBSum(projN, 0);
    vector<int> colorGSum(projN, 0);
    vector<int> colorRSum(projN, 0);
    vector<int> hitCount(projN, 0);

    // 1) 聚合到投影网格 (a,b)
    for (int py = 0; py < h; py++)
        for (int px = 0; px < w; px++)
        {
            if (mask->at<uchar>(py, px) == 0)
                continue;

            const uchar d = depth->at<uchar>(py, px);
            if (d == DEPTH_U8_EMPTY)
                continue;

            const Vec3b c = img->at<Vec3b>(py, px);

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
                a = px * GRID_Z / w;            // z
                b = (h - py - 1) * GRID_Y / h;  // y
                break;
            case ViewId::Top:
                a = px * GRID_X / w;            // x
                b = py * GRID_Z / h;            // z
                break;
            }

            if (a < 0 || b < 0 || a >= aSize || b >= bSize)
                continue;

            const int pidx = a * bSize + b;
            depthSum[pidx] += (int)d;
            colorBSum[pidx] += (int)c[0];
            colorGSum[pidx] += (int)c[1];
            colorRSum[pidx] += (int)c[2];
            hitCount[pidx]++;
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

    // 2) 生成中心体素并沿厚度轴连续化
    for (int a = 0; a < aSize; a++)
        for (int b = 0; b < bSize; b++)
        {
            const int pidx = a * bSize + b;
            const int cnt = hitCount[pidx];
            if (cnt <= 0) continue;

            const int avgDepth = depthSum[pidx] / cnt;

            Vec3b avgColor(
                (uchar)(colorBSum[pidx] / cnt),
                (uchar)(colorGSum[pidx] / cnt),
                (uchar)(colorRSum[pidx] / cnt)
            );

            int vx = 0, vy = 0, vz = 0;

            switch (id)
            {
            case ViewId::Front:
                vx = a; vy = b;
                vz = (int)(provToAxisCoord((uchar)avgDepth, GRID_Z, false) + 0.5f);
                break;
            case ViewId::Back:
                vx = a; vy = b;
                vz = (int)(provToAxisCoord((uchar)avgDepth, GRID_Z, true) + 0.5f);
                break;
            case ViewId::Left:
                vz = a; vy = b;
                vx = (int)(provToAxisCoord((uchar)avgDepth, GRID_X, false) + 0.5f);
                break;
            case ViewId::Right:
                vz = a; vy = b;
                vx = (int)(provToAxisCoord((uchar)avgDepth, GRID_X, true) + 0.5f);
                break;
            case ViewId::Top:
                vx = a; vz = b;
                vy = (int)(provToAxisCoord((uchar)avgDepth, GRID_Y, false) + 0.5f);
                break;
            }

            vx = min(GRID_X - 1, max(0, vx));
            vy = min(GRID_Y - 1, max(0, vy));
            vz = min(GRID_Z - 1, max(0, vz));

            if (thicknessAxis == 0)
            {
                int x0 = max(0, vx - thickness);
                int x1 = min(GRID_X - 1, vx + thickness);
                for (int xx = x0; xx <= x1; xx++)
                    safeSetColor(xx, vy, vz, avgColor, srcBit);
            }
            else if (thicknessAxis == 1)
            {
                int y0 = max(0, vy - thickness);
                int y1 = min(GRID_Y - 1, vy + thickness);
                for (int yy = y0; yy <= y1; yy++)
                    safeSetColor(vx, yy, vz, avgColor, srcBit);
            }
            else
            {
                int z0 = max(0, vz - thickness);
                int z1 = min(GRID_Z - 1, vz + thickness);
                for (int zz = z0; zz <= z1; zz++)
                    safeSetColor(vx, vy, zz, avgColor, srcBit);
            }
        }
    removeIsolatedVoxels(4);
    keepConnectedComponentsBySize(200, 0.03f);
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
    const std::string frontPath = "data/samples/multiview/f08.png";
    const std::string backPath = "data/samples/multiview/b08.png";
    const std::string leftPath = "data/samples/multiview/l08.png";
    const std::string rightPath = "data/samples/multiview/r08.png";
    const std::string topPath = "data/samples/multiview/t08.png";

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
        else
        {
            if (depthRightF.size() != rightImg.size())
                resize(depthRightF, depthRightF, rightImg.size());
            flip(depthRightF, depthRightF, 1);
        }
    }

    // Pass1：原始 float depth → 粗 depth → 粗建模 → 去噪
    cout << "=== Pass1: raw depth -> coarse model -> denoise ===" << endl;
    normalizeDepthsProvisional(
        depthFrontF, maskFront,
        depthBackF, maskBack,
        depthLeftF, maskLeft,
        depthRightF, maskRight,
        depthTopF, maskTop,
        depthFront, depthBack, depthLeft, depthRight, depthTop);

    initVoxel();
    buildAllViewsForThicknessProbe();
    denoiseVoxels();

    // 得到去噪 depth：仅保留去噪体素壳层内的 Pass1 depth，extent 外为 0
    cout << "=== Extract denoised shell depth ===" << endl;
    Mat depthFrontDen, depthBackDen, depthLeftDen, depthRightDen, depthTopDen;
    extractDenoisedShellDepth(depthFront, maskFront, SRC_FRONT, 2, false, depthFrontDen);
    extractDenoisedShellDepth(depthBack, maskBack, SRC_BACK, 2, true, depthBackDen);
    extractDenoisedShellDepth(depthLeft, maskLeft, SRC_LEFT, 0, false, depthLeftDen);
    extractDenoisedShellDepth(depthRight, maskRight, SRC_RIGHT, 0, true, depthRightDen);
    extractDenoisedShellDepth(depthTop, maskTop, SRC_TOP, 1, false, depthTopDen);

    // Pass2：全部基于去噪 depth 做 meet 伸缩 → 规范化 depth
    normalizeDepthsFromDenoisedShell(
        depthFrontDen, depthBackDen,
        depthLeftDen, depthRightDen,
        depthTopDen,
        depthFront, depthBack, depthLeft, depthRight, depthTop);

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
    saveDepthDebug(makeDepthDebugPath(frontPath, "front_denoised").c_str(), depthFrontDen);
    saveDepthDebug(makeDepthDebugPath(backPath, "back_denoised").c_str(), depthBackDen);

    initVoxel();

    // Pass3：规范化 depth 精建 → 新模型
    buildView(ViewId::Front);
    buildView(ViewId::Back);
    //buildView(ViewId::Left);
    //buildView(ViewId::Right);
    //buildView(ViewId::Top);

    denoiseVoxels();

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
}*/