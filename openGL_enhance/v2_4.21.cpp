/*#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cstdint>

using namespace cv;
using namespace std;

////////////////////////////////////////////////////////
// 图像
////////////////////////////////////////////////////////

Mat frontImg, backImg;
Mat leftImg, rightImg;
Mat topImg;

Mat maskFront, maskBack;
Mat maskLeft, maskRight;
Mat maskTop;

Mat depthFront;
Mat depthBack;
Mat depthLeft;
Mat depthRight;
Mat depthTop;

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
static const uint8_t SRC_BACK  = 1 << 1;
static const uint8_t SRC_LEFT  = 1 << 2;
static const uint8_t SRC_RIGHT = 1 << 3;
static const uint8_t SRC_TOP   = 1 << 4;

static inline int voxelIndex(int x, int y, int z)
{
    return (x * GRID_Y + y) * GRID_Z + z;
}

static inline bool inBounds(int x, int y, int z)
{
    return (x >= 0 && y >= 0 && z >= 0 &&
        x < GRID_X && y < GRID_Y && z < GRID_Z);
}

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

Mat removeWhite(Mat src, Mat& outMask)
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
        outMask =
            Mat::ones(src.size(), CV_8U) * 255;

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

    Rect box =
        boundingRect(contours[maxIndex]);

    outMask =
        mask(box);

    return src(box);
}

////////////////////////////////////////////////////////
// alignImages（只增加 GRID 比例）
////////////////////////////////////////////////////////

void alignImages()
{
    frontImg =
        removeWhite(frontImg, maskFront);

    leftImg =
        removeWhite(leftImg, maskLeft);

    topImg =
        removeWhite(topImg, maskTop);

    if (!backImg.empty())
        backImg =
        removeWhite(backImg, maskBack);

    if (!rightImg.empty())
        rightImg =
        removeWhite(rightImg, maskRight);

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
    }

    if (rightImg.empty())
    {
        // 左视图已做过一次左右翻转，这里直接拷贝给右视图，避免再次反向
        rightImg = leftImg.clone();
        maskRight = maskLeft.clone();
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

    const int MAX_GRID = 1024;
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
    imwrite("debug_left_aligned.png", leftImg);
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
        outShape[1],
        outShape[2],
        CV_32F,
        data);

    Mat safeDepth =
        depth.clone();

    Mat depth8U;

    normalize(
        safeDepth,
        depth8U,
        0,
        255,
        NORM_MINMAX);

    depth8U.convertTo(
        depth8U,
        CV_8U);

    resize(
        depth8U,
        depth8U,
        inputImg.size());

    return depth8U.clone();
}

static void saveDepthDebug(const char* filename, const Mat& depth8U)
{
    if (depth8U.empty()) return;
    imwrite(filename, depth8U);
}

static Mat applyMaskBackground(const Mat& bgr, const Mat& mask, const Scalar& bgBgr)
{
    if (bgr.empty()) return Mat();
    if (mask.empty()) return bgr.clone();

    Mat out = Mat(bgr.size(), bgr.type(), bgBgr);
    bgr.copyTo(out, mask);
    return out;
}

static Mat maskDepthForDebug(const Mat& depth8U, const Mat& mask)
{
    if (depth8U.empty()) return Mat();
    if (mask.empty()) return depth8U.clone();
    Mat out = depth8U.clone();
    out.setTo(0, mask == 0);
    return out;
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

static void applySideBoundsFromLeftRight(int toleranceVoxel)
{
    // 从“已经建出来的 left/right 体素”统计 (y,z) 的边界，仅裁剪 TOP 贡献的越界体素
    // 重要：left/right 都存在时才做双边夹；只存在单边时只做单边约束，避免把 top 全裁掉
    // left 给出的是左表面（应取最小 x），right 给出右表面（应取最大 x）
    vector<int> xLeft((size_t)GRID_Y * (size_t)GRID_Z, GRID_X);      // left: min x
    vector<int> xRight((size_t)GRID_Y * (size_t)GRID_Z, -1);         // right: max x
    vector<uint8_t> hasLeft((size_t)GRID_Y * (size_t)GRID_Z, 0);
    vector<uint8_t> hasRight((size_t)GRID_Y * (size_t)GRID_Z, 0);

    auto yzIndex = [&](int y, int z) -> int { return y * GRID_Z + z; };

    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                const int idx = voxelIndex(x, y, z);
                if (!voxelOcc[idx]) continue;
                const uint8_t src = voxelSrc[idx];

                const int id = yzIndex(y, z);
                if (src & SRC_LEFT)
                {
                    if (!hasLeft[id])
                    {
                        xLeft[id] = x;
                        hasLeft[id] = 1;
                    }
                    else
                    {
                        xLeft[id] = min(xLeft[id], x);
                    }
                }
                if (src & SRC_RIGHT)
                {
                    if (!hasRight[id])
                    {
                        xRight[id] = x;
                        hasRight[id] = 1;
                    }
                    else
                    {
                        xRight[id] = max(xRight[id], x);
                    }
                }
            }

    for (int y = 0; y < GRID_Y; y++)
        for (int z = 0; z < GRID_Z; z++)
        {
            const int id = yzIndex(y, z);
            if (!hasLeft[id] && !hasRight[id]) continue;

            int lo = 0;
            int hi = GRID_X - 1;
            if (hasLeft[id])
                lo = max(lo, xLeft[id] - toleranceVoxel);
            if (hasRight[id])
                hi = min(hi, xRight[id] + toleranceVoxel);
            lo = max(0, lo);
            hi = min(GRID_X - 1, hi);
            if (lo > hi) continue;

            for (int x = 0; x < GRID_X; x++)
            {
                const int idx = voxelIndex(x, y, z);
                if (!voxelOcc[idx]) continue;

                // 只裁剪 top 造成的 overhang，不动 left/right 自己
                if ((voxelSrc[idx] & SRC_TOP) == 0) continue;
                if (x < lo || x > hi)
                {
                    voxelOcc[idx] = 0;
                    voxelSrc[idx] &= (uint8_t)~SRC_TOP;
                }
            }
        }
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

    // 2) 生成中心体素并沿厚度轴连续化
    uint8_t srcBit = 0;
    switch (id)
    {
    case ViewId::Front: srcBit = SRC_FRONT; break;
    case ViewId::Back:  srcBit = SRC_BACK; break;
    case ViewId::Left:  srcBit = SRC_LEFT; break;
    case ViewId::Right: srcBit = SRC_RIGHT; break;
    case ViewId::Top:   srcBit = SRC_TOP; break;
    }

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
                vz = avgDepth * GRID_Z / 255;
                break;
            case ViewId::Back:
                vx = a; vy = b;
                vz = GRID_Z - avgDepth * GRID_Z / 255 - 1;
                break;
            case ViewId::Left:
                vz = a; vy = b;
                vx = avgDepth * GRID_X / 255;
                break;
            case ViewId::Right:
                vz = a; vy = b;
                vx = GRID_X - avgDepth * GRID_X / 255 - 1;
                break;
            case ViewId::Top:
                vx = a; vz = b;
                {
                    int vy0 = avgDepth * GRID_Y / 255;
                    int yTop = GRID_Y - 1;
                    vy = yTop - (int)((yTop - vy0) * 1.5f);
                }
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
    frontImg = imread("input/f01.png");
    backImg = imread("input/b01.png");
    hasRealBack = !backImg.empty();

    leftImg = imread("input/l01.png");
    rightImg = imread("input/r01.png");
    hasRealRight = !rightImg.empty();

    topImg = imread("input/t01.png");

    if (frontImg.empty() ||
        leftImg.empty() ||
        topImg.empty())
    {
        cout << "Need front left top" << endl;
        return -1;
    }

    alignImages();

    // 用 mask 把背景替换为中性灰，避免侧视图大面积白背景导致 depth 退化成“灰条”
    const Scalar bg(127, 127, 127);
    Mat frontIn = applyMaskBackground(frontImg, maskFront, bg);
    Mat backIn = applyMaskBackground(backImg, maskBack, bg);
    Mat leftIn = applyMaskBackground(leftImg, maskLeft, bg);
    Mat rightIn = applyMaskBackground(rightImg, maskRight, bg);
    Mat topIn = applyMaskBackground(topImg, maskTop, bg);

    depthFront = runDepthAnything(frontIn);
    depthBack = runDepthAnything(backIn);
    depthLeft = runDepthAnything(leftIn);
    depthRight = runDepthAnything(rightIn);
    depthTop = runDepthAnything(topIn);

    // 保存5张深度图到当前目录（便于观察）
    // 保存 depth（背景置 0，便于观察真实前景）
    saveDepthDebug("output/depth_front.png", maskDepthForDebug(depthFront, maskFront));
    saveDepthDebug("output/depth_back.png", maskDepthForDebug(depthBack, maskBack));
    saveDepthDebug("output/depth_left.png", maskDepthForDebug(depthLeft, maskLeft));
    saveDepthDebug("output/depth_right.png", maskDepthForDebug(depthRight, maskRight));
    saveDepthDebug("output/depth_top.png", maskDepthForDebug(depthTop, maskTop));

    // 额外保存 mask 后的输入图，方便对照
    //imwrite("output/input_front_masked.png", frontIn);
    //imwrite("output/input_back_masked.png", backIn);
    //imwrite("output/input_left_masked.png", leftIn);
    //imwrite("output/input_right_masked.png", rightIn);
    //imwrite("output/input_top_masked.png", topIn);

    initVoxel();

    // 一个通用 build 函数，按视角调用
    buildView(ViewId::Front);
    buildView(ViewId::Back);
    buildView(ViewId::Left);
    buildView(ViewId::Right);
    buildView(ViewId::Top);

    // 用 left/right 建模后的边界约束裁剪 top 的越界体素（防止超过侧视位置）
    // 值越大越宽松；建议 1~3
    applySideBoundsFromLeftRight(3);

    // 清理孤立噪声点（值越大去噪越强）
    removeIsolatedVoxels(4);
    // 去掉漂浮“成团噪声”：保留足够大的连通块（避免误删手臂等小部件）
    keepConnectedComponentsBySize(200, 0.03f);

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