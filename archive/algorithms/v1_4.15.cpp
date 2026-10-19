/*#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>

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

bool voxel[128][128][128];
Vec3b voxelColor[128][128][128];
unsigned char voxelVote[128][128][128];

float cubeSize = 2.0f;

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
        Scalar(0, 0, 200),
        Scalar(180, 40, 255),
        mask);

    bitwise_not(mask, mask);

    Mat kernel =
        getStructuringElement(
            MORPH_ELLIPSE,
            Size(5, 5));

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

    resize(
        topImg,
        topImg,
        Size(W, D));

    resize(
        maskTop,
        maskTop,
        topImg.size());

    // 顶视图坐标系修正：逆时针旋转 90 度后再参与建模
    rotate(topImg, topImg, ROTATE_90_COUNTERCLOCKWISE);
    rotate(maskTop, maskTop, ROTATE_90_COUNTERCLOCKWISE);

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

    const int MAX_GRID = 128;
    // 体素数组是固定 128^3，统一按同一缩放因子保持 XYZ 原比例

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

////////////////////////////////////////////////////////
// voxel
////////////////////////////////////////////////////////

void initVoxel()
{
    memset(voxel, false, sizeof(voxel));
    memset(voxelVote, 0, sizeof(voxelVote));
}

void safeSetColor(int x, int y, int z, Vec3b c)
{
    if (x < 0 || y < 0 || z < 0) return;

    if (x >= GRID_X ||
        y >= GRID_Y ||
        z >= GRID_Z) return;

    if (!voxel[x][y][z])
    {
        voxel[x][y][z] = true;
        voxelColor[x][y][z] = c;
    }
    else
    {
        // 多视角命中同一体素时做简单均值，避免单视角颜色突变
        Vec3b old = voxelColor[x][y][z];
        voxelColor[x][y][z] = Vec3b(
            (uchar)(((int)old[0] + (int)c[0]) / 2),
            (uchar)(((int)old[1] + (int)c[1]) / 2),
            (uchar)(((int)old[2] + (int)c[2]) / 2)
        );
    }

    if (voxelVote[x][y][z] < 255)
        voxelVote[x][y][z]++;
}

void finalizeVoxelByVote(unsigned char minVote)
{
    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
                if (voxel[x][y][z] && voxelVote[x][y][z] < minVote)
                    voxel[x][y][z] = false;
}

void removeIsolatedVoxels(int minNeighborCount)
{
    static bool keep[128][128][128];
    memset(keep, false, sizeof(keep));

    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
            {
                if (!voxel[x][y][z]) continue;

                int neighbors = 0;
                for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dz = -1; dz <= 1; dz++)
                        {
                            if (dx == 0 && dy == 0 && dz == 0) continue;

                            int nx = x + dx;
                            int ny = y + dy;
                            int nz = z + dz;

                            if (nx < 0 || ny < 0 || nz < 0) continue;
                            if (nx >= GRID_X || ny >= GRID_Y || nz >= GRID_Z) continue;
                            if (voxel[nx][ny][nz]) neighbors++;
                        }

                if (neighbors >= minNeighborCount)
                    keep[x][y][z] = true;
            }

    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
                voxel[x][y][z] = keep[x][y][z];
}

////////////////////////////////////////////////////////
// ⭐build 全部保留（只改 GRID→GRID_X/Y/Z）
////////////////////////////////////////////////////////

void buildFront()
{
    int w = depthFront.cols;
    int h = depthFront.rows;

    static int depthSum[128][128];
    static int colorBSum[128][128];
    static int colorGSum[128][128];
    static int colorRSum[128][128];
    static int hitCount[128][128];

    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
        {
            depthSum[x][y] = 0;
            colorBSum[x][y] = 0;
            colorGSum[x][y] = 0;
            colorRSum[x][y] = 0;
            hitCount[x][y] = 0;
        }

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            if (maskFront.at<uchar>(y, x) == 0)
                continue;

            uchar d =
                depthFront.at<uchar>(y, x);

            int vx = x * GRID_X / w;
            int vy = (h - y - 1) * GRID_Y / h;

            Vec3b c =
                frontImg.at<Vec3b>(y, x);

            if (vx < 0 || vy < 0 ||
                vx >= GRID_X || vy >= GRID_Y)
                continue;

            depthSum[vx][vy] += (int)d;
            colorBSum[vx][vy] += (int)c[0];
            colorGSum[vx][vy] += (int)c[1];
            colorRSum[vx][vy] += (int)c[2];
            hitCount[vx][vy]++;
        }

    // 关键：同一前视投影位置(vx,vy)只生成一个中心深度(vz)
    // 并在z方向做小范围连续化，减少条纹和空洞
    const int frontThickness = 2; // 可调：0更薄，2~3更连续
    for (int vx = 0; vx < GRID_X; vx++)
        for (int vy = 0; vy < GRID_Y; vy++)
        {
            int cnt = hitCount[vx][vy];
            if (cnt <= 0) continue;

            int avgDepth = depthSum[vx][vy] / cnt;
            int vz = avgDepth * GRID_Z / 255;

            Vec3b avgColor(
                (uchar)(colorBSum[vx][vy] / cnt),
                (uchar)(colorGSum[vx][vy] / cnt),
                (uchar)(colorRSum[vx][vy] / cnt)
            );

            int z0 = max(0, vz - frontThickness);
            int z1 = min(GRID_Z - 1, vz + frontThickness);
            for (int zz = z0; zz <= z1; zz++)
                safeSetColor(vx, vy, zz, avgColor);
        }
}

void buildBack()
{
    int w = depthBack.cols;
    int h = depthBack.rows;

    static int depthSum[128][128];
    static int colorBSum[128][128];
    static int colorGSum[128][128];
    static int colorRSum[128][128];
    static int hitCount[128][128];

    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
        {
            depthSum[x][y] = 0;
            colorBSum[x][y] = 0;
            colorGSum[x][y] = 0;
            colorRSum[x][y] = 0;
            hitCount[x][y] = 0;
        }

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            if (maskBack.at<uchar>(y, x) == 0)
                continue;

            uchar d =
                depthBack.at<uchar>(y, x);

            int vx = x * GRID_X / w;
            int vy = (h - y - 1) * GRID_Y / h;

            Vec3b c =
                backImg.at<Vec3b>(y, x);

            if (vx < 0 || vy < 0 ||
                vx >= GRID_X || vy >= GRID_Y)
                continue;

            depthSum[vx][vy] += (int)d;
            colorBSum[vx][vy] += (int)c[0];
            colorGSum[vx][vy] += (int)c[1];
            colorRSum[vx][vy] += (int)c[2];
            hitCount[vx][vy]++;
        }

    const int backThickness = 2;
    for (int vx = 0; vx < GRID_X; vx++)
        for (int vy = 0; vy < GRID_Y; vy++)
        {
            int cnt = hitCount[vx][vy];
            if (cnt <= 0) continue;

            int avgDepth = depthSum[vx][vy] / cnt;
            int vz = GRID_Z - avgDepth * GRID_Z / 255 - 1;

            Vec3b avgColor(
                (uchar)(colorBSum[vx][vy] / cnt),
                (uchar)(colorGSum[vx][vy] / cnt),
                (uchar)(colorRSum[vx][vy] / cnt)
            );

            int z0 = max(0, vz - backThickness);
            int z1 = min(GRID_Z - 1, vz + backThickness);
            for (int zz = z0; zz <= z1; zz++)
                safeSetColor(vx, vy, zz, avgColor);
        }
}

void buildLeft()
{
    int w = depthLeft.cols;
    int h = depthLeft.rows;

    static int depthSum[128][128];
    static int colorBSum[128][128];
    static int colorGSum[128][128];
    static int colorRSum[128][128];
    static int hitCount[128][128];

    for (int z = 0; z < GRID_Z; z++)
        for (int y = 0; y < GRID_Y; y++)
        {
            depthSum[z][y] = 0;
            colorBSum[z][y] = 0;
            colorGSum[z][y] = 0;
            colorRSum[z][y] = 0;
            hitCount[z][y] = 0;
        }

    for (int y = 0; y < h; y++)
        for (int z = 0; z < w; z++)
        {
            if (maskLeft.at<uchar>(y, z) == 0)
                continue;

            uchar d =
                depthLeft.at<uchar>(y, z);

            int vy = (h - y - 1) * GRID_Y / h;
            int vz = z * GRID_Z / w;

            Vec3b c =
                leftImg.at<Vec3b>(y, z);

            if (vz < 0 || vy < 0 ||
                vz >= GRID_Z || vy >= GRID_Y)
                continue;

            depthSum[vz][vy] += (int)d;
            colorBSum[vz][vy] += (int)c[0];
            colorGSum[vz][vy] += (int)c[1];
            colorRSum[vz][vy] += (int)c[2];
            hitCount[vz][vy]++;
        }

    const int leftThickness = 2;
    for (int vz = 0; vz < GRID_Z; vz++)
        for (int vy = 0; vy < GRID_Y; vy++)
        {
            int cnt = hitCount[vz][vy];
            if (cnt <= 0) continue;

            int avgDepth = depthSum[vz][vy] / cnt;
            int vx = avgDepth * GRID_X / 255;

            Vec3b avgColor(
                (uchar)(colorBSum[vz][vy] / cnt),
                (uchar)(colorGSum[vz][vy] / cnt),
                (uchar)(colorRSum[vz][vy] / cnt)
            );

            int x0 = max(0, vx - leftThickness);
            int x1 = min(GRID_X - 1, vx + leftThickness);
            for (int xx = x0; xx <= x1; xx++)
                safeSetColor(xx, vy, vz, avgColor);
        }
}

void buildRight()
{
    int w = depthRight.cols;
    int h = depthRight.rows;

    static int depthSum[128][128];
    static int colorBSum[128][128];
    static int colorGSum[128][128];
    static int colorRSum[128][128];
    static int hitCount[128][128];

    for (int z = 0; z < GRID_Z; z++)
        for (int y = 0; y < GRID_Y; y++)
        {
            depthSum[z][y] = 0;
            colorBSum[z][y] = 0;
            colorGSum[z][y] = 0;
            colorRSum[z][y] = 0;
            hitCount[z][y] = 0;
        }

    for (int y = 0; y < h; y++)
        for (int z = 0; z < w; z++)
        {
            if (maskRight.at<uchar>(y, z) == 0)
                continue;

            uchar d =
                depthRight.at<uchar>(y, z);

            int vy = (h - y - 1) * GRID_Y / h;
            int vz = z * GRID_Z / w;

            Vec3b c =
                rightImg.at<Vec3b>(y, z);

            if (vz < 0 || vy < 0 ||
                vz >= GRID_Z || vy >= GRID_Y)
                continue;

            depthSum[vz][vy] += (int)d;
            colorBSum[vz][vy] += (int)c[0];
            colorGSum[vz][vy] += (int)c[1];
            colorRSum[vz][vy] += (int)c[2];
            hitCount[vz][vy]++;
        }

    const int rightThickness = 2;
    for (int vz = 0; vz < GRID_Z; vz++)
        for (int vy = 0; vy < GRID_Y; vy++)
        {
            int cnt = hitCount[vz][vy];
            if (cnt <= 0) continue;

            int avgDepth = depthSum[vz][vy] / cnt;
            int vx = GRID_X - avgDepth * GRID_X / 255 - 1;

            Vec3b avgColor(
                (uchar)(colorBSum[vz][vy] / cnt),
                (uchar)(colorGSum[vz][vy] / cnt),
                (uchar)(colorRSum[vz][vy] / cnt)
            );

            int x0 = max(0, vx - rightThickness);
            int x1 = min(GRID_X - 1, vx + rightThickness);
            for (int xx = x0; xx <= x1; xx++)
                safeSetColor(xx, vy, vz, avgColor);
        }
}

void buildTop()
{
    int w = depthTop.cols;
    int h = depthTop.rows;

    static int depthSum[128][128];
    static int colorBSum[128][128];
    static int colorGSum[128][128];
    static int colorRSum[128][128];
    static int hitCount[128][128];

    for (int x = 0; x < GRID_X; x++)
        for (int z = 0; z < GRID_Z; z++)
        {
            depthSum[x][z] = 0;
            colorBSum[x][z] = 0;
            colorGSum[x][z] = 0;
            colorRSum[x][z] = 0;
            hitCount[x][z] = 0;
        }

    for (int z = 0; z < h; z++)
        for (int x = 0; x < w; x++)
        {
            if (maskTop.at<uchar>(z, x) == 0)
                continue;

            uchar d =
                depthTop.at<uchar>(z, x);

            int vx = x * GRID_X / w;
            int vz = z * GRID_Z / h;

            Vec3b c =
                topImg.at<Vec3b>(z, x);

            if (vx < 0 || vz < 0 ||
                vx >= GRID_X || vz >= GRID_Z)
                continue;

            depthSum[vx][vz] += (int)d;
            colorBSum[vx][vz] += (int)c[0];
            colorGSum[vx][vz] += (int)c[1];
            colorRSum[vx][vz] += (int)c[2];
            hitCount[vx][vz]++;
        }

    const int topThickness = 2;
    for (int vx = 0; vx < GRID_X; vx++)
        for (int vz = 0; vz < GRID_Z; vz++)
        {
            int cnt = hitCount[vx][vz];
            if (cnt <= 0) continue;

            int avgDepth = depthSum[vx][vz] / cnt;
            int vy0 = avgDepth * GRID_Y / 255;
            // 顶端锚定不动，仅向下方向拉伸 1.5 倍
            int yTop = GRID_Y - 1;
            int vy = yTop - (int)((yTop - vy0) * 1.5f);
            vy = min(GRID_Y - 1, max(0, vy));
            Vec3b avgColor(
                (uchar)(colorBSum[vx][vz] / cnt),
                (uchar)(colorGSum[vx][vz] / cnt),
                (uchar)(colorRSum[vx][vz] / cnt)
            );

            int y0 = max(0, vy - topThickness);
            int y1 = min(GRID_Y - 1, vy + topThickness);
            for (int yy = y0; yy <= y1; yy++)
                safeSetColor(vx, yy, vz, avgColor);
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
                if (!voxel[x][y][z])
                    continue;

                Vec3b c =
                    voxelColor[x][y][z];

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
    frontImg = imread("data/samples/multiview/front.png");
    backImg = imread("data/samples/multiview/back.png");
    hasRealBack = !backImg.empty();

    leftImg = imread("data/samples/multiview/left.png");
    rightImg = imread("data/samples/multiview/right.png");
    hasRealRight = !rightImg.empty();

    topImg = imread("data/samples/multiview/top.png");

    if (frontImg.empty() ||
        leftImg.empty() ||
        topImg.empty())
    {
        cout << "Need front left top" << endl;
        return -1;
    }

    alignImages();

    depthFront = runDepthAnything(frontImg);
    depthBack = runDepthAnything(backImg);
    depthLeft = runDepthAnything(leftImg);
    depthRight = runDepthAnything(rightImg);
    depthTop = runDepthAnything(topImg);

    // 保存5张深度图到当前目录（便于观察）
    saveDepthDebug("depth/depth_front.png", depthFront);
    saveDepthDebug("depth/depth_back.png", depthBack);
    saveDepthDebug("depth/depth_left.png", depthLeft);
    saveDepthDebug("depth/depth_right.png", depthRight);
    saveDepthDebug("depth/depth_top.png", depthTop);

    initVoxel();

    buildFront();
    buildLeft();
    buildTop();

    
    buildBack();

    
   buildRight();

    // 清理孤立噪声点（值越大去噪越强）
    removeIsolatedVoxels(4);

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