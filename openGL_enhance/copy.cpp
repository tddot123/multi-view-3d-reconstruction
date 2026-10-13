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

////////////////////////////////////////////////////////
// ⭐比例 GRID（只新增）
////////////////////////////////////////////////////////

//const int BASE_GRID = 128;

int GRID_X;
int GRID_Y;
int GRID_Z;

bool voxel[128][128][128];
Vec3b voxelColor[128][128][128];

float cubeSize = 1.5f;

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

    int D = leftImg.cols;

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
        flip(leftImg, rightImg, 1);
        flip(maskLeft, maskRight, 1);
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

    int MAX_RES = 256;
    // 推荐范围：
    // 192 → 普通电脑
    // 256 → 推荐（大多数电脑可跑）
    // 320 → 高性能电脑

    int maxDim =
        max({ W,H,D });

    float nscale =
        min(1.0f,
            (float)MAX_RES /
            maxDim);

    GRID_X = int(W * nscale);
    GRID_Y = int(H * nscale);
    GRID_Z = int(D * nscale);

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
    static Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "Depth");

    static Ort::SessionOptions opts;

    static Ort::Session session(
        env,
        L"depth_anything_v2_large.onnx",
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

    Mat depth8U;

    normalize(
        depth,
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

    return depth8U;
}

////////////////////////////////////////////////////////
// voxel
////////////////////////////////////////////////////////

void initVoxel()
{
    memset(voxel, false, sizeof(voxel));
}

void safeSetColor(int x, int y, int z, Vec3b c)
{
    if (x < 0 || y < 0 || z < 0) return;

    if (x >= GRID_X ||
        y >= GRID_Y ||
        z >= GRID_Z) return;

    voxel[x][y][z] = true;
    voxelColor[x][y][z] = c;
}

////////////////////////////////////////////////////////
// ⭐build 全部保留（只改 GRID→GRID_X/Y/Z）
////////////////////////////////////////////////////////

void buildFront()
{
    int w = depthFront.cols;
    int h = depthFront.rows;

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            if (maskFront.at<uchar>(y, x) == 0)
                continue;

            uchar d =
                depthFront.at<uchar>(y, x);

            int vx = x * GRID_X / w;
            int vy = (h - y - 1) * GRID_Y / h;
            int vz = d * GRID_Z / 255;

            Vec3b c =
                frontImg.at<Vec3b>(y, x);

            safeSetColor(vx, vy, vz, c);
        }
}

void buildBack()
{
    int w = depthBack.cols;
    int h = depthBack.rows;

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            if (maskBack.at<uchar>(y, x) == 0)
                continue;

            uchar d =
                depthBack.at<uchar>(y, x);

            int vx = x * GRID_X / w;
            int vy = (h - y - 1) * GRID_Y / h;
            int vz = GRID_Z - d * GRID_Z / 255 - 1;

            Vec3b c =
                backImg.at<Vec3b>(y, x);

            safeSetColor(vx, vy, vz, c);
        }
}

void buildLeft()
{
    int w = depthLeft.cols;
    int h = depthLeft.rows;

    for (int y = 0; y < h; y++)
        for (int z = 0; z < w; z++)
        {
            if (maskLeft.at<uchar>(y, z) == 0)
                continue;

            uchar d =
                depthLeft.at<uchar>(y, z);

            int vx = d * GRID_X / 255;
            int vy = (h - y - 1) * GRID_Y / h;
            int vz = z * GRID_Z / w;

            Vec3b c =
                leftImg.at<Vec3b>(y, z);

            safeSetColor(vx, vy, vz, c);
        }
}

void buildRight()
{
    int w = depthRight.cols;
    int h = depthRight.rows;

    for (int y = 0; y < h; y++)
        for (int z = 0; z < w; z++)
        {
            if (maskRight.at<uchar>(y, z) == 0)
                continue;

            uchar d =
                depthRight.at<uchar>(y, z);

            int vx = GRID_X - d * GRID_X / 255 - 1;
            int vy = (h - y - 1) * GRID_Y / h;
            int vz = z * GRID_Z / w;

            Vec3b c =
                rightImg.at<Vec3b>(y, z);

            safeSetColor(vx, vy, vz, c);
        }
}

void buildTop()
{
    int w = depthTop.cols;
    int h = depthTop.rows;

    for (int z = 0; z < h; z++)
        for (int x = 0; x < w; x++)
        {
            if (maskTop.at<uchar>(z, x) == 0)
                continue;

            uchar d =
                depthTop.at<uchar>(z, x);

            int vx = x * GRID_X / w;
            int vy = d * GRID_Y / 255;
            int vz = z * GRID_Z / h;

            Vec3b c =
                topImg.at<Vec3b>(z, x);

            safeSetColor(vx, vy, vz, c);
        }
}

////////////////////////////////////////////////////////
// OpenGL（完全保留）
////////////////////////////////////////////////////////

void drawCube(float x, float y, float z, float size) {
    float s = size / 1.0f;

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
    frontImg = imread("six/front.png");
    backImg = imread("six/back.png");

    leftImg = imread("six/left.png");
    rightImg = imread("six/right.png");

    topImg = imread("six/top.png");

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

    initVoxel();

    // buildFront();
    // buildBack();
     //buildLeft();
    // buildRight();
    buildTop();

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