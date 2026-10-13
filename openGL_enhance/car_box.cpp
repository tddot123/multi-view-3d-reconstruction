/*#include <opencv2/opencv.hpp>
#include <GL/freeglut.h>
#include <iostream>
#ifndef GL_BGR
#define GL_BGR 0x80E0
#endif

#ifndef GL_BGRA
#define GL_BGRA 0x80E1
#endif

using namespace cv;
using namespace std;

////////////////////////////////////////////////////////
// 图像
////////////////////////////////////////////////////////

Mat frontImg, backImg;
Mat leftImg, rightImg;
Mat topImg;

GLuint texFront, texBack;
GLuint texLeft, texRight;
GLuint texTop;

////////////////////////////////////////////////////////
// 相机
////////////////////////////////////////////////////////

float angleX = 20;
float angleY = 30;
float distanceZ = 800;

int lastX, lastY;
bool isDragging = false;

////////////////////////////////////////////////////////
// 六视图几何对齐
////////////////////////////////////////////////////////

Mat removeWhiteAndEnhanceEdges(Mat src)
{
    //////////////////////////////////////
    // Step1: HSV 去白
    //////////////////////////////////////

    Mat hsv;

    cvtColor(
        src,
        hsv,
        COLOR_BGR2HSV);

    Mat maskWhite;

    inRange(
        hsv,
        Scalar(0, 0, 200),
        Scalar(180, 40, 255),
        maskWhite);

    Mat mask;

    bitwise_not(
        maskWhite,
        mask);

    //////////////////////////////////////
    // Step2: 边缘增强
    //////////////////////////////////////

    Mat gray;

    cvtColor(
        src,
        gray,
        COLOR_BGR2GRAY);

    Mat edges;

    Canny(
        gray,
        edges,
        80,
        160);

    bitwise_or(
        mask,
        edges,
        mask);

    //////////////////////////////////////
    // Step3: 填洞
    //////////////////////////////////////

    Mat kernel =
        getStructuringElement(
            MORPH_RECT,
            Size(7, 7));

    morphologyEx(
        mask,
        mask,
        MORPH_CLOSE,
        kernel);

    //////////////////////////////////////
    // Step4: 找最大轮廓
    //////////////////////////////////////

    vector<vector<Point>> contours;

    findContours(
        mask,
        contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        return src;

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

    //////////////////////////////////////
    // Step5: 精确 mask
    //////////////////////////////////////

    Mat objectMask =
        Mat::zeros(
            src.size(),
            CV_8UC1);

    drawContours(
        objectMask,
        contours,
        maxIndex,
        Scalar(255),
        FILLED);

    //////////////////////////////////////
    // Step6: bounding box
    //////////////////////////////////////

    Rect box =
        boundingRect(
            contours[maxIndex]);

    int pad = 5;

    box.x = max(0, box.x - pad);
    box.y = max(0, box.y - pad);

    box.width =
        min(src.cols - box.x,
            box.width + pad * 2);

    box.height =
        min(src.rows - box.y,
            box.height + pad * 2);

    //////////////////////////////////////
    // Step7: 裁剪 mask + image
    //////////////////////////////////////

    Mat croppedImg =
        src(box);

    Mat croppedMask =
        objectMask(box);

    //////////////////////////////////////
    // Step8: 生成 RGBA（关键）
    //////////////////////////////////////

    Mat rgba;

    cvtColor(
        croppedImg,
        rgba,
        COLOR_BGR2BGRA);

    vector<Mat> channels;

    split(
        rgba,
        channels);

    channels[3] =
        croppedMask;   // Alpha

    merge(
        channels,
        rgba);

    return rgba;
}
void alignImages()
{
    //////////////////////////////////////
    // Step0: 所有图先做预处理
    //////////////////////////////////////

    frontImg =
        removeWhiteAndEnhanceEdges(frontImg);

    leftImg =
        removeWhiteAndEnhanceEdges(leftImg);

    topImg =
        removeWhiteAndEnhanceEdges(topImg);

    if (!backImg.empty())
        backImg =
        removeWhiteAndEnhanceEdges(backImg);

    if (!rightImg.empty())
        rightImg =
        removeWhiteAndEnhanceEdges(rightImg);

    //////////////////////////////////////
    // Step1: 统一方向
    //////////////////////////////////////

    rotate(frontImg, frontImg, ROTATE_180);

    if (!backImg.empty())
        rotate(backImg, backImg, ROTATE_180);

    rotate(leftImg, leftImg, ROTATE_180);

    if (!rightImg.empty())
        rotate(rightImg, rightImg, ROTATE_180);

    rotate(topImg, topImg, ROTATE_180);

    //////////////////////////////////////
    // Step2: top 再旋转90°
    //////////////////////////////////////

    rotate(
        topImg,
        topImg,
        ROTATE_90_COUNTERCLOCKWISE);

    //////////////////////////////////////
    // Step3: front 作为基准
    //////////////////////////////////////

    int W = frontImg.cols;
    int H = frontImg.rows;

    //////////////////////////////////////
    // Step4: left 高对齐
    //////////////////////////////////////

    float scaleL =
        (float)H / leftImg.rows;

    resize(
        leftImg,
        leftImg,
        Size(),
        scaleL,
        scaleL);

    int D = leftImg.cols;

    //////////////////////////////////////
    // Step5: top 匹配
    //////////////////////////////////////

    resize(
        topImg,
        topImg,
        Size(W, D));

    //////////////////////////////////////
    // Step6: 自动补 back/right
    //////////////////////////////////////

    if (backImg.empty())
        flip(frontImg,
            backImg,
            1);

    if (rightImg.empty())
        flip(leftImg,
            rightImg,
            1);

    //////////////////////////////////////
    // Step7: 统一尺寸
    //////////////////////////////////////

    resize(
        backImg,
        backImg,
        Size(W, H));

    resize(
        rightImg,
        rightImg,
        Size(D, H));

    cout << "Final Size:"
        << " W=" << W
        << " H=" << H
        << " D=" << D
        << endl;
}

////////////////////////////////////////////////////////
// 纹理
////////////////////////////////////////////////////////

GLuint loadTexture(Mat img)
{
    GLuint tex;

    glPixelStorei(
        GL_UNPACK_ALIGNMENT,
        1);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    //////////////////////////////////////
    // 判断是否有透明通道
    //////////////////////////////////////

    GLenum format;

    if (img.channels() == 4)
        format = GL_BGRA;
    else
        format = GL_BGR;

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        img.cols,
        img.rows,
        0,
        format,
        GL_UNSIGNED_BYTE,
        img.data);

    //////////////////////////////////////

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MIN_FILTER,
        GL_LINEAR);

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_LINEAR);

    //////////////////////////////////////
    // 开启透明
    //////////////////////////////////////

    glEnable(GL_BLEND);

    glBlendFunc(
        GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA);

    return tex;
}

////////////////////////////////////////////////////////
// OpenGL 初始化
////////////////////////////////////////////////////////

void initGL()
{
    glEnable(GL_DEPTH_TEST);

    glClearColor(
        0.1f,
        0.1f,
        0.1f,
        1.0f);

    texFront = loadTexture(frontImg);
    texBack = loadTexture(backImg);
    texLeft = loadTexture(leftImg);
    texRight = loadTexture(rightImg);
    texTop = loadTexture(topImg);
}

////////////////////////////////////////////////////////
// 绘制盒子（无bottom）
////////////////////////////////////////////////////////

void drawBox()
{
    float W = frontImg.cols * 0.5f;
    float H = frontImg.rows * 0.5f;
    float D = leftImg.cols * 0.5f;

    glEnable(GL_TEXTURE_2D);

    // FRONT

    glBindTexture(GL_TEXTURE_2D, texFront);

    glBegin(GL_QUADS);

    glTexCoord2f(1, 0);
    glVertex3f(-W, -H, D);

    glTexCoord2f(0, 0);
    glVertex3f(W, -H, D);

    glTexCoord2f(0, 1);
    glVertex3f(W, H, D);

    glTexCoord2f(1, 1);
    glVertex3f(-W, H, D);

    glEnd();

    // BACK

    glBindTexture(GL_TEXTURE_2D, texBack);

    glBegin(GL_QUADS);

    glTexCoord2f(0, 0);
    glVertex3f(W, -H, -D);

    glTexCoord2f(1, 0);
    glVertex3f(-W, -H, -D);

    glTexCoord2f(1, 1);
    glVertex3f(-W, H, -D);

    glTexCoord2f(0, 1);
    glVertex3f(W, H, -D);

    glEnd();

    // LEFT

    glBindTexture(GL_TEXTURE_2D, texLeft);

    glBegin(GL_QUADS);

    glTexCoord2f(0, 0);
    glVertex3f(-W, -H, -D);

    glTexCoord2f(1, 0);
    glVertex3f(-W, -H, D);

    glTexCoord2f(1, 1);
    glVertex3f(-W, H, D);

    glTexCoord2f(0, 1);
    glVertex3f(-W, H, -D);

    glEnd();

    // RIGHT

    glBindTexture(GL_TEXTURE_2D, texRight);

    glBegin(GL_QUADS);

    glTexCoord2f(0, 0);
    glVertex3f(W, -H, D);

    glTexCoord2f(1, 0);
    glVertex3f(W, -H, -D);

    glTexCoord2f(1, 1);
    glVertex3f(W, H, -D);

    glTexCoord2f(0, 1);
    glVertex3f(W, H, D);

    glEnd();

    // TOP

    glBindTexture(GL_TEXTURE_2D, texTop);

    glBegin(GL_QUADS);

    glTexCoord2f(0, 0);
    glVertex3f(-W, H, D);

    glTexCoord2f(1, 0);
    glVertex3f(W, H, D);

    glTexCoord2f(1, 1);
    glVertex3f(W, H, -D);

    glTexCoord2f(0, 1);
    glVertex3f(-W, H, -D);

    glEnd();

    glDisable(GL_TEXTURE_2D);
}

////////////////////////////////////////////////////////

void display()
{
    glClear(
        GL_COLOR_BUFFER_BIT |
        GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();

    glTranslatef(0, 0, -distanceZ);

    glRotatef(angleX, 1, 0, 0);
    glRotatef(angleY, 0, 1, 0);

    drawBox();

    glutSwapBuffers();
}

////////////////////////////////////////////////////////
// 鼠标
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

void reshape(int w, int h)
{
    if (h == 0) h = 1;

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
// MAIN
////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    frontImg = imread("./input/f0.png");
    backImg = imread("./input/b01.png");

    leftImg = imread("./input/l01.png");
    rightImg = imread("./ input / r01.png");

    topImg = imread("./input/t01.png");

    if (frontImg.empty() ||
        leftImg.empty() ||
        topImg.empty())
    {
        cout << "Need front left top" << endl;
        return -1;
    }

    alignImages();

    glutInit(&argc, argv);

    glutInitDisplayMode(
        GLUT_DOUBLE |
        GLUT_RGB |
        GLUT_DEPTH);

    glutInitWindowSize(900, 700);

    glutCreateWindow(
        "Six View Box");

    initGL();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);

    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    glutIdleFunc(display);

    glutMainLoop();

    return 0;
}*/