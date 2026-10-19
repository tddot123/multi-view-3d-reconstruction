/*#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <algorithm>

// 图像
cv::Mat depthImg;
cv::Mat colorImg;

int width, height;

// 模型参数
float scale = 50.0f;
float modelSize = 0.0f;

// 相机
float angleX = 0.0f;
float angleY = 0.0f;
float distance = 300.0f;

// 鼠标
int lastX, lastY;
bool isDragging = false;

// 高度
float getHeight(int x, int y) {
    unsigned char d = depthImg.at<uchar>(y, x);
    return (d / 255.0f) * scale;
}

// 颜色
void getColor(int x, int y, float* r, float* g, float* b) {
    cv::Vec3b color = colorImg.at<cv::Vec3b>(y, x);
    *b = color[0] / 255.0f;
    *g = color[1] / 255.0f;
    *r = color[2] / 255.0f;
}

// ⭐ HeightMap（改成 XY 平面 + Z 深度）
void drawHeightMap() {

    int step = std::max(1, width / 300);  // ⭐自适应密度

    for (int y = 0; y < height - step; y += step) {
        glBegin(GL_TRIANGLE_STRIP);
        for (int x = 0; x < width; x += step) {

            float r, g, b;

            // 当前点
            float h1 = getHeight(x, y);
            getColor(x, y, &r, &g, &b);
            glColor3f(r, g, b);

            float worldX1 = (x - width / 2.0f);
            float worldY1 = (height / 2.0f - y);
            float worldZ1 = h1;

            glVertex3f(worldX1, worldY1, worldZ1);

            // 下一行
            float h2 = getHeight(x, y + step);
            getColor(x, y + step, &r, &g, &b);
            glColor3f(r, g, b);

            float worldX2 = (x - width / 2.0f);
            float worldY2 = (height / 2.0f - (y + step));
            float worldZ2 = h2;

            glVertex3f(worldX2, worldY2, worldZ2);
        }
        glEnd();
    }
}

// 渲染
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float radX = angleX * 3.14159f / 180.0f;
    float radY = angleY * 3.14159f / 180.0f;

    float camX = distance * sin(radY) * cos(radX);
    float camY = distance * sin(radX);
    float camZ = distance * cos(radY) * cos(radX);

    gluLookAt(camX, camY, camZ,
        0, 0, 0,
        0, 1, 0);

    drawHeightMap();

    glutSwapBuffers();
}

// 窗口
void reshape(int w, int h) {
    if (h == 0) h = 1;

    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float)w / h, 1.0, 5000.0);

    glMatrixMode(GL_MODELVIEW);
}

// 初始化
void init() {
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
}

// 鼠标
void mouse(int button, int state, int x, int y) {

    if (button == GLUT_LEFT_BUTTON) {
        isDragging = (state == GLUT_DOWN);
        lastX = x;
        lastY = y;
    }

    // 滚轮缩放（更平滑）
    if (button == 3) distance *= 0.9f;
    if (button == 4) distance *= 1.1f;

    if (distance < 10) distance = 10;

    glutPostRedisplay();
}

// 鼠标移动
void motion(int x, int y) {
    if (isDragging) {
        angleY += (x - lastX) * 0.5f;
        angleX += (y - lastY) * 0.5f;

        if (angleX > 89) angleX = 89;
        if (angleX < -89) angleX = -89;

        lastX = x;
        lastY = y;

        glutPostRedisplay();
    }
}

// 主函数
int main(int argc, char** argv) {

    depthImg = cv::imread("data/samples/rgbd/depth4.png", cv::IMREAD_GRAYSCALE);
    colorImg = cv::imread("data/samples/rgbd/color4.jpg", cv::IMREAD_COLOR);

    if (depthImg.empty() || colorImg.empty()) {
        printf("Failed to load images\n");
        return -1;
    }

    width = depthImg.cols;
    height = depthImg.rows;

    // ⭐ 限制分辨率（防爆）
    int maxRes = 400;
    if (width > maxRes || height > maxRes) {
        float scaleFactor = (float)maxRes / std::max(width, height);
        cv::resize(depthImg, depthImg, cv::Size(), scaleFactor, scaleFactor);
        cv::resize(colorImg, colorImg, depthImg.size());

        width = depthImg.cols;
        height = depthImg.rows;
    }

    if (colorImg.size() != depthImg.size()) {
        cv::resize(colorImg, colorImg, depthImg.size());
    }

    // ⭐ 深度平滑（解决你“离散”问题）
    cv::medianBlur(depthImg, depthImg, 5);

    // ⭐ 自动缩放模型
    float maxDim = std::max(width, height);
    scale = maxDim / 3.0f;

    modelSize = maxDim;

    // ⭐ 自动相机距离
    distance = modelSize * 1.5f;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("HeightMap 3D Viewer (Adaptive)");

    init();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutIdleFunc(display);

    glutMainLoop();
    return 0;
}*/