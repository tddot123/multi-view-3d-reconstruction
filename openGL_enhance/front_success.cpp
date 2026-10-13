/*#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
cv::Mat runDepthAnything(cv::Mat& inputImg) {

    try {

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DepthAnything");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        Ort::Session session(env, L"model/depth_anything_v2_large.onnx", session_options);

        cv::Mat resized;
        cv::resize(inputImg, resized, cv::Size(384, 384));

        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);

        std::vector<float> inputTensorValues(1 * 3 * 384 * 384);

        int index = 0;
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < 384; y++) {
                for (int x = 0; x < 384; x++) {
                    inputTensorValues[index++] = resized.at<cv::Vec3f>(y, x)[c];
                }
            }
        }

        std::array<int64_t, 4> inputShape = { 1, 3, 384, 384 };

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memory_info,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size()
        );

        const char* input_names[] = { "pixel_values" };
        const char* output_names[] = { "predicted_depth" };

        auto outputTensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_names,
            &inputTensor,
            1,
            output_names,
            1
        );

        float* outputData = outputTensors[0].GetTensorMutableData<float>();

        auto shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        int outH = shape[1];
        int outW = shape[2];

        cv::Mat depth(outH, outW, CV_32F, outputData);

        cv::Mat depthNorm;
        cv::normalize(depth, depthNorm, 0, 255, cv::NORM_MINMAX);
        depthNorm.convertTo(depthNorm, CV_8U);

        cv::Mat finalDepth;
        cv::resize(depthNorm, finalDepth, inputImg.size());

        return finalDepth.clone();
    }
    catch (const std::exception& e) {
        std::cout << "ONNX Error: " << e.what() << std::endl;
        return cv::Mat();
    }
}
// 图像
cv::Mat depthImg;
cv::Mat colorImg;

int width, height;

// voxel参数
float cubeSize = 2.0f;
float heightScale = 50.0f;

// 模型尺寸
float modelSize = 0.0f;

// 相机
float angleX = 0.0f;
float angleY = 0.0f;
float distance = 400.0f;

// 鼠标
int lastX, lastY;
bool isDragging = false;

// 获取高度
float getHeight(int x, int y) {
    unsigned char d = depthImg.at<uchar>(y, x);
    return (d / 255.0f) * heightScale;
}

// 获取颜色
void getColor(int x, int y, float* r, float* g, float* b) {
    cv::Vec3b color = colorImg.at<cv::Vec3b>(y, x);
    *b = color[0] / 255.0f;
    *g = color[1] / 255.0f;
    *r = color[2] / 255.0f;
}

// 画立方体
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

// voxel绘制
void drawVoxelModel() {

    int step = std::max(1, width / 200);  // ⭐自适应密度

    for (int y = 0; y < height; y += step) {
        for (int x = 0; x < width; x += step) {

            float h = getHeight(x, y);
            if (h < 1.0f) continue;

            float r, g, b;
            getColor(x, y, &r, &g, &b);
            glColor3f(r, g, b);

            // ⭐XY平面 + Z深度（关键）
            float worldX = (x - width / 2.0f) * cubeSize;
            float worldY = (height / 2.0f - y) * cubeSize;
            float worldZ = h;

            drawCube(worldX, worldY, worldZ, cubeSize);
        }
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

    drawVoxelModel();

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
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
}

// 鼠标
void mouse(int button, int state, int x, int y) {

    if (button == GLUT_LEFT_BUTTON) {
        isDragging = (state == GLUT_DOWN);
        lastX = x;
        lastY = y;
    }

    // 滚轮缩放
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

    colorImg = cv::imread("input/f04.png", cv::IMREAD_COLOR);
    if (colorImg.empty()) {
        printf("Failed to load image\n");
        return -1;
    }

        depthImg = runDepthAnything(colorImg);

        if (depthImg.empty() || colorImg.empty()) {
            printf("Failed to load images\n");
            return -1;
        }

        width = depthImg.cols;
        height = depthImg.rows;

        // ⭐限制最大分辨率（防卡死）
        int maxRes = 300;
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

        // ⭐自适应模型尺寸
        float maxDim = std::max(width, height);

        cubeSize = 200.0f / maxDim;   // 控制整体大小
        heightScale = 100.0f;

        modelSize = maxDim * cubeSize;

        // ⭐自动相机距离
        distance = modelSize * 2.0f;

        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(800, 600);
        glutCreateWindow("Voxel 3D Model (Adaptive)");

        init();

        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutIdleFunc(display);

        glutMainLoop();
        return 0;
    }*/