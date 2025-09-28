#include <fstream>
#include <opencv2/opencv.hpp>

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.5;
const float CONFIDENCE_THRESHOLD = 0.5;

std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("../yolo/coco-classes.txt");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void load_model(cv::dnn::Net &model) {
    auto result = cv::dnn::readNet("../yolo/yolov5s.onnx");
    result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    model = result;
}

cv::Mat format_input(const cv::Mat &image) {
    int col = image.cols;
    int row = image.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    image.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void forward_model(cv::Mat &input_image, cv::Mat &output_image, cv::dnn::Net &model) {
    cv::Mat blob;
    std::vector<cv::Mat> outputs;

    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    model.setInput(blob);
    
    model.forward(outputs, model.getUnconnectedOutLayersNames());
    output_image = outputs[0];
}

cv::Rect get_box(float *data, float x_factor, float y_factor) {
    float x = data[0];
    float y = data[1];
    float w = data[2];
    float h = data[3];

    int left = int((x - 0.5 * w) * x_factor);
    int top = int((y - 0.5 * h) * y_factor);
    int width = int(w * x_factor);
    int height = int(h * y_factor);

    return cv::Rect(left, top, width, height);
}

void detect(cv::Mat &image, cv::dnn::Net &model, const std::vector<std::string> &className) {
    auto input_image = format_input(image);
    cv::Mat output_image;
    
    forward_model(input_image, output_image, model);
    float *data = (float *)output_image.data;

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    const int rows = 25200;
    const int dimensions = 85;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];

        if (confidence >= CONFIDENCE_THRESHOLD) {
            float *classes_scores = data + 5;

            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                boxes.push_back(get_box(data, x_factor, y_factor));
            }
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];

        std::string label = className[class_ids[idx]] + ": ." + std::to_string(lround(confidences[idx] * 1000));
        cv::rectangle(image, boxes[idx], cv::Scalar(0, 255, 255), 1);
        cv::putText(image, label, cv::Point(boxes[idx].x, boxes[idx].y - 5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 1);
    }
}

int main(int argc, char **argv) {
    std::vector<std::string> class_list = load_class_list();

    cv::dnn::Net model;
    load_model(model);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam.\n";
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        detect(frame, model, class_list);

        cv::imshow("C++ Object Detector", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}