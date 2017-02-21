#include "opencv2/opencv.hpp"
#include <random>
#include <fstream>

using namespace cv;
using namespace std;

void primitive_creation() {
    Mat cross(600, 600, CV_8UC3, Scalar::all(0));
    line(cross, Point(250, 300), Point(350, 300), Scalar(255, 255, 255), 4);
    line(cross, Point(300, 250), Point(300, 350), Scalar(255, 255, 255), 4);
    imwrite("cross.png", cross);
    //
    Mat rect(600, 600, CV_8UC3, Scalar::all(0));
    rectangle(rect, Point(250, 250), Point(350, 350), Scalar(255, 255, 255), 4);
    imwrite("rectangle.png", rect);
    //
    Mat cir(600, 600, CV_8UC3, Scalar::all(0));
    circle(cir, Point(250, 250), 50, Scalar(255, 255, 255), 4);
    imwrite("circle.png", cir);
}
//
void rotation(Mat &target, double angle) {
    Point center = Point(target.cols / 2, target.rows / 2);
    double scale = 1;
    Mat rot_mat = getRotationMatrix2D(center, angle, scale);
    warpAffine(target, target, rot_mat, target.size());
}
//
void scale(Mat &target, double cof, double offset = 0) {

    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> uni(-1, 1); // guaranteed unbiased

    auto random_integer = uni(rng);


    Point2f inputQuad[4];
    Point2f outputQuad[4];
    Mat lambda(2, 4, CV_32FC1);

    lambda = Mat::zeros(target.rows, target.cols, target.type());

    Point2f center(target.rows / 2, target.cols / 2);
    double help = target.rows * cof - target.rows;

    inputQuad[0] = Point2f(0, 0);
    inputQuad[1] = Point2f(0, target.cols);
    inputQuad[2] = Point2f(target.rows, target.cols);
    inputQuad[3] = Point2f(target.rows, 0);
    //
    outputQuad[0] = Point2f(-help + offset * uni(rng), -help + offset * uni(rng));
    outputQuad[1] = Point2f(-help + offset * uni(rng), target.cols * cof + offset * uni(rng));
    outputQuad[2] = Point2f(target.rows * cof + offset * uni(rng), target.cols * cof + offset * uni(rng));
    outputQuad[3] = Point2f(target.rows * cof + offset * uni(rng), -help + offset * uni(rng));
    //
    lambda = getPerspectiveTransform(inputQuad, outputQuad);
    warpPerspective(target, target, lambda, target.size());
}
//
void smooth(Mat &target, int degree) {
    if (degree % 2 == 0) degree++;
    GaussianBlur(target, target, Size(degree, degree), 0, 0);
}
//
void write_training(const string& case_name, const string& path,const string& results, int N){
    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> angle(0, 45); // guaranteed unbiased
    std::uniform_int_distribution<int> smth(0, 10);
    std::uniform_real_distribution<double> scl(0.8, 1.2);
    std::uniform_real_distribution<double> offset(0, 100);
    Mat original_image = imread(path + "/" + case_name + ".png", CV_LOAD_IMAGE_COLOR);
    ofstream training_file (path + results + "/" + case_name + ".txt" );
    training_file <<"/" + case_name << endl;
    for (int i = 0; i < N; ++i) {
        Mat image = original_image.clone();  // Read the file
        rotation(image, angle(rng));
        scale(image, scl(rng), offset(rng));
        smooth(image, smth(rng));
        //
        vector<Point> locations;   // output, locations of non-zero pixels
        cvtColor(image, image, CV_BGR2GRAY);
        cv::findNonZero(image, locations);
        cv::Rect myROI = boundingRect(locations);
        cv::Mat croppedImage = image(myROI);
        string write = path + results + "/" + case_name + "/" + case_name + to_string(i) + ".jpg";
        training_file <<case_name + "/" + case_name + to_string(i) + ".jpg";
        training_file<< " 1 0 0 ";
        training_file << to_string(croppedImage.cols)<< " " << to_string(croppedImage.rows)<<endl;
        imwrite(write, croppedImage);
    }
    training_file.close();

}
//
void add_bad(const vector<string>& cases, const string& results, int N ){
    string target = results + "/" + cases[0] + ".txt";
    ofstream target_file(target, std::ios_base::app);
    string a = "/" + cases[1] + cases[2];
    target_file <<a << endl;
    for(int i = 0; i < N; ++i){
        a = cases[1] + "/" + cases[1] + to_string(i) + ".jpg";
        target_file <<a<<endl;
        a = cases[2] + "/" + cases[2] + to_string(i) + ".jpg";
        target_file <<a<<endl;
    }

}

int main(int argc, char **argv) {
    const string path = "/home/nikita/ClionProjects/cascade_generator";
    const string results = "/trans_results";
    const int N = 10;

    write_training("cross", path ,results, N );
    write_training("circle", path ,results, N );
    write_training("rectangle", path ,results, N );
    //add bad
    vector<string> cross_bad = {"cross", "circle" ,"rectangle"};
    vector<string> cirle_bad = {"circle", "cross" ,"rectangle"};
    vector<string> rectangle_bad = {"rectangle", "cross" ,"circle"};
    add_bad(cross_bad, path + results, N);
    add_bad(cirle_bad, path + results, N);
    add_bad(rectangle_bad, path + results, N);
    return 0;

}
