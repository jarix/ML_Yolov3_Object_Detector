/*-------------------------------------------------------------------*\

  NAME
   yolov3_object_detector


  DESCRIPTION
    YOLOv3 darknet inferencing for basic object detection.

  AUTHOR
    Jari Honkanen


\*-------------------------------------------------------------------*/

#include <iostream>
#include <fstream>
#include <numeric>

#include <boost/filesystem.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>


// Bounding Box for a detected object
typedef struct {
    int boxID; // unique identifier   
    int classID;  // Class ID 
    double confidence; // classification confidence level
    cv::Rect roi; // Region-Of-Interest in image coordinates
} BoundingBoxS;

// Yolo V3 Configuration Files
std::string yoloClassesFile = "../yolov3/coco.names";
std::string yoloConfigFile = "../yolov3/yolov3.cfg";
std::string yoloWeightsFile = "../yolov3/yolov3.weights";


int main (int argc, char *argv[])
{

    /*---------------------------------------------------------------*\
       Check command line parameters and config files
    \*---------------------------------------------------------------*/
    if (argc != 2) {
        std::cout << argv[0] << ": Yolov3 Object Detector Inference Example" << std::endl;
        std::cout << "Usage: " << argv[0] << " image_file" << std::endl;
        return EXIT_FAILURE;
    }

    // Check Image file
    boost::filesystem::path p(argv[1]);
    if (!is_regular_file(p)) {
        std::cerr << "*** ERROR: File '" << argv[1] << "' is not a image file" << std::endl;
        std::cerr << "Usage: " << argv[0] << " image_file" << std::endl;
        return EXIT_FAILURE;
    }

    std::string inputFileName = argv[1];

    // Check Existance of Yolo Classes File
    boost::filesystem::path pc(yoloClassesFile);
    if (!is_regular_file(pc)) {
        std::cerr << "*** ERROR: Yolo Classes File '" << yoloClassesFile << "' not found" << std::endl;
        return EXIT_FAILURE;
    }    

    // Check Existance of Yolo Config File
    boost::filesystem::path pg(yoloConfigFile);
    if (!is_regular_file(pg)) {
        std::cerr << "*** ERROR: Yolo Classes File '" << yoloConfigFile << "' not found" << std::endl;
        return EXIT_FAILURE;
    }   

    // Check Existance of Yolo Config File
    boost::filesystem::path pw(yoloWeightsFile);
    if (!is_regular_file(pw)) {
        std::cerr << "*** ERROR: Yolo Classes File '" << yoloWeightsFile << "' not found" << std::endl;
        return EXIT_FAILURE;
    }   

    /*---------------------------------------------------------------*\
       Read in and prepare image
    \*---------------------------------------------------------------*/
    cv::Mat imgOriginal;
    cv::Mat imgUse;
    
    imgOriginal = cv::imread(argv[1]);
    //cv::resize(imgOriginal, imgUse, cv::Size(), 0.4, 0.4, cv::INTER_CUBIC); 
    imgUse = imgOriginal.clone();

    // Display original image
    std::string winName1 = "Input Image: " + inputFileName;
    cv::namedWindow( winName1, 1 );
    cv::imshow( winName1, imgUse );
    //cv::waitKey(0); // wait for key to be pressed

    /*---------------------------------------------------------------*\
       Prep YOLO detector
    \*---------------------------------------------------------------*/
    std::vector<std::string> classes;
    std::ifstream ifClasses(yoloClassesFile.c_str());
    std::string oneLine;
    while (std::getline(ifClasses, oneLine)) {
        classes.push_back(oneLine);
    }
    std::cout << "Loaded " << classes.size() << " classes from " << yoloClassesFile << std::endl;

    // Load network, use CPU for inference for keeping setup simple
    cv::dnn::Net yoloNet = cv::dnn::readNetFromDarknet(yoloConfigFile, yoloWeightsFile);
    yoloNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    yoloNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Create input blob from image
    cv::Mat inputBlob;
    double scaleFactor = 1/255.0;
    //cv::Size size = cv::Size(320, 320);
    //cv::Size size = cv::Size(416, 416);
    cv::Size size = cv::Size(608, 608);
    cv::Scalar mean = cv::Scalar(0,0,0);
    bool swapRB = false;
    bool crop = false;
    cv::dnn::blobFromImage(imgUse, inputBlob, scaleFactor, size, mean, swapRB, crop);

    /*---------------------------------------------------------------*\
       Figure out Output Names
    \*---------------------------------------------------------------*/
    // Get names of all layers
    std::vector<cv::String> layerNames = yoloNet.getLayerNames();
    // get indices for layers with unconnected outputs
    std::vector<int> outputLayers = yoloNet.getUnconnectedOutLayers();

    std::vector<cv::String> outputNames;
    outputNames.resize(outputLayers.size());
    // Retrieve names of output layers
    for (size_t i = 0; i < outputLayers.size(); i++) {
        outputNames[i] = layerNames[outputLayers[i] - 1];
    } 

    // for( const auto& value: outputNames) {
    //     std::cout << value << "\n";
    // }

    /*---------------------------------------------------------------*\
       Forward Pass thru Network
    \*---------------------------------------------------------------*/
    std::vector<cv::Mat> yoloNetOutput;
    yoloNet.setInput(inputBlob);
    yoloNet.forward(yoloNetOutput, outputNames);

    /*---------------------------------------------------------------*\
       Process Results, create bounding boxes if confidence threshold
       is exceeded
    \*---------------------------------------------------------------*/
    float confidenceThreshold = 0.2;  
    std::vector<int> classIds;
    std::vector<float> confidenceValues;
    std::vector<cv::Rect> boundingBoxes;

    std::cout << "Detector Results:" << std::endl;
    std::cout << "-----------------" << std::endl;

    for (size_t i = 0; i < yoloNetOutput.size(); i++)
    {
        float *data = (float *)yoloNetOutput[i].data;

        for (int j = 0; j < yoloNetOutput[i].rows; j++, data += yoloNetOutput[i].cols) 
        {
            cv::Mat scores = yoloNetOutput[i].row(j).colRange(5, yoloNetOutput[i].cols);
            cv::Point classId;
            double confidenceVal;

            // Retrieve the maximum score and save if it exceeds confidence threshold
            cv::minMaxLoc(scores, 0, &confidenceVal, 0, &classId);
            if (confidenceVal > confidenceThreshold) {
                cv::Rect bbox;
                // get center point 
                int cx, cy;
                cx = (int)(data[0] * imgUse.cols);
                cy = (int)(data[1] * imgUse.rows);
                bbox.width = (int)(data[2] * imgUse.cols);
                bbox.height = (int)(data[3] * imgUse.rows);
                bbox.x = cx - bbox.width/2; // left
                bbox.y = cy - bbox.height/2; // top

                std::cout << "Bounding Box: (" << bbox.x << "," << bbox.y << "," << bbox.width << "," << bbox.height << ")" << std::endl;
                std::cout << "     class Id: " << classId.x << std::endl;
                std::cout << "   confidence: " << confidenceVal << std::endl << std::endl;
                
                boundingBoxes.push_back(bbox);
                classIds.push_back(classId.x);
                confidenceValues.push_back((float)confidenceVal);
            } 
        }
    }
    
    /*---------------------------------------------------------------*\
       Do Non-Maximal Suppression
    \*---------------------------------------------------------------*/
    float nmsThreshold = 0.4;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boundingBoxes, confidenceValues, confidenceThreshold, nmsThreshold, indices);
    std::vector<BoundingBoxS> nmsBBoxes;

    for (auto it = indices.begin(); it != indices.end(); it++) 
    {
        BoundingBoxS nmsBBox;
        nmsBBox.roi = boundingBoxes[*it];
        nmsBBox.classID = classIds[*it];
        nmsBBox.confidence = confidenceValues[*it];
        nmsBBox.boxID = (int)nmsBBoxes.size();

        nmsBBoxes.push_back(nmsBBox);
    }

    std::cout << "Detector Results after Non-Maximal Supression:" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    for (auto it = nmsBBoxes.begin(); it != nmsBBoxes.end(); it++) {
        std::cout << "Bounding Box[" << it->boxID << "]: (" << it->roi.x << "," << it->roi.y << "," << it->roi.width << "," << it->roi.height << ")" << std::endl;
        std::cout << "     class Id: " << it->classID << std::endl;
        std::cout << "   confidence: " << it->confidence << std::endl << std::endl;
    }

    /*---------------------------------------------------------------*\
       Display results in the image
    \*---------------------------------------------------------------*/
    cv::Mat imgResults = imgUse.clone();
    for (auto it = nmsBBoxes.begin(); it != nmsBBoxes.end(); it++) 
    {
        // Draw Bounding Box
        cv::rectangle(imgResults, cv::Point(it->roi.x, it->roi.y), cv::Point(it->roi.x + it->roi.width, it->roi.y + it->roi.height), cv::Scalar(255,0,255), 2);

        // Create and display object label
        std::string objectLabel = classes[it->classID] + ":" + cv::format("%.3f", it->confidence);
        int baseLine;
        cv::Size textSize = cv::getTextSize(objectLabel, cv::FONT_ITALIC, 0.5, 1, &baseLine);
        int topLine = std::max(it->roi.y, textSize.height);
        cv::rectangle(imgResults, cv::Point(it->roi.x, topLine - round(1.5 * textSize.height)), cv::Point(it->roi.x + round(1.5 * textSize.width), topLine + baseLine), cv::Scalar(255,0,255), cv::FILLED);
        cv::putText(imgResults, objectLabel, cv::Point(it->roi.x, topLine), cv::FONT_ITALIC, 0.75, cv::Scalar(255, 255, 255), 1);
    }

    // Display results image with bounding boxes
    std::string winName2 = "Yolo v3 Object Detection Results: " + inputFileName;
    cv::namedWindow( winName2, 1 );
    cv::imshow( winName2, imgResults );
    cv::waitKey(0); // wait for key to be pressed

    return EXIT_SUCCESS;
}

