#ifndef FRAME_HPP
#define FRAME_HPP
#include <vector>
#include <map>
#include <memory>

#include <opencv2/opencv.hpp>
using namespace std;

namespace CameraTracking {

class Frame
{
public:
    Frame(cv::Matx33f&_camera_matrix,cv::Mat& _distort_coeffs)
        :camera_matrix(_camera_matrix),distort_coeffs(_distort_coeffs)
    {
        frame_id = next_id++;
        is_keyframe = false;
    }
    void addObservation(int tag_id,pair<float,float> *pos_pixel)
    {
        //each tag has 4 point features
        for(int i = 0;i < 4; ++i)
        {
            int obj_pt_id = tag_id * 4 + i;
            cv::Mat pt_d = (cv::Mat_<cv::Point2f>(1,1)<<cv::Point2f(pos_pixel[i].first,
                                                                    pos_pixel[i].second));
            cv::Mat pt_u;
            cv::Mat new_camera_matrix = (cv::Mat_<float>(3,3)<<1.0,0.0,0.0,
                                                               0.0, 1.0,0.0,
                                                               0.0,0.0,1.0);
            cv::undistortPoints(pt_d,pt_u,camera_matrix,distort_coeffs,new_camera_matrix);
            observations_in_cam[obj_pt_id] = pair<float,float>(pt_u.at<cv::Point2f>(0,0).x,
                                                               pt_u.at<cv::Point2f>(0,0).y);
            observations_in_image[obj_pt_id] = pos_pixel[i];
        }
    }

public:
    map<int, pair<float,float> > observations_in_cam;//<obj_id,obs>
    map<int, pair<float,float> > observations_in_image;//<obj_id,obs>
    int frame_id;
    //frame world to camera
    cv::Mat R_w_c;
    cv::Mat t_w_c;
    bool is_keyframe;

private:
    static int next_id;
    cv::Matx33f camera_matrix;
    cv::Mat distort_coeffs;
};
typedef  std::shared_ptr<Frame> FramePtr;
typedef  const std::shared_ptr<Frame> FrameConstPtr;
int Frame::next_id = 0;

}//namespace CameraTracking

#endif // FRAME_HPP
