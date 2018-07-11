#include <iostream>
#include "IOstreamColor.h"
#include <map>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>

#include <apriltags/TagDetector.h>
#include <apriltags/TagDetection.h>
#include <apriltags/Tag36h11.h>
#include <apriltags/Tag36h9.h>
#include <apriltags/Tag25h9.h>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>

#include "Frame.hpp"
#include "CamPoseEstimator.hpp"
using namespace std;
namespace CameraTracking  {
class TrackingSystem
{
public:
    TrackingSystem(ros::NodeHandle &_nh)
        :nh(_nh),tag_codes(AprilTags::tagCodes36h11),pose_estimator(_nh)
    {
        float fx,fy,cx,cy;
        float k1,k2,k3,p1,p2;
        nh.param<int>("resolution/height",image_height,480);
        nh.param<int>("resolution/width",image_width,640);

        nh.param<float>("intrinsic/fx",fx, 1.0);
        nh.param<float>("intrinsic/fy",fy, 1.0);
        nh.param<float>("intrinsic/cx",cx, image_width/2);
        nh.param<float>("intrinsic/cy",cy, image_height/2);
        nh.param<float>("distortion/k1",k1, 0.0);
        nh.param<float>("distortion/k2",k2, 0.0);
        nh.param<float>("distortion/p1",p1, 0.0);
        nh.param<float>("distortion/p2",p2, 0.0);
        nh.param<float>("distortion/k3",k3, 0.0);

        nh.param<float>("image/scale/width",scale_x, 1);
        nh.param<float>("image/scale/height",scale_y, 1);
        pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/camera/Pose",3);

        //No amplification;
        if(scale_x < 1.0)
            scale_x = 1.0;
        if(scale_y < 1.0)
            scale_y = 1.0;
        distort_coeffs = (cv::Mat_<float>(1,5)<< k1,k2,p1,p2,k3);
        camera_matrix = cv::Matx33f(fx/scale_x,0.0,       cx/scale_x,
                                    0.0,       fy/scale_y,cy/scale_y,
                                    0.0,       0.0,       1.0);
        pose_estimator.set_intrinsic(camera_matrix);
        pose_estimator.set_distortion(distort_coeffs);
        string tag_code_str;
        nh.param<string>("apriltag/tag_codes",tag_code_str,"");
        if(tag_code_str.empty())
            tag_codes = AprilTags::tagCodes36h11;//default codes
        else
        {
            if(tag_code_str == "36h9")
                tag_codes = AprilTags::tagCodes36h9;
            else if(tag_code_str == "25h9")
                tag_codes = AprilTags::tagCodes25h9;
            //else if ...
        }
        tag_detector_ptr = new AprilTags::TagDetector(tag_codes);
        ros::Subscriber image_sub = nh.subscribe("image",3,&TrackingSystem::imageCallback,this);
        ros::spin();
    }
private:
    void imageCallback(const sensor_msgs::ImageConstPtr &img_ptr)
    {
        cv_bridge::CvImageConstPtr bridge_img = cv_bridge::toCvShare(img_ptr,sensor_msgs::image_encodings::MONO8);
        cv::Mat image = bridge_img->image;
        if(image.channels() == 3)
        {
            if(is_rgb)
                cv::cvtColor(image,image,CV_RGB2GRAY);

            else
                cv::cvtColor(image,image,CV_BGR2GRAY);
        }
        else if(image.channels() == 4)
        {
            if(is_rgb)
                cv::cvtColor(image,image,CV_RGBA2GRAY);

            else
                cv::cvtColor(image,image,CV_BGRA2GRAY);
        }

        cv::Size dst_size(int(image_width / scale_x),int(image_height / scale_y));
        cv::resize(image,image,dst_size,CV_INTER_LINEAR);
        vector<AprilTags::TagDetection> detections = tag_detector_ptr->extractTags(image);
        if(detections.empty())
        {
            cout<<RED<<"No april tag detected!"<<NOCOLOR<<endl;
            return;
        }
        bool draw = true;
        if(draw)
        {
            for(int i = 0;i < detections.size();++i)
            {
                AprilTags::TagDetection &det = detections[i];
                det.draw(image);
            }
            cv::imshow("detection",image);
            cv::waitKey(2);
        }

        if(detections.size() == 1)
            cout<<YELLOW<<"Warning: Only one tag detected! Recovered pose may be unstable!"<<NOCOLOR<<endl;
        FramePtr frame = createCameraFrame(detections);
        if(frame == NULL || !pose_estimator.framePoseEstimae(frame))
        {
            cout<<YELLOW<<"Null pose published!"<<NOCOLOR<<std::endl;
            return;
        }
        else
        {
            cv::Mat R_c_w = frame->R_w_c.t();
            cv::Mat t_c_w = - R_c_w * frame->t_w_c;

            Eigen::Matrix<double,3,3> rot;
            cv::cv2eigen(R_c_w,rot);

            Eigen::Quaterniond quat(rot);

            geometry_msgs::PoseStamped pose;
            pose.header.frame_id = bridge_img->header.frame_id;
            pose.header.stamp = bridge_img->header.stamp;

            pose.pose.position.x = t_c_w.ptr<float>(0)[0];
            pose.pose.position.y = t_c_w.ptr<float>(1)[0];
            pose.pose.position.z = t_c_w.ptr<float>(2)[0];

            pose.pose.orientation.w = quat.w();
            pose.pose.orientation.x = quat.x();
            pose.pose.orientation.y = quat.y();
            pose.pose.orientation.z = quat.z();
            pose_pub.publish(pose);
        }
    }
    FramePtr createCameraFrame(vector<AprilTags::TagDetection>& detections)
    {
        FramePtr pframe(new Frame(camera_matrix,distort_coeffs));
        for(int i = 0; i < detections.size();++i)
        {
            AprilTags::TagDetection &detection = detections[i];
            if(!detection.good)
                continue;
            pframe->addObservation(detection.id,detection.p);
        }
        if(pframe->observations_in_image.size() < 4)
        {
            cout<<RED<<"Not proper view towards apriltags pattern!"<<NOCOLOR<<endl;
            return NULL;
        }
        
        return pframe;
    }
private:
    bool is_rgb;
    bool ba_refine;
    ros::NodeHandle nh;
    AprilTags::TagDetector *tag_detector_ptr;
    AprilTags::TagCodes tag_codes;
    CameraPoseEstimator pose_estimator;
    cv::Mat distort_coeffs;
    cv::Matx33f camera_matrix;
    int image_height;
    int image_width;
    float scale_x,scale_y;
    ros::Publisher pose_pub;
};

}//namespace CameraTracking

int main(int argc,char** argv)
{
    ros::init(argc,argv,"apriltag_pose_est_node");
    ros::NodeHandle nh("~");
    CameraTracking::TrackingSystem system(nh);
}

