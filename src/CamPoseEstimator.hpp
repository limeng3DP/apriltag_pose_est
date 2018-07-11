#ifndef CAMPOSEESTIMATOR_HPP
#define CAMPOSEESTIMATOR_HPP
#include <vector>
#include <iostream>
#include "IOstreamColor.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <eigen3/Eigen/Eigen>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <ros/ros.h>

#include "Utility.hpp"
#include "Frame.hpp"
using namespace std;

namespace CameraTracking {

struct ReprojectionError3D
{
    ReprojectionError3D(double observed_u, double observed_v)
        :observed_u(observed_u), observed_v(observed_v)
        {}

    template <typename T>
    bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
    {
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        residuals[0] = xp - T(observed_u);
        residuals[1] = yp - T(observed_v);
        return true;
    }
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y)
    {
      return (new ceres::AutoDiffCostFunction<
              ReprojectionError3D, 2, 4, 3, 3>(
                new ReprojectionError3D(observed_x,observed_y)));
    }

    double observed_u;
    double observed_v;
};

struct ReprojectionErrorPixel
{
    ReprojectionErrorPixel(double observed_u, double observed_v)
        :observed_u(observed_u), observed_v(observed_v)
        {}

    template <typename T>
    bool operator()(const T* camera_params,const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
    {
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T fx = camera_params[0]; T fy = camera_params[1];
        T cx = camera_params[2]; T cy = camera_params[3];
        T k1 = camera_params[4]; T k2 = camera_params[5];
        T p1 = camera_params[6]; T p2 = camera_params[7];
        T k3 = camera_params[8];

        T x_2 = xp * xp;
        T y_2 = yp * yp;
        T r_2 = x_2 + y_2;
        T r_4 = r_2 * r_2;
        T r_6 = r_4 * r_2;

        T xp_d = xp * (T(1) + k1 * r_2 + k2 * r_4 + k3 * r_6) + T(2) * p1 * xp * yp + p2 * (r_2 + T(2) * x_2);
        T yp_d = yp * (T(1) + k1 * r_2 + k2 * r_4 + k3 * r_6) + p1 * (r_2 + T(2) * y_2) + T(2) * p2 * xp * yp;

        T u = xp_d * fx + cx;
        T v = yp_d * fy + cy;

        residuals[0] = u - T(observed_u);
        residuals[1] = v - T(observed_v);
        return true;
    }
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y)
    {
      return (new ceres::AutoDiffCostFunction<
              ReprojectionErrorPixel, 2, 9, 4, 3, 3>(
                new ReprojectionErrorPixel(observed_x,observed_y)));
    }
private:
    double observed_u;
    double observed_v;
};


struct Feat3D
{
    Feat3D(){x = y = z = 0.0;}
    Feat3D(float _x,float _y,float _z):x(_x),y(_y),z(_z){}
    float x;
    float y;
    float z;
};
class CameraPoseEstimator
{

public:
    CameraPoseEstimator(ros::NodeHandle &_nh):nh(_nh)
    {
        nh.param<float>("apriltag/tag_size",tag_size, 0.088);
        nh.param<float>("apriltag/tag_spacing",tag_spacing, 0.0264);
        nh.param<int>("apriltag/tags_row_num",tags_row_num, 6);
        nh.param<int>("apriltag/tags_col_num",tags_col_num, 6);

        nh.param<bool>("ba_refine",ba_refine, true);
        nh.param<int>("win_keyframe_size",ba_win_size, 10);
        vFrameBA.reserve(ba_win_size + 1);//ba window + current frame

        float half_tag_size = tag_size / 2.0;
        for(int row = 0; row < tags_row_num; ++row)
            for(int col = 0; col < tags_col_num;++col)
            {
                int tag_id = row * tags_col_num + col;
                float tag_center_pos_x = col * tag_size + (col + 1) * tag_spacing + half_tag_size;
                float tag_center_pos_y = row * tag_size + (row + 1) * tag_spacing + half_tag_size;
                //each tag has 4 obj point
                obj_pts_map[4*tag_id] = Feat3D(tag_center_pos_x - half_tag_size,
                                               tag_center_pos_y - half_tag_size,
                                               0.0);

                obj_pts_map[4*tag_id+1] = Feat3D(tag_center_pos_x + half_tag_size,
                                                 tag_center_pos_y - half_tag_size,
                                                 0.0);

                obj_pts_map[4*tag_id+2] = Feat3D(tag_center_pos_x + half_tag_size,
                                                 tag_center_pos_y + half_tag_size,
                                                 0.0);

                obj_pts_map[4*tag_id+3] = Feat3D(tag_center_pos_x - half_tag_size,
                                                 tag_center_pos_y + half_tag_size,
                                                 0.0);
            }
    }
    void set_intrinsic(cv::Matx33f& _camera_matrix)
    {
        camera_matrix = _camera_matrix;
    }
    void set_distortion(cv::Mat& _distort_coeff)
    {
        distort_matrix = _distort_coeff;
    }

    bool framePoseEstimae(FramePtr& pFrame)
    {

        vector<cv::Point3f> vPt3D;
        vector<cv::Point2f> vPt2D;
        vPt3D.reserve(obj_pts_map.size());
        vPt2D.reserve(obj_pts_map.size());

        map<int, pair<float,float> >&frame_obs = pFrame->observations_in_image;
        for(auto it_frame_obs = frame_obs.begin(),it_end = frame_obs.end();
            it_frame_obs != it_end; ++it_frame_obs)
        {
            int obj_pt_id = it_frame_obs->first;
            auto it_in_map = obj_pts_map.find(obj_pt_id);
            pair<float,float>&pt2D = it_frame_obs->second;
            Feat3D&pt3D = it_in_map->second;
            vPt3D.push_back(cv::Point3f(pt3D.x,pt3D.y,pt3D.z));
            vPt2D.push_back(cv::Point2f(pt2D.first,pt2D.second));
        }
        cv::Mat rvec,t;
        bool pnp_success = cv::solvePnP(vPt3D,vPt2D,camera_matrix,distort_matrix,rvec,t,false);
        if(!pnp_success)
        {
            cout<<RED<<"PnP solving failed!"<<NOCOLOR<<endl;
            return false;
        }
        cv::Rodrigues(rvec,pFrame->R_w_c);
        t.copyTo(pFrame->t_w_c);
        if(ba_refine)
        {
            localBundleAdjustment(pFrame);
        }
    }

private:
    bool localBundleAdjustment(FramePtr &pFrame)
    {
        Utility::ScopedChronoTime time("BA Optimize");
        if(vFrameBA.size() <= ba_win_size)
        {
            pFrame->is_keyframe = true;
            vFrameBA.push_back(pFrame);
            return true;
        }
        vFrameBA[vFrameBA.size() - 1] = pFrame;
        //preprocess
        double **rotations = NULL;
        double **translations = NULL;
        double *camera_params = new double[9];
        camera_params[0] = camera_matrix(0,0);//fx
        camera_params[1] = camera_matrix(1,1);//fy
        camera_params[2] = camera_matrix(0,2);//cx
        camera_params[3] = camera_matrix(1,2);//cy

        camera_params[4] = distort_matrix.ptr<float>(0)[0];//k1
        camera_params[5] = distort_matrix.ptr<float>(0)[1];//k2
        camera_params[6] = distort_matrix.ptr<float>(0)[2];//p1
        camera_params[7] = distort_matrix.ptr<float>(0)[3];//p2
        camera_params[8] = distort_matrix.ptr<float>(0)[4];//k3


        //ba_frame_size == ba_win_size + 1
        int ba_frame_size = vFrameBA.size();
        rotations = new double* [ba_frame_size];
        translations = new double* [ba_frame_size];
        for(int frame_index = 0; frame_index < ba_frame_size;++frame_index)
        {
            rotations[frame_index] = new double[4];
            translations[frame_index] = new double[3];

            //transformation from world to camera
            cv::Mat &trans = vFrameBA[frame_index]->t_w_c;
            translations[frame_index][0] = trans.ptr<float>(0)[0];
            translations[frame_index][1] = trans.ptr<float>(1)[0];
            translations[frame_index][2] = trans.ptr<float>(2)[0];
            Eigen::Matrix3d eigRot;
            cv::Mat &rot = vFrameBA[frame_index]->R_w_c;
            cv::cv2eigen(rot,eigRot);
            Eigen::Quaterniond quat(eigRot);
            rotations[frame_index][0] = quat.w();
            rotations[frame_index][1] = quat.x();
            rotations[frame_index][2] = quat.y();
            rotations[frame_index][3] = quat.z();
        }
        //full bundle adjustment
        ceres::Problem problemOptimizer;
        ceres::LocalParameterization* rot_local_parameterization = new ceres::QuaternionParameterization();
        for(int i = 0; i < ba_frame_size;++i)
        {
            problemOptimizer.AddParameterBlock(rotations[i],4,rot_local_parameterization);
            problemOptimizer.AddParameterBlock(translations[i],3);
        }

        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.5);
        double **features3D = new double* [obj_pts_map.size()];
        int feature_cnt = 0;
        for(auto it_obj_pts_map = obj_pts_map.begin(),it_end = obj_pts_map.end();
            it_obj_pts_map != it_end; ++it_obj_pts_map)
        {
            features3D[feature_cnt] = new double[3];
            Feat3D &obj_pt_pos = it_obj_pts_map->second;
            features3D[feature_cnt][0] = obj_pt_pos.x;
            features3D[feature_cnt][1] = obj_pt_pos.y;
            features3D[feature_cnt][2] = obj_pt_pos.z;
            problemOptimizer.AddParameterBlock(features3D[feature_cnt],3);
            problemOptimizer.SetParameterBlockConstant(features3D[feature_cnt]);

            ++feature_cnt;

        }
        feature_cnt = 0;
        for(auto it_obj_pts_map = obj_pts_map.begin(),it_end = obj_pts_map.end();
            it_obj_pts_map != it_end; ++it_obj_pts_map)
        {
            int obj_pt_id = it_obj_pts_map->first;
            for(int frame_index = 0;frame_index < ba_frame_size; ++frame_index)
            {
                FramePtr& currFrame = vFrameBA[frame_index];
                map<int, pair<float,float> >&frame_obs = currFrame->observations_in_image;
                auto it_frame_obs = frame_obs.find(obj_pt_id);
                if(it_frame_obs == frame_obs.end())
                    continue;
                pair<float,float>& feature_obs = it_frame_obs->second;
                ceres::CostFunction *cost_function = ReprojectionErrorPixel::Create(feature_obs.first,feature_obs.second);
                problemOptimizer.AddResidualBlock(cost_function,loss_function,camera_params,rotations[frame_index],
                                                  translations[frame_index],features3D[feature_cnt]);
            }
            ++feature_cnt;
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

        //options.minimizer_progress_to_stdout = true;
        options.max_solver_time_in_seconds = 0.6;
        ceres::Solver::Summary summary;
        ceres::Solve(options,&problemOptimizer,&summary);
        if(summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 1e-4)
        {
            cout<<GREEN<<"Bundle Adjustment finished!"<<NOCOLOR<<endl;
            //std::cout<<summary.FullReport()<<std::endl;
            //gather result
            //update camera poses
            for(int i = 0; i < ba_frame_size; ++i)
            {
                Eigen::Quaterniond q(rotations[i][0],
                                     rotations[i][1],
                                     rotations[i][2],
                                     rotations[i][3]);
                cv::Mat rot;
                cv::eigen2cv(q.toRotationMatrix(),rot);
                rot.convertTo(vFrameBA[i]->R_w_c,vFrameBA[i]->R_w_c.type());

                vFrameBA[i]->t_w_c.ptr<float>(0)[0] = translations[i][0];
                vFrameBA[i]->t_w_c.ptr<float>(1)[0] = translations[i][1];
                vFrameBA[i]->t_w_c.ptr<float>(2)[0] = translations[i][2];
            }
            //update camera parameters
            camera_matrix(0,0) = camera_params[0];//fx
            camera_matrix(1,1) = camera_params[1];//fy
            camera_matrix(0,2) = camera_params[2] ;//cx
            camera_matrix(1,2) = camera_params[3];//cy

            distort_matrix.ptr<float>(0)[0] = camera_params[4];//k1
            distort_matrix.ptr<float>(0)[1] = camera_params[5];//k2
            distort_matrix.ptr<float>(0)[2] = camera_params[6];//p1
            distort_matrix.ptr<float>(0)[3] = camera_params[7];//p2
            distort_matrix.ptr<float>(0)[4] = camera_params[8];//k3
        }
        else
        {
            cout<<RED<<"Ceres Optimize failed!"<<NOCOLOR<<endl;
            return false;
        }
        adjustOptimiseWindow();
    }

    void adjustOptimiseWindow()
    {
        //check whether choosing current frame as keyframe
        int N = vFrameBA.size();
        cv::Mat curr_R_c_w = vFrameBA[N - 1]->R_w_c.t();
        cv::Mat curr_t_c_w = -curr_R_c_w * vFrameBA[N - 1]->t_w_c;

        cv::Mat last_keyframe_R_c_w = vFrameBA[N-2]->R_w_c.t();
        cv::Mat last_keyframe_t_c_w = -last_keyframe_R_c_w * vFrameBA[N-2]->t_w_c;

        float delta_trans = cv::norm(last_keyframe_t_c_w - curr_t_c_w);
        cv::Mat delta_rot_mat = last_keyframe_R_c_w.t() * curr_R_c_w;
        cv::Mat delta_rod;
        cv::Rodrigues(delta_rot_mat,delta_rod);
        float delta_rot = cv::norm(delta_rod);

        const float pi = 3.141592653;
        if(delta_trans > 0.2 || delta_rot > 10.0/180.0 * pi)//meters, degree
        {
            vFrameBA[N-1]->is_keyframe = true;

            vector<FramePtr> tmpFrames;
            tmpFrames.resize(ba_win_size + 1);
            for(int i = 0; i < ba_win_size; ++i)
                tmpFrames[i] = vFrameBA[i + 1];//marginalizing old frame
            vFrameBA.swap(tmpFrames);
        }
    }
private:
    map<int,Feat3D > obj_pts_map;
    ros::NodeHandle nh;

    //camera intrinsic;
    cv::Matx33f camera_matrix;
    cv::Mat distort_matrix;

    float tag_size,tag_spacing;
    int tags_row_num;
    int tags_col_num;

    //from world to camera
    cv::Mat rot_last_frame;
    cv::Mat trans_last_frame;
    //bundle adjustment
    bool ba_refine;
    int ba_win_size;
    bool optimize_obj_pts = true;
    vector<FramePtr> vFrameBA;
    const float feat3D_moving_averaging = 0.995;

};

}//namespace CameraTracking

#endif // CAMPOSEESTIMATOR_HPP
