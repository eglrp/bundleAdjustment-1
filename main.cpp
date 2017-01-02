/*
 * 不懂: 总误差函数计算时, 一个误差项怎么影响另一个误差项?????
 * 怎么建立约束????? 怎么把噪声加入?????
 * Bundle Adjustment
 * 读取两张图像 -> 特征匹配 -> 建图 -> 使用g2o工具minimum error -> 计算X(特征点物理坐标),R,T(相机姿态)
 * 典型的BA问题
 */
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/concept_check.hpp>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;
using namespace cv;

/* camera parameter */
double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;

/* 寻找两个图像的matching points
 * input: two images
 * output: matching (u,v) points
 */
int findCorrespondingPoints( const Mat &img1, const Mat &img2, vector<Point2f> &points1, vector<Point2f> &points2)
{
        ORB orb;
        vector<KeyPoint> kp1, kp2;
        Mat desp1, desp2;
        orb(img1, Mat(), kp1, desp1); // 提取符合orb性质的点
        orb(img2, Mat(), kp2, desp2);
        imshow("1", img1);
        imshow("2", img2);
        waitKey(0);
        cout << "Find " << kp1.size() << " " << kp2.size() << endl;

        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce-Hamming" ); // Matching的计算方式
        double knn_match_ratio = 0.8; // threshold
        vector< vector<DMatch> > matches_knn; // 存储Match的结果
        matcher->knnMatch( desp1, desp2, matches_knn, 2); // 匹配
        vector<DMatch> matches; // 存储符合要求的Match的结果
        for ( size_t i=0; i<matches_knn.size(); i++)
        {
                if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance)
                    matches.push_back(matches_knn[i][0]);
        }

        if ( matches.size() <= 20)
            return false;

        for ( auto m:matches )
        {
                points1.push_back( kp1[m.queryIdx].pt );
                points2.push_back( kp2[m.trainIdx].pt );
        }
        return true;
}


int main( int argc, char **argv)
{
        /* 读取图像 */
        Mat image1 = imread("/home/chiao/documents/bundleAdjustment/data/1.png");
        Mat image2 = imread("/home/chiao/documents/bundleAdjustment/data/2.png");

        /* 特征匹配 */
        vector<Point2f> pts1, pts2;
        if ( findCorrespondingPoints( image1, image2, pts1, pts2) == false)
        {
                cout << "Not enough matching points !!!" << endl;
                return 0;
        }
        cout << "Find " << pts1.size() << " matching points !!!" << endl;

        /* 使用g2o的部分
          * 构造求解器 -> 设置L-M下降算法 -> 构建两帧图像的节点 -> 使用相机参数 -> 构建边 ->
          * 优化 -> 输出变换矩阵与特征点位置 -> 输出inlier
          */
        g2o::SparseOptimizer optimizer; // 维护optimizer

        /* 优化方法 <- Solver <- 求解器 */
        // 使用Cholmod的线性方程求解器
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> ();
        // 6*3 的参数
        g2o::BlockSolver_6_3 *block_solver = new g2o::BlockSolver_6_3(linearSolver);
        // L-M下降法
        g2o::OptimizationAlgorithmLevenberg *algorithm = new g2o::OptimizationAlgorithmLevenberg(block_solver);

        optimizer.setAlgorithm( algorithm );
        optimizer.setVerbose( false );

        /* 添加两个位姿节点 */
        for ( int i=0; i<2; i++)
        {
                g2o::VertexSE3Expmap *v = new g2o::VertexSE3Expmap();
                v->setId(i); // 节点编号
                if ( i == 0 ) v->setFixed(true); // 第一个点固定为零
                // 预设值为单位Pose, 因为我们不知道任何信息
                v->setEstimate( g2o::SE3Quat() ); // 相机姿态的初值
                optimizer.addVertex( v );
        }

        /* 添加pts.size()个特征点的节点 */
        for ( size_t i=0; i<pts1.size(); i++)
        {
                g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ(); // 点的类型是空间坐标点
                v->setId( 2+i );
                // 由于深度不知道, 只能把深度设置为1 1*[z, 1]^T = C[X, Y, Z]^T
                double z = 1;
                double x = ( pts1[i].x - cx ) * z / fx;
                double y = ( pts1[i].y - cy ) * z / fy;
                v->setMarginalized(true);
                v->setEstimate( Eigen::Vector3d(x,y,z) ); // 特征点的初值
                optimizer.addVertex( v );
        }

        /* 使用相机参数 */
        g2o::CameraParameters *camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
        camera->setId(0);
        optimizer.addParameter( camera );

        /* 构建边 */
        vector<g2o::EdgeProjectXYZ2UV *> edges;
        for ( size_t i=0; i<pts1.size(); i++)
        {
                // 连边 边的类型是重投影误差
                g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
                edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ *> (optimizer.vertex(i+2)));
                edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap *> (optimizer.vertex(0)));

                edge->setMeasurement( Eigen::Vector2d(pts1[i].x, pts1[i].y)); //设置边的观测信息
                edge->setInformation( Eigen::Matrix2d::Identity()); // 信息矩阵作为不确定性的度量
                edge->setParameterId( 0, 0 );
                edge->setRobustKernel( new g2o::RobustKernelHuber() ); // 设置核函数
                optimizer.addEdge( edge );
                edges.push_back( edge );
        }

        for ( size_t i=0; i<pts2.size(); i++)
        {
                g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
                edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ *> (optimizer.vertex(i+2)));
                edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap *> (optimizer.vertex(1)));

                edge->setMeasurement( Eigen::Vector2d(pts2[i].x, pts2[i].y)); // ?????
                edge->setInformation( Eigen::Matrix2d::Identity());
                edge->setParameterId( 0, 0);
                edge->setRobustKernel( new g2o::RobustKernelHuber() );
                optimizer.addEdge( edge );
                edges.push_back( edge );
        }

        /* 优化 */
        cout << "Begin optimizing ..." << endl;
        optimizer.setVerbose(true);
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cout << "Finish optimizing !!!" << endl;

        /* 输出第二帧图像的姿态 (因为第一帧姿态为初始值所以这个姿态可以为变换矩阵) */
        g2o::VertexSE3Expmap *v = dynamic_cast<g2o::VertexSE3Expmap *> (optimizer.vertex(0) );
        Eigen::Isometry3d pose = v->estimate(); // 变换矩阵T
        cout << "Pose1 = " << endl << pose.matrix() << endl;
        v = dynamic_cast<g2o::VertexSE3Expmap *> (optimizer.vertex(1) );
        pose = v->estimate();
        cout << "Pose2 = " << endl << pose.matrix() << endl;

        /* 所有特征点的世界坐标系位置 */
        for ( size_t i=0; i<pts1.size(); i++)
        {
                g2o::VertexSBAPointXYZ *v = dynamic_cast<g2o::VertexSBAPointXYZ *> (optimizer.vertex(i+2));
                cout << "Vertex Id = " << i+2 << " and Position = ";
                Eigen::Vector3d position = v->estimate();
                cout << position(0) << ", " << position(2) << endl;
        }

        /* 估计inlier的个数 */
        int inliers = 0;
        for ( auto e:edges )
        {
                e->computeError();
                // 某个边的误差项太大, 说明这个边与其他边不符合, 因为优化是对所有边作用的
                if (e->chi2() > 1)
                {
                        cout << "error = " << e->chi2() << endl;
                }
                else
                {
                        inliers++;
                }
        }
        cout << "inliers in total points : " << inliers << "/" << pts1.size() + pts2.size() << endl;
        optimizer.save("BundleAdjustment.g2o");
        return 0;
}

















