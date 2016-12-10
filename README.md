# 3D-Reconstruction from Motion

 * Captured images of the same scene from the same camera in different positions.
 * Extraction of Keypoints using the SURF feature detector.
 * Estimation of Fundamental and Essential Matrix.
 * Find Rotation and Translation matrix from the decomposition of the Essential Matrix.
 * Find 3D point cloud using the Linear Triangulation method.
 * Display the 3D points in the world frame using (PCL) Point Cloud Library. 

Youtube link:
https://www.youtube.com/watch?v=YmOfUyeRtk8&t=1s

## Build
./build.sh

## Run
./StructureFromMotion

(Give the number of corresponding points: 100)

## Visualize Point Cloud using PCL
pcl_viewer eg.pcd
