/*
 * 3D Reconstruction from Motion
 * Created on : April, 2014
 * Author: Alexandros Lioulemes
 * OpenCV - PCL - C++
 * Captured images of the same scene from the same camera in different position.
 * Extraction of Keypoints using the SURF feature detector.
 * Estimation of Fundamental and Essen1al Matrix.
 * Find Rotation and Translation matrix from the decomposi1on of the Essential Matrix.
 * Find 3D points cloud from the Linear Triangulation method .
 * Display the 3D points in the world using (PCL) Point Cloud Libraty.
 */

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#ifdef USE_EIGEN
#include <Eigen/Eigen>
#endif

#define PI 3.1415927
double flag_size = 0.1;
int surf_parameter = 1500;

using namespace cv;
using namespace std;

Mat_<double> LinearLSTriangulation(Point3d u,Matx34d P,Point3d u1,Matx34d P1);
double reprojection_error(vector<Point2f> a, vector<Point2f> b);
void printIplImage(const IplImage* src, int row, int col, int RGB[]);

static double compareSURFDescriptors( const float* d1, const float* d2, double best, int length ){
    
	double total_cost = 0;
    assert( length % 4 == 0 );
    for( int i = 0; i < length; i += 4 ){
        double t0 = d1[i] - d2[i];
        double t1 = d1[i+1] - d2[i+1];
        double t2 = d1[i+2] - d2[i+2];
        double t3 = d1[i+3] - d2[i+3];
        total_cost += t0*t0 + t1*t1 + t2*t2 + t3*t3;
        if( total_cost > best )
            break;
    }
	//printf("total_cost=%g\n",total_cost);
    return total_cost;
}

static int naiveNearestNeighbor( const float* vec, int laplacian, const CvSeq* model_keypoints, const CvSeq* model_descriptors ){
	
    int length = (int)(model_descriptors->elem_size/sizeof(float));
    int i, neighbor = -1;
    double d, dist1 = 1e6, dist2 = 1e6;
    CvSeqReader reader, kreader;
    cvStartReadSeq( model_keypoints, &kreader, 0 );
    cvStartReadSeq( model_descriptors, &reader, 0 );
	
    for( i = 0; i < model_descriptors->total; i++ ){
        const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
        const float* mvec = (const float*)reader.ptr;
        CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        if( laplacian != kp->laplacian )
            continue;
        d = compareSURFDescriptors( vec, mvec, dist2, length );
        if( d < dist1 ){
            dist2 = dist1;
            dist1 = d;
            neighbor = i;
        }
        else if ( d < dist2 )
            dist2 = d;
    }
	//  2) flag_size variable is important in order to take as much pairs point I want
    if ( dist1 < flag_size*dist2 ){
        return neighbor;
	}
    return -1;
}

static void findPairs(const CvSeq* objectKeypoints, const CvSeq* objectDescriptors, const CvSeq* imageKeypoints, const CvSeq* imageDescriptors, vector<int>& ptpairs ){
	
	printf("findPairs\n");
    
	int i;
	//Returns the current reader position
    CvSeqReader reader, kreader;
    //Initializes process of sequential reading from sequence
	cvStartReadSeq( objectKeypoints, &kreader );
    cvStartReadSeq( objectDescriptors, &reader );
    //Removes all elements from the vector (which are destroyed),
	//leaving the container with a size of 0.
	ptpairs.clear();
    for( i = 0; i < objectDescriptors->total; i++ ){
        const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
        const float* descriptor = (const float*)reader.ptr;
        CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        int nearest_neighbor = naiveNearestNeighbor( descriptor, kp->laplacian, imageKeypoints, imageDescriptors );
		//printf("nearest_neightbor =%d\n",nearest_neighbor);
		if( nearest_neighbor >= 0 ){
            ptpairs.push_back(i);
            ptpairs.push_back(nearest_neighbor);
        }
    }
}

int main(int argc, char** argv){
    
	const char* image1_filename = argc == 3 ? argv[1] : "image1.jpg";
    const char* image2_filename = argc == 3 ? argv[2] : "image2.jpg";
    
	cv::initModule_nonfree();
	
	int number_corr=0;
	printf("Give the number of corresponding points:\n");
	scanf("%d",&number_corr);
	
	IplImage* image1 = cvLoadImage( image1_filename, CV_LOAD_IMAGE_GRAYSCALE );
    IplImage* image2 = cvLoadImage( image2_filename, CV_LOAD_IMAGE_GRAYSCALE );
	
	IplImage* image1rgb = cvLoadImage( image1_filename);
	
	
	if( !image1 || !image2 )
    {
        fprintf( stderr, "Can not load %s and/or %s\n",image1_filename, image2_filename);
        exit(-1);
    }
    
	CvMemStorage* storage = cvCreateMemStorage(0);
	
	cvNamedWindow("first", 1);
	cvNamedWindow("second", 1);
    cvNamedWindow("Correspondences", 1);
    
	static CvScalar colors[] = {
        {{0,0,255}},
        {{0,128,255}},
        {{0,255,255}},
        {{0,255,0}},
        {{255,255,0}},
        {{0,255,255}},
        {{255,0,0}},
        {{255,0,255}},
        {{255,255,255}} };
	
	//RGB --> GRAY
	IplImage* image1_color = cvCreateImage(cvGetSize(image1), 8, 3);
    cvCvtColor( image1, image1_color, CV_GRAY2BGR );
	IplImage* image2_color = cvCreateImage(cvGetSize(image2), 8, 3);
    cvCvtColor( image2, image2_color, CV_GRAY2BGR );
	
	CvSeq* image1_Keypoints = 0, *image1_Descriptors = 0;
    CvSeq* image2_Keypoints = 0, *image2_Descriptors = 0;
	int i;
	
	CvSURFParams params = cvSURFParams(surf_parameter, 1);
	
    double tt = (double)cvGetTickCount();
	
	cvExtractSURF( image1, 0, &image1_Keypoints, &image1_Descriptors, storage, params );
    //------------------------------------------------------------------------------------
	
	//printf("Image1 Keypoints: %d\n", image1_Keypoints->total);
	for (i=0; i<image1_Keypoints->total; i++) {
		CvSURFPoint* r = (CvSURFPoint*) cvGetSeqElem(image1_Keypoints,i);
		//printf("r[%d]=\t%g\t%g\t%d\t%d\n",i,r->dir,r->hessian,r->laplacian,r->size);
		//printf("pos=[%d,%d]\n",cvRound(r->pt.x),cvRound(r->pt.y));
	}
	
	//------------------------------------------------------------------------------------
    cvExtractSURF( image2, 0, &image2_Keypoints, &image2_Descriptors, storage, params );
	//------------------------------------------------------------------------------------
    //printf("\nImage2 Keypoints: %d\n", image2_Keypoints->total);
	for (i=0; i<image2_Keypoints->total; i++) {
		CvSURFPoint* r = (CvSURFPoint*) cvGetSeqElem(image2_Keypoints,i);
		//printf("r[%d]=\t%g\t%g\t%d\t%d\n",i,r->dir,r->hessian,r->laplacian,r->size);
		//printf("pos=[%d,%d]\n",cvRound(r->pt.x),cvRound(r->pt.y));
	}
	
    tt = (double)cvGetTickCount() - tt;
	
    printf( "\nExtraction time = %gms\n", tt/(cvGetTickFrequency()*1000.));
	
	CvPoint src_corners[4] = {{0,0}, {image1->width,0}, {image1->width, image1->height}, {0, image1->height}};
    CvPoint dst_corners[4];
	
    IplImage* correspond = cvCreateImage( cvSize(image2->width, image1->height+image2->height), 8, 1 );
	
    cvSetImageROI( correspond, cvRect( 0, 0, image1->width, image1->height ) );
    cvCopy( image1, correspond );
    cvSetImageROI( correspond, cvRect( 0, image1->height, correspond->width, correspond->height ) );
    cvCopy( image2, correspond );
    cvResetImageROI( correspond );
	
	printf("Using approximate nearest neighbor search\n");
	///////////////////////////////////////////////////////////////////
    vector<int> ptpairs;
	
	int temp=0;
	while (temp < number_corr) {
		findPairs( image1_Keypoints, image1_Descriptors, image2_Keypoints, image2_Descriptors, ptpairs );
		temp = (int)(ptpairs.size()/2);
		flag_size = flag_size+0.01;
	}
	
    
    //-------------------------------------------------------------------
	//DRAW CORRESPONDING LINES
	//-------------------------------------------------------------------
    char cntr_s1[5];
	char cntr_s2[5];
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_PLAIN|CV_FONT_ITALIC,1,1,1,1);
	CvPoint center1;
	CvPoint center2;
    printf("ptpairs = %d.\n",(int)(ptpairs.size()/2));
    for( i = 0; i < (int)ptpairs.size(); i += 2 )
    {
        CvSURFPoint* r1 = (CvSURFPoint*)cvGetSeqElem( image1_Keypoints, ptpairs[i] );
        CvSURFPoint* r2 = (CvSURFPoint*)cvGetSeqElem( image2_Keypoints, ptpairs[i+1] );
        // draw the line
		printf("Pairs:\tdir\thess\tlap\tsize\n");
		printf("%d=\t%g\t%g\t%d\t%d\n",i,r1->dir,r1->hessian,r1->laplacian,r1->size);
		printf("%d=\t%g\t%g\t%d\t%d\n",i+1,r2->dir,r2->hessian,r2->laplacian,r2->size);
		printf("\n");
		center1.x = cvRound(r1->pt.x);
		center1.y = cvRound(r1->pt.y);
		snprintf(cntr_s1, sizeof(cntr_s1),"%d",i);
		center2.x = cvRound(r2->pt.x);
		center2.y = cvRound(r2->pt.y+image1->height);
		snprintf(cntr_s2, sizeof(cntr_s2),"%d",i+1);
		cvCircle(correspond,center1,1,colors[0],1);
		cvPutText(correspond,cntr_s1,center1,&font,colors[4]);
		cvCircle(correspond,center2,1,colors[0],1);
		cvPutText(correspond,cntr_s2,center2,&font,colors[4]);
		cvLine( correspond, cvPointFrom32f(r1->pt),
			   cvPoint(cvRound(r2->pt.x), cvRound(r2->pt.y+image1->height)), colors[4] );
    }
    
    
	//-------------------------------------------------------------------
	//DRAW KEYPOINTS
	//-------------------------------------------------------------------
	cvShowImage( "Correspondences", correspond );
	// draw keypoint for object
	for( i = 0; i < image1_Keypoints->total; i++ )
    {
        CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( image1_Keypoints, i );
        CvPoint center;
        int radius;
        center.x = cvRound(r->pt.x);
        center.y = cvRound(r->pt.y);
        radius = cvRound(r->size*1.2/9.*2);
        cvCircle( image1_color, center, radius, colors[0], 1, 8, 0 );
    }
    cvShowImage( "first", image1_color );
	// draw keypoints for image
	for( i = 0; i < image2_Keypoints->total; i++ )
    {
        CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( image2_Keypoints, i );
        CvPoint center;
        int radius;
		// ginte strogylopoihsh
        center.x = cvRound(r->pt.x);
        center.y = cvRound(r->pt.y);
        radius = cvRound(r->size*1.2/9.*2);
        cvCircle( image2_color, center, radius, colors[2], 1, 8, 0 );
    }
    cvShowImage( "second", image2_color );
	
	printf("flag_size = %g\n",flag_size);
	//-------------------------------------------------------------------
    
    
    
    //================================================================
	// Here I have to transform the vector to points
    int i1=0;
    int numPoints=number_corr;
    int iter=0;
    iter = number_corr;
    CvMat* points1;
	CvMat* points2;
    CvMat* status;
	CvMat* fundMatr;
	status = cvCreateMat(1,numPoints,CV_8UC1);
	points1 = cvCreateMat(2,numPoints,CV_32FC1);
	points2 = cvCreateMat(2,numPoints,CV_32FC1);
    CvSize imageSize = cvGetSize(image1);
	
    vector<Point2f> U1, U2;
    
    
	for( i = 0; i < 2*iter; i += 2 )
    {
        CvSURFPoint* r1 = (CvSURFPoint*)cvGetSeqElem( image1_Keypoints, ptpairs[i] );
        CvSURFPoint* r2 = (CvSURFPoint*)cvGetSeqElem( image2_Keypoints, ptpairs[i+1] );
        
        
        
		center1.x = cvRound(r1->pt.x);
		center1.y = cvRound(r1->pt.y);
		printf("Point1,%d=(%d,%d)\n",i1,center1.x,center1.y);
        
        U1.push_back(center1);
        U2.push_back(center2);
        
		
		center2.x = cvRound(r2->pt.x);
		center2.y = cvRound(r2->pt.y+image1->height);
		printf("Point2,%d=(%d,%d)\n",i1,center2.x,center2.y-imageSize.height);
		
		cvSetReal2D(points1,0,i1,center1.x);
		cvSetReal2D(points1,1,i1,center1.y);
		
		cvSetReal2D(points2,0,i1,center2.x);
		cvSetReal2D(points2,1,i1,center2.y-imageSize.height);
		
		i1++;
		
	}
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // REPORT 3 Implementation
    // 03/23/2014
    // Alexandros Lioulemes
    
    
    // PRINT CORRESPONDECE POINTS
	
	printf("\nCORRESPONDECE POINTS\n\n");
	
	for (i1=0; i1<iter; i1++) {
		printf("%d point1 = (%f,%f)\n",i1,cvGetReal2D(points1,0,i1),cvGetReal2D(points1,1,i1));
		printf("%d point2 = (%f,%f)\n",i1,cvGetReal2D(points2,0,i1),cvGetReal2D(points2,1,i1));
	}
    
    
    // CALCULATE FUNDAMENTAL MATRIX
    
    fundMatr = cvCreateMat(3,3,CV_32FC1);
	
	//see opencv manual for other options in computing the fundamental matrix
	int num = cvFindFundamentalMat(points1,points2,fundMatr,CV_FM_8POINT, 3 ,0.9999,status);
	if( num == 1 )
	{
		printf("\nCALCULATE FUNDAMENTAL MATRIX\n\n");
        cout << "F\n "<< endl << Mat(fundMatr) << endl;
	}
	else
	{
		printf("Fundamental matrix was not found\n");
	}
    
    
    
    CvMat *K = (CvMat*)cvLoad( "Intrinsics.xml" );
    
    cout << "K\n "<< endl << Mat(K) << endl;
    
    
    CvMat *KT = cvCreateMat(3,3,CV_32FC1);
	cvTranspose(K,KT);
    
    cout << "KT\n "<< endl << Mat(KT) << endl;
    
    
    CvMat *aa = cvCreateMat(3,3,CV_32FC1);
    CvMat *E = cvCreateMat(3,3,CV_32FC1);
    
    cvMatMul (KT,fundMatr,aa);
	cvMatMul (aa,K,E);
	
	printf("\n\n");
	cout << "\nE = K.t()*F*K\n "<< endl << Mat(E) << endl;
    
    
    Mat_<double> EE = Mat(E);
    
    
    cout << "\nDeterminant of EE: " << determinant(EE) << "\n";
    
    
    
    if (determinant(EE)<0) {
        cout << "\nDeterminant of EE: " << determinant(EE) << "\n";
        cout << "before -EE " << endl << EE << endl;
        EE = -EE;
        cout << "after  -EE " << endl << EE << endl;
        cout << "\nDeterminant of -EE: " << determinant(EE) << "\n";
        //return 0 ;
    }
    
    
    if (determinant(E) > 0.000001)
    {
        cout << "Essential Matrix constraint does not apply" << endl;
        return 0;
        
    }
    
    CvMat *invK = cvCreateMat(3,3,CV_32FC1);
     cvInvert (K, invK, CV_LU);
     CvMat *invKT = cvCreateMat(3,3,CV_32FC1);
     cvTranspose(invK,invKT);
     printf("lala\n");
    
    //Essential matrix: compute then extract cameras [R|t]
    //----------------------------------------------------
    
    Matx34d P1;
    
    cout << "\ndet(E) != 0 : " << cvDet(E) << "\n";
    
    Mat_<double> svd_u, svd_vt, svd_w;
    
    //Using OpenCV's SVD
    SVD svd(Mat(E),SVD::MODIFY_A);
	svd_u = svd.u;
	svd_vt = svd.vt;
	svd_w = svd.w;
    
	
	cout << "----------------------- SVD ------------------------\n";
	cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<endl;
	cout << "----------------------------------------------------\n";
    
    //check if first and second singular values are the same (as they should be)
    
    double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.7) {
		cout << "singular values are too far apart\n";
		//return false;
	}
    
    Mat_<double> R1(3,3);
    Mat_<double> R2(3,3);
    Mat_<double> t1(1,3);
    Mat_<double> t2(1,3);
    
    Mat_<double> R_final(3,3);
    Mat_<double> t_final(1,3);
    
    
    Matx33d W(0,-1,0,	//HZ 9.13
              1,0,0,
              0,0,1);
	Matx33d Wt(0,1,0,
               -1,0,0,
               0,0,1);
    
    
    R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
	R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
	t1 = svd_u.col(2); //u3
	t2 = -svd_u.col(2); //u3
    
    cout << "\ndet(R1) = " << determinant(R1) << endl;
    cout << "det(R2) = " << determinant(R2) << endl;
    
    

    
    cout << "\nR1 : " << Mat(R1) << "\n";
    cout << "\nR2 : " << Mat(R2) << "\n";
    cout << "\nt1 : " << Mat(t1) << "\n";
    cout << "\nt2 : " << Mat(t2) << "\n";
    
    // CHECK THE EPIPOLAR CONSTRAINT
    
    CvMat* x1 = cvCreateMat(3,1,CV_32F);
	cvZero(x1);
	CvMat* y1 = cvCreateMat(1,3,CV_32F);
	cvZero(y1);
	CvMat* lala = cvCreateMat(1,3,CV_32F);
	CvMat* lele = cvCreateMat(1,1,CV_32F);
    
    for (int k=0; k<iter; k++) {
        
        cvmSet( x1, 0, 0, cvGetReal2D(points1,0,k));
		cvmSet( x1, 1, 0, cvGetReal2D(points1,1,k));
		cvmSet( x1, 2, 0, 1 );
		cvmSet( y1, 0, 0, cvGetReal2D(points2,0,k));
		cvmSet( y1, 0, 1, cvGetReal2D(points2,1,k));
		cvmSet( y1, 0, 2, 1 );
        
        //cout << "\nx1 : " << Mat(x1) << "\n";
        //cout << "\ny1 : " << Mat(y1) << "\n";
        
        cvMatMul (y1,fundMatr,lala);
		cvMatMul (lala,x1,lele);
        printf("y%d'*F*x%d = %+10f\n",k,k,CV_MAT_ELEM( *lele, float,  0, 0));
        // cout << "\nu1 : " << u1 << "\n";
        // cout << "\nu2 : " << u2 << "\n";
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // REPORT 5 Implementation
    // 04/20/2014
    // Alexandros Lioulemes
    
    Matx34d PP0,PP1;
    
    
    PP0 = Matx34d(1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0);
    
    //cout << "Testing P0 " << endl << Mat(P0) << endl;
    
    PP1 = Matx34f(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
                  R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
                  R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
    //cout << "Testing P1 " << endl << Mat(P1) << endl;
    
    Mat pt_3d_h(1,numPoints,CV_32FC4);
    
    // Convert CvMat points1 to Point2f vector
    
    vector<Point2f> _pt_set1_pt, _pt_set2_pt;
    
    for (int k=0; k<numPoints; k++)
    {
        _pt_set1_pt.push_back(Point2f((float)cvGetReal2D(points1,0,k),(float)cvGetReal2D(points1,1,k)));
        _pt_set2_pt.push_back(Point2f((float)cvGetReal2D(points2,0,k),(float)cvGetReal2D(points2,1,k)));
    }
    
    
    cout << "Points in pixels coordinates\n";
    cout << "\npoints1:" << endl << _pt_set1_pt << endl;
    cout << "\npoints2:" << endl << _pt_set2_pt << endl;
    
    //undistort
	Mat pt_set1_pt,pt_set2_pt;
	undistortPoints(_pt_set1_pt, pt_set1_pt, Mat(K), Mat());
	undistortPoints(_pt_set2_pt, pt_set2_pt, Mat(K), Mat());
    
    cout << "Points in image plane coordinates\n";
    cout << "\npoints1:" << endl << pt_set1_pt << endl;
    cout << "\npoints2:" << endl << pt_set2_pt << endl;
    
    
    Mat_<double> eye = (cv::Mat_<double>(3,3) << PP0(0,0),PP0(0,1),PP0(0,2), PP0(1,0),PP0(1,1),PP0(1,2),PP0(2,0),PP0(2,1),PP0(2,2));
    
    
    Vec3d rvec;
    Rodrigues(eye ,rvec);
	Vec3d tvec(0,0,0);
    
    vector<Point2f> reproject_U1, reproject_U2;
    
    vector<Point3f> reproject_P_3D;
    
    int positive_z = 0;
    
    double whole_error = 0;
    
    int situation = 0;
    
    cout << "1------------------------------------------------------------" << endl;
    
    PP1 = Matx34f(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
                  R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
                  R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
    cout << "Testing PP1 " << endl << Mat(PP1) << endl;
    
    CvMat* u1 = cvCreateMat(3,1,CV_32F);
	cvZero(u1);
    CvMat* um1 = cvCreateMat(3,1,CV_32F);
	cvZero(um1);
    CvMat* u2 = cvCreateMat(3,1,CV_32F);
	cvZero(u2);
    CvMat* um2 = cvCreateMat(3,1,CV_32F);
	cvZero(um2);
    
    //Triangulate
	
    for (int i=0; i<numPoints; i++) {
        
        //convert to normalized homogeneous coordinates
        cvmSet( u1, 0, 0, cvGetReal2D(points1,0,i));
		cvmSet( u1, 1, 0, cvGetReal2D(points1,1,i));
		cvmSet( u1, 2, 0, 1 );
        //cout << "\nu : " << Mat(u) << "\n";
        cvMatMul (invK,u1,um1);
        //cout << "\num : " << Mat(um) << "\n";
        cvmSet( u2, 0, 0, cvGetReal2D(points2,0,i));
		cvmSet( u2, 1, 0, cvGetReal2D(points2,1,i));
		cvmSet( u2, 2, 0, 1 );
        //cout << "\nu1 : " << Mat(u1) << "\n";
        cvMatMul (invK,u2,um2);
        //cout << "\num1 : " << Mat(um1) << "\n";
        Point3d s1(cvGetReal2D(um1,0,0),cvGetReal2D(um1,1,0),1.0);
        Point3d s2(cvGetReal2D(um2,0,0),cvGetReal2D(um2,1,0),1.0);
        
        //triangulate
        Mat_<double> X = LinearLSTriangulation(s1,PP0,s2,PP1);
        cout <<	"3DPoint: " << "\t" << X(0) << "\t" << X(1) << "\t" << X(2) << endl;
        reproject_P_3D.push_back(Point3f(X(0),X(1),X(2)));
        
        if (X(2)>0) positive_z ++;
        
	}
    
    projectPoints(reproject_P_3D,rvec,tvec,Mat(K),Mat(),reproject_U1);
    cout << "\nreproject_U1:" << Mat(reproject_U1) << endl;
    cout << "\nReprojection error for camera 1\n: " << reprojection_error(U1,reproject_U1) << endl;
    
    Mat_<double> c2_R_c1 = (cv::Mat_<double>(3,3) << PP1(0,0),PP1(0,1),PP1(0,2), PP1(1,0),PP1(1,1),PP1(1,2),PP1(2,0),PP1(2,1),PP1(2,2));
    Mat_<double> c2_T_c1 = (cv::Mat_<double>(3,1) << PP1(0,3),PP1(1,3),PP1(2,3));
    projectPoints(reproject_P_3D,c2_R_c1.t(),-c2_R_c1.t()*c2_T_c1,Mat(K),Mat(),reproject_U2);
    cout << "\nreproject_U2:" << Mat(reproject_U2) << endl;
    cout << "\nReprojection error for camera 2\n: " << reprojection_error(U2,reproject_U2) << endl;
    
    whole_error = reprojection_error(U1,reproject_U1) + reprojection_error(U2,reproject_U2);
    situation = 1;
    
    R_final = c2_R_c1;
    t_final = c2_T_c1;
    
    cout << "\nNumber of point in front of camera: " << positive_z << endl;
    
    cout << "2------------------------------------------------------------" << endl;
    
    
    positive_z = 0;
    
    reproject_P_3D.clear();
    //reproject_U2.clear();
    //reproject_U1.clear();
    
    
    PP1 = Matx34f(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
                  R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
                  R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
    cout << "\nPP1:" << Mat(PP1) << endl;
    
    //triangulate
	
    for (int i=0; i<numPoints; i++) {
        
        //convert to normalized homogeneous coordinates
        cvmSet( u1, 0, 0, cvGetReal2D(points1,0,i));
		cvmSet( u1, 1, 0, cvGetReal2D(points1,1,i));
		cvmSet( u1, 2, 0, 1 );
        //cout << "\nu : " << Mat(u) << "\n";
        cvMatMul (invK,u1,um1);
        //cout << "\num : " << Mat(um) << "\n";
        cvmSet( u2, 0, 0, cvGetReal2D(points2,0,i));
		cvmSet( u2, 1, 0, cvGetReal2D(points2,1,i));
		cvmSet( u2, 2, 0, 1 );
        //cout << "\nu1 : " << Mat(u1) << "\n";
        cvMatMul (invK,u2,um2);
        //cout << "\num1 : " << Mat(um1) << "\n";
        Point3d s1(cvGetReal2D(um1,0,0),cvGetReal2D(um1,1,0),1.0);
        Point3d s2(cvGetReal2D(um2,0,0),cvGetReal2D(um2,1,0),1.0);
        
        //triangulate
        Mat_<double> X = LinearLSTriangulation(s1,PP0,s2,PP1);
        cout <<	"3DPoint: " << "\t" << X(0) << "\t" << X(1) << "\t" << X(2) << endl;
        reproject_P_3D.push_back(Point3f(X(0),X(1),X(2)));
        
        if (X(2)>0) positive_z ++;
	}
    
    projectPoints(reproject_P_3D,rvec,tvec,Mat(K),Mat(),reproject_U1);
    cout << "\nreproject_U1:" << Mat(reproject_U1) << endl;
    cout << "Reprojection error: " << reprojection_error(U1,reproject_U1) << endl;
    
    c2_R_c1 = (cv::Mat_<double>(3,3) << PP1(0,0),PP1(0,1),PP1(0,2), PP1(1,0),PP1(1,1),PP1(1,2),PP1(2,0),PP1(2,1),PP1(2,2));
    c2_T_c1 = (cv::Mat_<double>(3,1) << PP1(0,3),PP1(1,3),PP1(2,3));
    projectPoints(reproject_P_3D,c2_R_c1.t(),-c2_R_c1.t()*c2_T_c1,Mat(K),Mat(),reproject_U2);
    cout << "\nreproject_U2:" << Mat(reproject_U2) << endl;
    cout << "\nReprojection error for camera 2\n: " << reprojection_error(U2,reproject_U2) << endl;
    
    if (whole_error > reprojection_error(U1,reproject_U1) + reprojection_error(U2,reproject_U2)) {
        
        if (positive_z > numPoints-3){
        
			whole_error = reprojection_error(U1,reproject_U1) + reprojection_error(U2,reproject_U2);
			situation = 2;
			
			R_final = c2_R_c1;
			t_final = c2_T_c1;
		}
    }
    
    cout << "\nNumber of point in front of camera: " << positive_z << endl;
    
    cout << "3------------------------------------------------------------" << endl;
    
    positive_z = 0;
    
    reproject_P_3D.clear();
    
    
    
    PP1 = Matx34f(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
                  R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
                  R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
    cout << "\nPP1:" << Mat(PP1) << endl;
    
    //triangulate
	
    for (int i=0; i<numPoints; i++) {
        
        //convert to normalized homogeneous coordinates
        cvmSet( u1, 0, 0, cvGetReal2D(points1,0,i));
		cvmSet( u1, 1, 0, cvGetReal2D(points1,1,i));
		cvmSet( u1, 2, 0, 1 );
        //cout << "\nu : " << Mat(u) << "\n";
        cvMatMul (invK,u1,um1);
        //cout << "\num : " << Mat(um) << "\n";
        cvmSet( u2, 0, 0, cvGetReal2D(points2,0,i));
		cvmSet( u2, 1, 0, cvGetReal2D(points2,1,i));
		cvmSet( u2, 2, 0, 1 );
        //cout << "\nu1 : " << Mat(u1) << "\n";
        cvMatMul (invK,u2,um2);
        //cout << "\num1 : " << Mat(um1) << "\n";
        Point3d s1(cvGetReal2D(um1,0,0),cvGetReal2D(um1,1,0),1.0);
        Point3d s2(cvGetReal2D(um2,0,0),cvGetReal2D(um2,1,0),1.0);
        
        //triangulate
        Mat_<double> X = LinearLSTriangulation(s1,PP0,s2,PP1);
        cout <<	"3DPoint: " << "\t" << X(0) << "\t" << X(1) << "\t" << X(2) << endl;
        reproject_P_3D.push_back(Point3f(X(0),X(1),X(2)));
        if (X(2)>0) positive_z ++;
        
	}
    
    projectPoints(reproject_P_3D,rvec,tvec,Mat(K),Mat(),reproject_U1);
    cout << "\nreproject_U1:" << Mat(reproject_U1) << endl;
    cout << "Reprojection error: " << reprojection_error(U1,reproject_U1) << endl;
    
    
    
    c2_R_c1 = (cv::Mat_<double>(3,3) << PP1(0,0),PP1(0,1),PP1(0,2), PP1(1,0),PP1(1,1),PP1(1,2),PP1(2,0),PP1(2,1),PP1(2,2));
    c2_T_c1 = (cv::Mat_<double>(3,1) << PP1(0,3),PP1(1,3),PP1(2,3));
    projectPoints(reproject_P_3D,c2_R_c1.t(),-c2_R_c1.t()*c2_T_c1,Mat(K),Mat(),reproject_U2);
    cout << "\nreproject_U2:" << Mat(reproject_U2) << endl;
    cout << "\nReprojection error for camera 2\n: " << reprojection_error(U2,reproject_U2) << endl;
    
    
    if (whole_error > reprojection_error(U1,reproject_U1) + reprojection_error(U2,reproject_U2)) {
        
        if (positive_z > numPoints-3){
        
			whole_error = reprojection_error(U1,reproject_U1) + reprojection_error(U2,reproject_U2);
			situation = 3;
			
			R_final = c2_R_c1;
			t_final = c2_T_c1;
			
		}
        
    }
    
    cout << "\nNumber of point in front of camera: " << positive_z << endl;
    
    
    cout << "4------------------------------------------------------------" << endl;
    
    positive_z = 0;
    reproject_P_3D.clear();
    
    
    
    PP1 = Matx34f(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
                  R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
                  R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
    
    cout << "\nPP1:" << Mat(PP1) << endl;
    
    //triangulate
	
    for (int i=0; i<numPoints; i++) {
        
        //convert to normalized homogeneous coordinates
        cvmSet( u1, 0, 0, cvGetReal2D(points1,0,i));
		cvmSet( u1, 1, 0, cvGetReal2D(points1,1,i));
		cvmSet( u1, 2, 0, 1 );
        //cout << "\nu : " << Mat(u) << "\n";
        cvMatMul (invK,u1,um1);
        //cout << "\num : " << Mat(um) << "\n";
        cvmSet( u2, 0, 0, cvGetReal2D(points2,0,i));
		cvmSet( u2, 1, 0, cvGetReal2D(points2,1,i));
		cvmSet( u2, 2, 0, 1 );
        //cout << "\nu1 : " << Mat(u1) << "\n";
        cvMatMul (invK,u2,um2);
        //cout << "\num1 : " << Mat(um1) << "\n";
        Point3d s1(cvGetReal2D(um1,0,0),cvGetReal2D(um1,1,0),1.0);
        Point3d s2(cvGetReal2D(um2,0,0),cvGetReal2D(um2,1,0),1.0);
        
        //triangulate
        Mat_<double> X = LinearLSTriangulation(s1,PP0,s2,PP1);
        cout <<	"3DPoint: " << "\t" << X(0) << "\t" << X(1) << "\t" << X(2) << endl;
        reproject_P_3D.push_back(Point3f(X(0),X(1),X(2)));
        if (X(2)>0) positive_z ++;
        
	}
    
    projectPoints(reproject_P_3D,rvec,tvec,Mat(K),Mat(),reproject_U1);
    cout << "\nreproject_U1:" << Mat(reproject_U1) << endl;
    cout << "Reprojection error: " << reprojection_error(U1,reproject_U1) << endl;
    
    
    c2_R_c1 = (cv::Mat_<double>(3,3) << PP1(0,0),PP1(0,1),PP1(0,2), PP1(1,0),PP1(1,1),PP1(1,2),PP1(2,0),PP1(2,1),PP1(2,2));
    c2_T_c1 = (cv::Mat_<double>(3,1) << PP1(0,3),PP1(1,3),PP1(2,3));
    projectPoints(reproject_P_3D,c2_R_c1.t(),-c2_R_c1.t()*c2_T_c1,Mat(K),Mat(),reproject_U2);
    cout << "\nreproject_U2:" << Mat(reproject_U2) << endl;
    cout << "\nReprojection error for camera 2\n: " << reprojection_error(U2,reproject_U2) << endl;
    
    
    
    if (whole_error > reprojection_error(U1,reproject_U1) + reprojection_error(U2,reproject_U2)) {
        
        if (positive_z > numPoints-3){
        
			whole_error = reprojection_error(U1,reproject_U1) + reprojection_error(U2,reproject_U2);
			situation = 4;
			
			R_final = c2_R_c1;
			t_final = c2_T_c1;
			
		}
        
    }
    
    cout << "\nNumber of point in front of camera: " << positive_z << endl;
    
    
    
    
    // 4 ambiguities
    // choose one
    printf("\nSituation %d\n",situation);
    
    
    cout << "\nR_final\n" << R_final << endl;
    cout << "\nt_final\n" << t_final << endl;
    
    
    /////////////////////////////////////////////////////////////////////////////////////////
    // Report 5
    // WRITE 3D POINTS TO A FILE
    
    char buff[200];
	FILE *fl;
	
	fl = fopen("eg.pcd","w");
	
	if (fl == NULL) {
		fprintf(stderr, "Can't open output file!\n");
		exit(1);
	}
	
	strcpy (buff,"# .PCD v.7 - Point Cloud Data file format\n");
	fprintf(fl, buff);
	strcpy (buff,"VERSION .7\n");
	fprintf(fl, buff);
	strcpy (buff,"FIELDS x y z rgb\n");
	fprintf(fl, buff);
	strcpy (buff,"SIZE 4 4 4 1\n");
	fprintf(fl, buff);
	strcpy (buff,"TYPE F F F I\n");
	fprintf(fl, buff);
	strcpy (buff,"COUNT 1 1 1 4\n");
	fprintf(fl, buff);
	sprintf (buff,"WIDTH %d\n", numPoints);
	fprintf(fl, buff);
	strcpy (buff,"HEIGHT 1\n");
	fprintf(fl, buff);
	strcpy (buff,"VIEWPOINT 0 0 0 1 0 0 0\n");
	fprintf(fl, buff);
	sprintf (buff,"POINTS %d\n", numPoints);
	fprintf(fl, buff);
	strcpy (buff,"DATA ascii\n");
	fprintf(fl, buff);
	
	
	reproject_P_3D.clear();
    
    
    
    PP1 = Matx34f(R_final(0,0),	R_final(0,1),	R_final(0,2),	t_final(0),
                  R_final(1,0),	R_final(1,1),	R_final(1,2),	t_final(1),
                  R_final(2,0),	R_final(2,1),	R_final(2,2),	t_final(2));
    cout << "\nPP1:" << Mat(PP1) << endl;
    
    //triangulate
    
    int RGB[3];
    int xx, yy;
	
    for (int i=0; i<numPoints; i++) {
        
        //convert to normalized homogeneous coordinates
        cvmSet( u1, 0, 0, cvGetReal2D(points1,0,i));
		cvmSet( u1, 1, 0, cvGetReal2D(points1,1,i));
		cvmSet( u1, 2, 0, 1 );
        //cout << "\nu : " << Mat(u) << "\n";
        cvMatMul (invK,u1,um1);
        //cout << "\num : " << Mat(um) << "\n";
        cvmSet( u2, 0, 0, cvGetReal2D(points2,0,i));
		cvmSet( u2, 1, 0, cvGetReal2D(points2,1,i));
		cvmSet( u2, 2, 0, 1 );
        //cout << "\nu1 : " << Mat(u1) << "\n";
        cvMatMul (invK,u2,um2);
        //cout << "\num1 : " << Mat(um1) << "\n";
        Point3d s1(cvGetReal2D(um1,0,0),cvGetReal2D(um1,1,0),1.0);
        Point3d s2(cvGetReal2D(um2,0,0),cvGetReal2D(um2,1,0),1.0);
        
        //triangulate
        Mat_<double> X = LinearLSTriangulation(s1,PP0,s2,PP1);
        cout <<	"3DPoint: " << "\t" << X(0) << "\t" << X(1) << "\t" << X(2) << endl;
		xx = cvGetReal2D(points1,0,i);
		yy = cvGetReal2D(points1,1,i);
		printIplImage(image1rgb, xx, yy, RGB);
		
		
		sprintf (buff,"%g %g %g %d %d %d 1\n", X(0), X(1), X(2), RGB[2], RGB[1], RGB[0]);
        
        //sprintf (buff,"%g %g %g 1 255 1 1\n", X(0), X(1), X(2));
        
        
        fprintf(fl, buff);
	}
	
	fclose(fl);
    
    // FINISH
    
    cvWaitKey(0);
    cvDestroyWindow("first");
	cvDestroyWindow("second");
    cvDestroyWindow("Correspondences");
	
    return 0;
}


Mat_<double> LinearLSTriangulation(Point3d u,Matx34d P,Point3d u1,Matx34d P1){
	
	//build matrix A
	Matx43d A(u.x*P(2,0)-P(0,0),	u.x*P(2,1)-P(0,1),		u.x*P(2,2)-P(0,2),
			  u.y*P(2,0)-P(1,0),	u.y*P(2,1)-P(1,1),		u.y*P(2,2)-P(1,2),
			  u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),	u1.x*P1(2,2)-P1(0,2),
			  u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),	u1.y*P1(2,2)-P1(1,2));
    
    //build B vector
	Matx41d B(-(u.x*P(2,3)	-P(0,3)),
			  -(u.y*P(2,3)	-P(1,3)),
			  -(u1.x*P1(2,3)	-P1(0,3)),
			  -(u1.y*P1(2,3)	-P1(1,3)));
	
    //solve for X
	Mat_<double> X;
	solve(A,B,X,DECOMP_SVD);
	
	return X;
}

double reprojection_error(vector<Point2f> a, vector<Point2f> b){
    
    double error = 0;
    int size = a.size();
    //cout << "size:" << size << endl;
    
    //cout << "a:" << a << endl;
    
    for (int i=0; i<size; i++) {
        error = error + abs(a[i].x-b[i].x) + abs(a[i].y-b[i].y);
    }
    
    return error;
}


void printIplImage(const IplImage* src, int row, int col, int RGB[])
{
    
    int rr, gg, bb;
    
    bb = (int)src->imageData[src->widthStep * row + col * 3];
    gg = (int)src->imageData[src->widthStep * row + col * 3 + 1];
    rr = (int)src->imageData[src->widthStep * row + col * 3 + 2];
    
    
    if (bb < 0) {
        bb = 255 + bb;
    }
    
    if (gg < 0) {
        gg = 255 + gg;
    }
    
    if (rr < 0) {
        rr = 255 + rr;
    }
    
    RGB[0]=rr;
    RGB[1]=gg;
    RGB[2]=bb;
}


