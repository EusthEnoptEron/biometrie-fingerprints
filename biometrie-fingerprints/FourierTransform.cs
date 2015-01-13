using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace Biometrie
{
    public static class FourierTransform
    {

        //----------------------------------------------------------
        // Recombinate quadrants
        //----------------------------------------------------------
        //public static void Recomb(Mat &src,Mat &dst)
        //{
        //    int cx = src.cols>>1;
        //    int cy = src.rows>>1;
        //    Mat tmp;
        //    tmp.create(src.size(),src.type());
        //    src(Rect(0, 0, cx, cy)).copyTo(tmp(Rect(cx, cy, cx, cy)));
        //    src(Rect(cx, cy, cx, cy)).copyTo(tmp(Rect(0, 0, cx, cy)));  
        //    src(Rect(cx, 0, cx, cy)).copyTo(tmp(Rect(0, cy, cx, cy)));
        //    src(Rect(0, cy, cx, cy)).copyTo(tmp(Rect(cx, 0, cx, cy)));
        //    dst=tmp;
        //}

        //----------------------------------------------------------
        // Forward fft
        //----------------------------------------------------------
        //public static void ForwardFFT(Matrix<float> Src, Matrix<float> FImg)
        //{
        //    int M = CvInvoke.cvGetOptimalDFTSize( Src.Rows );
        //    int N = CvInvoke.cvGetOptimalDFTSize( Src.Cols );
        //    Matrix<float> padded = new Matrix<float>(Src.Size);
        //    CvInvoke.cvCopyMakeBorder(Src, padded, new Point(M - Src.Rows, N - Src.Cols), Emgu.CV.CvEnum.BORDER_TYPE.CONSTANT, new Emgu.CV.Structure.MCvScalar(0, 0, 0, 0));

        //    // Create complex image
        //    // planes[0] image , planes[1] filled by zeroes
        //    var nulledMat = new Matrix<float>(padded.Size);
        //    nulledMat.SetValue(0);

        //    Matrix<float>[] planes = {padded, nulledMat};
        //    Matrix<float> complexImg = new Matrix<float>(padded.Size);

        //    //Emgu.CV.GPU.GpuInvoke.Merge(
        //    merge(planes, 2, complexImg); 
        //    dft(complexImg, complexImg,DFT_SCALE);    
        //    // After tranform we also get complex image 
        //    split(complexImg, planes);

        //    // 
        //    planes[0] = planes[0](Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
        //    planes[1] = planes[1](Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));

        //    Recomb(planes[0],planes[0]);
        //    Recomb(planes[1],planes[1]);

        //    FImg[0]=planes[0].clone();
        //    FImg[1]=planes[1].clone();
        //}
        ////----------------------------------------------------------
        //// Inverse FFT
        ////----------------------------------------------------------
        //public static void InverseFFT(Mat *FImg,Mat &Dst)
        //{
        //    Recomb(FImg[0],FImg[0]);
        //    Recomb(FImg[1],FImg[1]);
        //    Mat complexImg;
        //    merge(FImg, 2, complexImg);
        //    // Inverse transform
        //    dft(complexImg, complexImg,  DFT_INVERSE);
        //    split(complexImg, FImg);        
        //    FImg[0].copyTo(Dst);
        //}
        ////----------------------------------------------------------
        //// Forward FFT using Magnitude and phase
        ////----------------------------------------------------------
        //public static void ForwardFFT_Mag_Phase(Mat &src, Mat &Mag,Mat &Phase)
        //{
        //    Mat planes[2];
        //    ForwardFFT(src,planes);
        //    Mag.zeros(planes[0].rows,planes[0].cols,CV_32F);
        //    Phase.zeros(planes[0].rows,planes[0].cols,CV_32F);
        //    cv::cartToPolar(planes[0],planes[1],Mag,Phase);
        //}
        ////----------------------------------------------------------
        //// Inverse FFT using Magnitude and phase
        ////----------------------------------------------------------
        //public static void InverseFFT_Mag_Phase(Mat &Mag,Mat &Phase, Mat &dst)
        //{
        //    Mat planes[2];
        //    planes[0].create(Mag.rows,Mag.cols,CV_32F);
        //    planes[1].create(Mag.rows,Mag.cols,CV_32F);
        //    cv::polarToCart(Mag,Phase,planes[0],planes[1]);
        //    InverseFFT(planes,dst);
        //}
    }
}
