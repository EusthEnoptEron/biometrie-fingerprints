using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace Biometrie
{
    class Fourier
    {
        public static Matrix<float> DFT(Image<Gray, float> image)
        {

            // Transform 1 channel grayscale image into 2 channel image
            IntPtr complexImage = CvInvoke.cvCreateImage(image.Size, Emgu.CV.CvEnum.IPL_DEPTH.IPL_DEPTH_32F, 2);
            CvInvoke.cvSetImageCOI(complexImage, 1); // Select the channel to copy into
            CvInvoke.cvCopy(image, complexImage, IntPtr.Zero);
            CvInvoke.cvSetImageCOI(complexImage, 0); // Select all channels

            // This will hold the DFT data
            Matrix<float> forwardDft = new Matrix<float>(image.Rows, image.Cols, 2);
            CvInvoke.cvDFT(complexImage, forwardDft, CV_DXT.CV_DXT_FORWARD, 0);

            CvInvoke.cvReleaseImage(ref complexImage);

            return forwardDft;
        }

        // We have to switch quadrants so that the origin is at the image center
        public static void SwitchQuadrants(ref Matrix<float> matrix)
        {
            int cx = matrix.Cols / 2;
            int cy = matrix.Rows / 2;

            Matrix<float> q0 = matrix.GetSubRect(new Rectangle(0, 0, cx, cy));
            Matrix<float> q1 = matrix.GetSubRect(new Rectangle(cx, 0, cx, cy));
            Matrix<float> q2 = matrix.GetSubRect(new Rectangle(0, cy, cx, cy));
            Matrix<float> q3 = matrix.GetSubRect(new Rectangle(cx, cy, cx, cy));
            Matrix<float> tmp = new Matrix<float>(q0.Size);

            q0.CopyTo(tmp);
            q3.CopyTo(q0);
            tmp.CopyTo(q3);
            q1.CopyTo(tmp);
            q2.CopyTo(q1);
            tmp.CopyTo(q2);
        }

        // Real part is magnitude, imaginary is phase. 
        // Here we compute log(sqrt(Re^2 + Im^2) + 1) to get the magnitude and 
        // rescale it so everything is visible
        public static Matrix<float> GetDftMagnitude(Matrix<float> fftData)
        {
            //The Real part of the Fourier Transform
            Matrix<float> outReal = new Matrix<float>(fftData.Size);
            //The imaginary part of the Fourier Transform
            Matrix<float> outIm = new Matrix<float>(fftData.Size);
            CvInvoke.cvSplit(fftData, outReal, outIm, IntPtr.Zero, IntPtr.Zero);

            CvInvoke.cvPow(outReal, outReal, 2.0);
            CvInvoke.cvPow(outIm, outIm, 2.0);

            CvInvoke.cvAdd(outReal, outIm, outReal, IntPtr.Zero);
            CvInvoke.cvPow(outReal, outReal, 0.5);

            CvInvoke.cvAddS(outReal, new MCvScalar(1.0), outReal, IntPtr.Zero); // 1 + Mag
            CvInvoke.cvLog(outReal, outReal); // log(1 + Mag)            

            return outReal;
        }


    }
}
