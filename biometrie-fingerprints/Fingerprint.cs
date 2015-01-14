using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;

namespace Biometrie
{
    /// <summary>
    /// Represents a fingerprint.
    /// </summary>
    class Fingerprint
    {
        private const int BLOCK_SIZE = 8;

        private FileInfo fileInfo;
        private string baseName;
        public Fingerprint(string path)
        {
            fileInfo = new FileInfo(path);
            if (!fileInfo.Exists) throw new FileNotFoundException();

            baseName = fileInfo.Name.Substring(0, fileInfo.Name.Length - fileInfo.Extension.Length);
        }


        /// <summary>
        /// Saves the information of this fingerprint.
        /// </summary>
        public void Save()
        {
            using (var src = new Image<Gray, byte>(fileInfo.FullName))
            {
                debug(src, "original");
                var image = ImproveImage(src);
                debug(image, "improved");

                // 1. Make quality map, improve image, make segmentation, etc...
                var quality = src.SmoothGaussian(51, 51, 10, 10);
                debug(quality, "gaussian");


                // 2. Calculate vectors
                var G = GetVectorMatrix(image);
                G = GetAverageVectorMatrix(G, BLOCK_SIZE);

                // 3. Draw vectors on an image
                const int scale = 10;
                var vectorImage = src.Resize(scale, Emgu.CV.CvEnum.INTER.CV_INTER_NN);

                for (int x = 0; x < G.GetLength(1); x++)
                {
                    for (int y = 0; y < G.GetLength(0); y++)
                    {
                        if (G[y, x].Empty) continue;

                        // # Determine angles
                        double angle = Math.Atan2(G[y, x].y, G[y, x].x) * 0.5;
                        double perpAngle = angle <= 0
                            ? angle + 0.5 * Math.PI
                            : angle - 0.5 * Math.PI;


                        // # Output
                        int pxlX = x * BLOCK_SIZE * scale;
                        int pxlY = y * BLOCK_SIZE * scale;
                        int dx = (int)(Math.Cos(perpAngle) * scale / 2);
                        int dy = (int)(Math.Sin(perpAngle) * scale / 2);

                        for (int i = 0; i < BLOCK_SIZE; i++)
                        {
                            for (int j = 0; j < BLOCK_SIZE; j++)
                            {
                                var p1 = new Point(pxlX + i * scale, pxlY + j * scale);
                                var p2 = new Point(p1.X + dx, p1.Y + dy);

                                vectorImage.Draw(new LineSegment2D(p1, p2), new Gray(), 1);
                            }
                        }
                    }
                }

                // 4. Make simple skeleton
                var skeleton = Skeletize(image);

                debug(skeleton, "skeleton");
                save(vectorImage, "result");

                quality.Dispose();
                vectorImage.Dispose();

            }
        }

        private void debug<TDepth>(CvArray<TDepth> image, string name) where TDepth : new()
        {
            if (Debug)
                save(image, name);
        }


        private void save<TDepth>(CvArray<TDepth> image, string name) where TDepth : new()
        {
            image.Save(/*fileInfo.DirectoryName*/ AppDomain.CurrentDomain.BaseDirectory + @"\" + baseName + "." + name + ".png");
        }

        /// <summary>
        /// Conducts several improvements to, well, improve the result.
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        private Image<Gray, byte> ImproveImage(Image<Gray, byte> src)
        {
            var improvedImage = src.Clone().Convert<Gray, float>();
            var segmentation = CalculateSegmentation(improvedImage);

            improvedImage = ComputeFourier(improvedImage);

            debug(improvedImage, "after_fourier");

            // Raise contrast
            //improvedImage._GammaCorrect(2.8d);

            //improvedImage = (improvedImage / 255.0).Pow(4) * 255.0;
            improvedImage = (improvedImage / 255.0).Pow(0.5) * 255.0;

            // Remove dirty stuff
            improvedImage -= segmentation.Not();

            // Close
            //improvedImage._Dilate(1);
            //improvedImage._Erode(1);
            return improvedImage.Convert<Gray, byte>();
        }

        /// <summary>
        /// Computes the FFT of the image and cuts out a portion.
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        private Image<Gray, float> ComputeFourier(Image<Gray, float> src)
        {
            var improvedImage = src.Copy();

            // Adapted from: http://stackoverflow.com/questions/16812950/how-do-i-compute-dft-and-its-reverse-with-emgu
            Matrix<float> dft = Fourier.DFT(src);

            for (int x = 0; x < dft.Cols; x++)
            {
                for (int y = 0; y < dft.Rows; y++)
                {
                    int fX = (dft.Cols / 2) - Math.Abs(x - dft.Cols / 2);
                    int fY = ((dft.Rows / 2) - Math.Abs(y - dft.Rows / 2)); //* 0.6f;

                    var magnitude = Math.Sqrt(fX * fX + fY * fY);

                    // If frequencies are in a certain range...
                    if (magnitude < 30 || magnitude > 60)
                    {
                        // Clear frequencies
                        dft.Data[y, x * dft.NumberOfChannels + 0] = 0f;
                        dft.Data[y, x * dft.NumberOfChannels + 1] = 0f;
                    }
                }
            }

            // Make the amplitude image
            Matrix<float> forwardDftMagnitude = Fourier.GetDftMagnitude(dft);
            Fourier.SwitchQuadrants(ref forwardDftMagnitude);
            CvInvoke.cvNormalize(forwardDftMagnitude, forwardDftMagnitude, 0, 255.0, NORM_TYPE.CV_MINMAX, IntPtr.Zero);
            //forwardDftMagnitude = forwardDftMagnitude - mask;
            debug(forwardDftMagnitude, "fourier");


            Matrix<float> reverseDft = new Matrix<float>(improvedImage.Rows, improvedImage.Cols, 2);
            CvInvoke.cvDFT(dft, reverseDft, CV_DXT.CV_DXT_INV_SCALE, 0);
            CvInvoke.cvConvert(reverseDft.Split()[0], improvedImage);
            //CvInvoke.cvNormalize(improvedImage, improvedImage, 0, 255.0, NORM_TYPE.CV_MINMAX, IntPtr.Zero);

            return improvedImage;
        }

        /// <summary>
        /// Calculates a segmentation mask.
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        private Image<Gray, float> CalculateSegmentation(Image<Gray, float> src)
        {
            int size = 10;
            var mu = src.SmoothBlur(size, size);
            var mu2 = src.Mul(src).SmoothBlur(size, size);
            var sigma = src.Copy();
            CvInvoke.cvSqrt(mu2 - mu.Mul(mu), sigma);
            sigma._ThresholdBinary(new Gray(5), new Gray(256));

            debug(sigma, "segmentation");
            //sigma = sigma.Mul(sigma);
            return sigma.Erode(5);
        }

        /// <summary>
        /// Calculates the vector matrix for every field. (using the sobel filter)
        /// </summary>
        /// <param name="src"></param>
        /// <param name="mask"></param>
        /// <returns></returns>
        private Vector2[,] GetVectorMatrix(Image<Gray, byte> src, Image<Gray, byte> mask = null)
        {
            var G = new Vector2[src.Height, src.Width];

            // Create gradients
            using (var Gx = src.Sobel(1, 0, 3))
            using (var Gy = src.Sobel(0, 1, 3))
            {
                for (int x = 0; x < src.Width; x++)
                {
                    for (int y = 0; y < src.Height; y++)
                    {
                        if (mask != null && mask[y, x].Intensity <= 0) continue;

                        var gx = Gx[y, x];
                        var gy = Gy[y, x];


                        double gxInt = Math.Max(-1, Math.Min(1, (gx.Intensity / 255)));
                        double gyInt = Math.Max(-1, Math.Min(1, (gy.Intensity / 255)));

                        if (gyInt == 0 && gxInt == 0) continue;

                        G[y, x] = new Vector2(gxInt * gxInt - gyInt * gyInt,
                            2 * gxInt * gyInt);
                    }
                }


                debug(Gx, "Gx");
                debug(Gy, "Gy");
            }

            return G;
        }

        /// <summary>
        /// Averages a vector matrix according to BLOCK_SIZE
        /// </summary>
        /// <param name="G"></param>
        /// <param name="W"></param>
        /// <returns></returns>
        private Vector2[,] GetAverageVectorMatrix(Vector2[,] G, int W)
        {
            // |_ _ _| _ _ |
            // (+ W - 1) = Ceil
            var M = new Vector2[(G.GetLength(0) + W - 1) / W, (G.GetLength(1) + W - 1) / W];

            for (int x = 0; x < G.GetLength(1); x++)
            {

                for (int y = 0; y < G.GetLength(0); y++)
                {
                    int avgX = x / W;
                    int avgY = y / W;

                    M[avgY, avgX] += G[y, x];
                }
            }

            return M;
        }


        /// <summary>
        /// Tries to skeletonize the image.
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        private Image<Gray, byte> Skeletize(Image<Gray, byte> image)
        {
            // Create inverted copy
            image = image.Clone().Not();//.ThresholdBinary(new Gray(100), new Gray(255));
            // Code adapted from http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
            Image<Gray, Byte> eroded = new Image<Gray, byte>(image.Size);
            Image<Gray, Byte> temp = new Image<Gray, byte>(image.Size);
            Image<Gray, Byte> skel = new Image<Gray, byte>(image.Size);

            skel.SetValue(0);
            CvInvoke.cvThreshold(image, image, 220, 256, 0);


            StructuringElementEx kernel = new StructuringElementEx(3, 3, 1, 1, Emgu.CV.CvEnum.CV_ELEMENT_SHAPE.CV_SHAPE_CROSS);
            bool done = false;

            int i = 0;
            while (true)
            {

                CvInvoke.cvErode(image, eroded, kernel, 1);
                CvInvoke.cvDilate(eroded, temp, kernel, 1);
                temp = image - temp;
                skel = skel | temp;
                image = eroded.Clone();

                if (CvInvoke.cvCountNonZero(image) == 0)
                    break;
            }
            return skel.Not();
        }


        /// <summary>
        /// Gets or sets whether or not debug images should be printed.
        /// </summary>
        public bool Debug = false;
    }
}
