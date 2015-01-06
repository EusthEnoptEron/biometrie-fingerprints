using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.GPU;
using System.Drawing;
using System.Management.Instrumentation;

namespace Biometrie
{
    internal class Program
    {
        private const int BLOCK_SIZE = 8;

        private struct Vector2
        {
            public double x;
            public double y;

            public Vector2(double p1, double p2)
            {
                this.x = p1;
                this.y = p2;
            }

            public static Vector2 operator +(Vector2 v1, Vector2 v2)
            {
                return new Vector2(v1.x + v2.x, v1.y + v2.y);
            }

            public bool Empty
            {
                get { return x == 0 && y == 0; }
            }
        }

        private static void Main(string[] args)
        {
            args = new string[] {@"..\..\fp-images\10_2.bmp"};
            if (args.Length == 0) return;

            using (var src = new Image<Gray, byte>(args[0]))
            {
                var image = ImproveImage(src);

                // 1. Make quality map, improve image, make segmentation, etc...
                var quality = src.SmoothGaussian(51, 51, 10, 10);
                quality.Save("gaussian.png");
                image.Save("improvedImage.png");


                // 2. Calculate vectors
                var G = GetVectorMatrix(src);
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
                        int pxlX = x*BLOCK_SIZE*scale;
                        int pxlY = y*BLOCK_SIZE*scale;
                        int dx = (int) (Math.Cos(perpAngle)*scale/2);
                        int dy = (int) (Math.Sin(perpAngle)*scale/2);

                        for (int i = 0; i < BLOCK_SIZE; i++)
                        {
                            for (int j = 0; j < BLOCK_SIZE; j++)
                            {
                                var p1 = new Point(pxlX + i*scale, pxlY + j*scale);
                                var p2 = new Point(p1.X + dx, p1.Y + dy);

                                vectorImage.Draw(new LineSegment2D(p1, p2), new Gray(), 1);
                            }
                        }
                    }
                }


                // 4. Make simple skeleton
                var skeleton = Skeletize(image);


                skeleton.Save("skeleton.png");
                vectorImage.Save("vectors.png");

                quality.Dispose();
                vectorImage.Dispose();
            }
        }


        private static Image<Gray, byte> ImproveImage(Image<Gray, byte> src)
        {
            var improvedImage = src.Clone();
            //improvedImage._EqualizeHist();
            improvedImage._GammaCorrect(5.8d);
            improvedImage._Dilate(1);
            improvedImage._Erode(1);
           
            return improvedImage;
        }

        private static Vector2[,] GetVectorMatrix(Image<Gray, byte> src, Image<Gray, byte> mask = null)
        {
            var G = new Vector2[src.Height, src.Width];

            // Create gradients
            using(var  Gx = src.Sobel(1, 0, 3))
            using (var Gy = src.Sobel(0, 1, 3))
            {
                for (int x = 0; x < src.Width; x++)
                {
                    for (int y = 0; y < src.Height; y++)
                    {
                        if (mask != null && mask[y, x].Intensity <= 0) continue;

                        var gx = Gx[y, x];
                        var gy = Gy[y, x];


                        double gxInt = Math.Max(-1, Math.Min(1, (gx.Intensity/255)));
                        double gyInt = Math.Max(-1, Math.Min(1, (gy.Intensity/255)));

                        if (gyInt == 0 && gxInt == 0) continue;

                        G[y, x] = new Vector2(gxInt*gxInt - gyInt*gyInt,
                            2*gxInt*gyInt);
                    }
                }


                Gx.Save("Gx.png");
                Gy.Save("Gy.png");
            }

            return G;
        }

        private static Vector2[,] GetAverageVectorMatrix(Vector2[,] G, int W)
        {
            // |_ _ _| _ _ |
            // (+ W - 1) = Ceil
            var M = new Vector2[(G.GetLength(0) + W - 1)/W, (G.GetLength(1) + W - 1)/W];

            for (int x = 0; x < G.GetLength(1); x++)
            {

                for (int y = 0; y < G.GetLength(0); y++)
                {
                    int avgX = x/W;
                    int avgY = y/W;

                    M[avgY, avgX] += G[y, x];
                }
            }

            return M;
        }

        private static Image<Gray, byte> Skeletize(Image<Gray, byte> image)
        {
            // Create inverted copy
            image = image.Clone().Not();//.ThresholdBinary(new Gray(100), new Gray(255));

            // Code adapted from http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
            Image<Gray, Byte> eroded = new Image<Gray, byte>(image.Size);
            Image<Gray, Byte> temp = new Image<Gray, byte>(image.Size);
            Image<Gray, Byte> skel = new Image<Gray, byte>(image.Size);

            skel.SetValue(0);
            CvInvoke.cvThreshold(image, image, 127, 256, 0);

            StructuringElementEx kernel = new StructuringElementEx(3, 3, 1, 1, Emgu.CV.CvEnum.CV_ELEMENT_SHAPE.CV_SHAPE_RECT);
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

    }
}
