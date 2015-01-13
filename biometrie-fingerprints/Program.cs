using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Emgu.CV.Structure;
using Emgu.CV.GPU;
using System.Drawing;
using System.Management.Instrumentation;
using System.Threading;
using Emgu.CV.CvEnum;
using Emgu.CV;
using System.IO;
using System.Linq;

namespace Biometrie
{
    internal class Program
    {
        private static void Main(string[] args)
        {
         //   args = new string[] { "-d", @"..\..\fp-images\10_3.bmp" };
            //args = new string[] { @"..\..\fp-images\boy1.gif" };
            //args = new string[] { @"..\..\fp-images\sine16.png" };
            if (args.Length == 0) PrintHelp();

            bool debug = args.Any(arg => arg == "-d");

            foreach (string file in args)
            {
                if(file.StartsWith("-")) continue;

                try
                {
                    var fp = new Fingerprint(file);
                    fp.Debug = debug;
                    fp.Save();
                }
                catch (FileNotFoundException e)
                {
                    Console.Error.WriteLine("Couldn't process {0}", file);
                }
            }
        }

        private static void PrintHelp()
        {
            Console.WriteLine("fpanalyze.exe [-d] file1.bmp [file2.bmp ...]");
            Console.WriteLine("--------------------------------------------------------");
            Console.WriteLine("\t-d\tEnables debugging mode");

        }
    }
}
