using MathNet.Numerics.LinearAlgebra;
using Mnist.Functions;
using Mnist.Pictures;
using System;

namespace Mnist
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            //Test();
            Train();
            Console.WriteLine("Good bye World!");
            Console.ReadLine();
        }

        static void Test()
        {
            Vector<double> a = Vector<double>.Build.Dense(new double[2] { 0, 1 });
            Vector<double> b = Vector<double>.Build.Dense(new double[2] { 1, 1 });
            Vector<double> c;// = a.Map2((x, y) => x * y, b);


            Vector<double> calc = Vector<double>.Build.Dense(new double[2] { 1, 1 });
            Vector<double> truly = Vector<double>.Build.Dense(new double[2] { 0, 1 });

            Vector<double> dv = calc.Map2((x, y) => x * Math.Log(y), truly);

            double d = dv.Sum();

            Vector<double>[] input = new Vector<double>[4];
            input[0] = Vector<double>.Build.Dense(new double[2] { 0, 0 });
            input[1] = Vector<double>.Build.Dense(new double[2] { 0, 1 });
            input[2] = Vector<double>.Build.Dense(new double[2] { 1, 0 });
            input[3] = Vector<double>.Build.Dense(new double[2] { 1, 1 });

            Vector<double>[] output = new Vector<double>[4];
            output[0] = Vector<double>.Build.Dense(new double[1] { 0 });
            output[1] = Vector<double>.Build.Dense(new double[1] { 1 });
            output[2] = Vector<double>.Build.Dense(new double[1] { 1 });
            output[3] = Vector<double>.Build.Dense(new double[1] { 0 });

            int add = 4;
        }

        static void Train()
        {
            int numberInputData = 1;
            double percentData = 0.000018 * numberInputData;
            Data data = MnistConverter.OpenMnist(@"D:\Projects\Mnist\data\train-labels.idx1-ubyte", @"D:\Projects\Mnist\data\train-images.idx3-ubyte", percentData);

            int inputSize = 28 * 28, outputSize = 10, deep = 3, epoch = 5;
            int[] width = new int[2] { inputSize, 128 };
            double[] init = new double[3] { 1, 1, 1 }, bias = new double[3] { 2, 3, 1 };
            double teachRate = 1E+5;

            Model m = new Model(deep, width, init, bias, inputSize, outputSize);
            m.LogEpoch = 1;
            m.train(data, epoch, teachRate, new LogLoss());
        }
    }
}
