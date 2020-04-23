using MathNet.Numerics.LinearAlgebra;
using Mnist.Functions;
using Mnist.Pictures;
using System;
using System.Linq;

namespace Mnist
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!\n");
            //Test();
            Train();
            Console.WriteLine("Good bye World!");
            Console.ReadLine();
        }

        static void Test()
        {
            Vector<double> a = Vector<double>.Build.Dense(new double[2] { Math.Exp(1), Math.Exp(-1) });
            Vector<double> b = Vector<double>.Build.Dense(new double[2] { 6, 2 });
            Vector<double> c = Vector<double>.Build.Dense(new double[2] { 1, 1 });


            c = a.Map2((x, y) => y * Math.Log(x), b);

            ReLU f = new ReLU();
            Matrix<double> m = Matrix<double>.Build.DenseOfRowVectors(new Vector<double>[3] { a, b, c });
            Console.WriteLine(m.ToMatrixString());
            m = f.call(m);
            Console.WriteLine(m.ToMatrixString());

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
            //ILossFunction<double> loss = new L2Loss();
            //Vector<double>[] input = new Vector<double>[4];
            //input[0] = Vector<double>.Build.Dense(new double[2] { 0, 0 });
            //input[1] = Vector<double>.Build.Dense(new double[2] { 0, 1 });
            //input[2] = Vector<double>.Build.Dense(new double[2] { 1, 0 });
            //input[3] = Vector<double>.Build.Dense(new double[2] { 1, 1 });

            //Vector<double>[] output = new Vector<double>[4];
            //output[0] = Vector<double>.Build.Dense(new double[1] { 0 });
            //output[1] = Vector<double>.Build.Dense(new double[1] { 1 });
            //output[2] = Vector<double>.Build.Dense(new double[1] { 1 });
            //output[3] = Vector<double>.Build.Dense(new double[1] { 0 });

            //int dataSize = 4;

            //Data data = new Data(input.Take(dataSize).ToArray(), output.Take(dataSize).ToArray());

            //int inputSize = 2, outputSize = 1, deep = 4, epoch = 5;
            //int[] width = new int[3] { 3,7,11 };
            //double[] init = new double[4] { 1, 1, 1, 1 }, bias = new double[4] { 0, 0, 0, 0 };
            //double teachRate = 1E-3;

            int numberInputData = 2;
            double percentData = 0.000018 * numberInputData;
            Data data = MnistConverter.OpenMnist(@"D:\Projects\Mnist\data\train-labels.idx1-ubyte", @"D:\Projects\Mnist\data\train-images.idx3-ubyte", percentData);

            int inputSize = 28 * 28, outputSize = 10, deep = 3, epoch = 5;
            int[] width = new int[2] { inputSize / 2, inputSize / 4 };
            double[] init = new double[3] { 1, 1, 1 }, bias = new double[3] { 1, 1, 1 };
            double teachRate = 1E-3;
            ILossFunction<double> loss = new LogLoss();

            Model m = new Model(deep, width, init, bias, inputSize, outputSize);
            m.LogEpoch = 1;
            m.train(data, epoch, teachRate, loss);
        }
    }
}
