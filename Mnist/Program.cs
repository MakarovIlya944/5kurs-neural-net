using MathNet.Numerics.LinearAlgebra;
using Mnist.Functions;
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
        }

        static void Test()
        {
            Vector<double> a = Vector<double>.Build.Dense(3, 2);
            Vector<double> b = Vector<double>.Build.Dense(3, 3);
            Vector<double> c = a.Map2((x, y) => x * y, b);

            int add = 4;
        }

        static void Train()
        {
            Model m = new Model(3, 3, 1, 0.5);

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

            Data<double> data = new Data<double>(input, output);

            m.train(data, 1, 0.1, new SquareLoss());
        }
    }
}
