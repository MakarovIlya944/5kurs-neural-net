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
            Train();
        }

        static void Train()
        {
            Model m = new Model(3, 2, 1, 0.5);

            Vector<double>[] data = new Vector<double>[4];
            data[0] = Vector<double>.Build.Dense(new double[2] { 0, 0 });
            data[1] = Vector<double>.Build.Dense(new double[2] { 0, 1 });
            data[2] = Vector<double>.Build.Dense(new double[2] { 1, 0 });
            data[3] = Vector<double>.Build.Dense(new double[2] { 1, 1 });
            m.train(data, 1, 0.1, new SquareLoss());
        }
    }
}
