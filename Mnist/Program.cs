using Mnist.Functions;
using System;

namespace Mnist
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
        }

        static void Train()
        {
            Layer<float> l = new Layer<float>();
            Data<float> d = new Data<float>();
            SimpleActivation a = new SimpleActivation();
            WeightsMatrix w = new WeightsMatrix(2,3,1);
        }
    }
}
