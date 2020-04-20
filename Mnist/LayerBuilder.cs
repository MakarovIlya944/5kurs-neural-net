using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist.Fabrics
{
    static public class LayerBuilder
    {
        static public Layer BuildDense(int inputSize, int outputSize, double matrixInit, double bias, IActivationFunction<double> f)
        {
            return new Layer(outputSize, inputSize, matrixInit, bias, f);
        }


    }
}
