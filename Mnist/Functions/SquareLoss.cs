using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    /*
    Loss fuction:
    sum((y-x)^2)/2

    Derivative:
    -(y-x)
    */

    class SquareLoss : ILossFunction<double>
    {
        public Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            return -(truly - calc);
        }

        public double call(Vector<double> calc, Vector<double> truly)
        {
            return (truly - calc).Map(x=>x*x).Sum()/2;
        }
    }
}
 