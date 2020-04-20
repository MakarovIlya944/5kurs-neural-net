using System;
using System.Collections.Generic;
using System.Linq;
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

    class LogLoss : BaseLoss
    {
        override public double call(Vector<double> calc, Vector<double> truly)
        {
            return -(calc.Map2((x, y) => x * Math.Log(y), truly).Sum());
        }

        override public Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            return -(truly.PointwiseDivide(calc));
        }
    }
}
 