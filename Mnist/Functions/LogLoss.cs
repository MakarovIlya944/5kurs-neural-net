using System;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    /*
    Loss fuction:
    -sum(x*log(y))

    Derivative:
    -(y-x)
    */

    class LogLoss : BaseLoss
    {
        public override double call(Vector<double> calc, Vector<double> truly)
        {
            return -(calc.Map2((x, y) => y * Math.Log(x), truly).Sum());
        }

        public override Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            return -(truly.PointwiseDivide(calc));
        }
    }
}
 