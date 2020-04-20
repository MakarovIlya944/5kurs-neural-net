using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    /*
    Loss fuction:
    sqrt(sum((y-x)^2))

    Derivative:
    ?
    */

    class L2Loss : BaseLoss
    {
        override public Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            double l2norm = call(calc, truly);
            return (calc - truly) / l2norm;
        }

        override public double call(Vector<double> calc, Vector<double> truly)
        {
            return (calc - truly).L2Norm();
        }
    }
}
 