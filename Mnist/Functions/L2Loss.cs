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
        public override Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            return (calc - truly) / call(calc, truly);
        }

        public override double call(Vector<double> calc, Vector<double> truly)
        {
            return (calc - truly).L2Norm();
        }
    }
}
 