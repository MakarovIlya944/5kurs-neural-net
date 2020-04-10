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

    class L2LossFloat : ILossFunction<float>
    {
        public Vector<float> backPropagation(Vector<float> calc, Vector<float> truly)
        {
            throw new NotImplementedException();
        }

        public Matrix<float> backPropagation(Matrix<float> calc, Matrix<float> truly)
        {
            throw new NotImplementedException();
        }

        public float call(Vector<float> calc, Vector<float> truly)
        {
            return (float)(calc-truly).L2Norm();
        }

        public Vector<float> call(Matrix<float> calc, Matrix<float> truly)
        {
            throw new NotImplementedException();
        }
    }

    class L2LossDouble : ILossFunction<double>
    {
        public Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            throw new NotImplementedException();
        }

        public Matrix<double> backPropagation(Matrix<double> calc, Matrix<double> truly)
        {
            throw new NotImplementedException();
        }

        public double call(Vector<double> calc, Vector<double> truly)
        {
            return (calc - truly).L2Norm();
        }

        public Vector<double> call(Matrix<double> calc, Matrix<double> truly)
        {
            throw new NotImplementedException();
        }
    }
}
 