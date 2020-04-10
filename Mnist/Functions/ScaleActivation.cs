using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    class ScaleActivation<T> : IActivationFunction<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Vector<T> backPropagation(Vector<T> v)
        {
            throw new NotImplementedException();
        }

        public Matrix<T> backPropagation(Matrix<T> v)
        {
            throw new NotImplementedException();
        }

        public Vector<T> call(Vector<T> v)
        {
            return v.Map(x => f(x));
        }

        public Matrix<T> call(Matrix<T> v)
        {
            double max = v.PointwiseAbsoluteMaximum(0);
            return v.Map(x => f(x));
        }

        private T f(T x)
        {
            if (x is double) return (T)(object)f((double)(object)x);
            //else if(x is double) return (T)(object)f(x);

            throw new NotImplementedException();
        }

        private float f(float x)
        {
            if (x > 0)
                return x;
            else
                return 0;
        }

        private double f(double x)
        {
            if (x > 0)
                return x;
            else
                return 0;
        }
    }
}
