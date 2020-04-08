using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    class SimpleActivation<T> : IActivationFunction<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Vector<T> backPropagation(Vector<T> v)
        {
            throw new NotImplementedException();
        }

        public Vector<T> call(Vector<T> v)
        {
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
                if (x < 1)
                    return x;
                else
                    return 1;
            else
                return 0;
        }

        private double f(double x)
        {
            if (x > 0)
                if (x < 1)
                    return x;
                else
                    return 1;
            else
                return 0;
        }
    }
}
