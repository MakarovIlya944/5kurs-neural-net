using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    public class SoftMax : BaseActivation
    {
        int dim;

        public SoftMax(int dim)
        {
            this.dim = dim;
        }

        override protected Vector<double> f(Vector<double> x)
        {
            double s = x.Map(e => Math.Exp(e)).Sum();
            return x.Map(e => e = Math.Exp(e)/s);
        }

        override protected Vector<double> df(Vector<double> x)
        {
            Vector<double> v = f(x);
            Matrix<double> dm = Matrix<double>.Build.Dense(dim, dim);
            dm = dm.MapIndexed((i, j, val) => v[i] * (i == j ? 1 : 0 - v[j]));
            return x * dm;
        }
    }
}
