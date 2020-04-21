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

        protected override Vector<double> f(Vector<double> x)
        {
            double s;
            bool overflow = x.Exists(e => Math.Abs(e) > 709);
            if (overflow)
            {
                s = Double.MaxValue;
                return x.Map(e => e = Math.Abs(e) > 709 ? 1 : Math.Exp(e) / s);
            }
            else
            {
                s = x.Map(e => Math.Exp(e)).Sum();
                return x.Map(e => e = Math.Exp(e) / s);
            }
        }

        protected override Vector<double> df(Vector<double> x)
        {
            Vector<double> v = f(x);
            Matrix<double> dm = Matrix<double>.Build.Dense(dim, dim);
            dm = dm.MapIndexed((i, j, val) => v[i] * (i == j ? 1 : 0 - v[j]));
            return x * dm;
        }
    }
}
