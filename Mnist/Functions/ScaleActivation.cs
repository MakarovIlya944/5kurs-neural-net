using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;

namespace Mnist.Functions
{
    class ScaleActivation : BaseActivation
    {

        override protected Vector<double> f(Vector<double> x)
        {
            double max = x[0];
            foreach (double e in x)
                max = Math.Max(max, e);
            return x.Map(e => e = e > 0 ? e / max : 0);
        }

        override protected Vector<double> df(Vector<double> x)
        {
            double max = x[0];
            foreach (double e in x)
                max = Math.Max(max, e);
            return x.Map(e => e = e > 0 ? 1 / max : 0);
        }
    }
}
