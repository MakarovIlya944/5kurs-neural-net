using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    class Sigmoid : BaseActivation
    {
        double k = 1;

        public Sigmoid(double scale)
        {
            k = scale;
        }

        public Sigmoid()
        {
        }

        override protected Vector<double> f(Vector<double> x)
        {
            return x.Map(e => e = 1/(1+Math.Exp(-e*k)));
        }

        override protected Vector<double> df(Vector<double> x)
        {
            double y;
            return x.Map(e =>
            {
                y = 1 + Math.Exp(-e * k);
                return y / ((1 + y)* (1 + y));
            });
        }
    }
}
