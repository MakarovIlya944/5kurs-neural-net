using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    class ReLU : BaseActivation
    {
        double scale = 1;

        public ReLU(double scale)
        {
            this.scale = scale;
        }

        public ReLU()
        {
        }

        override protected Vector<double> f(Vector<double> x)
        {
            return x.Map(e => e = Math.Max(0, e) * scale);
        }

        override protected Vector<double> df(Vector<double> x)
        {
            return x.Map(e => e = e > 0 ? scale : 0);
        }
    }
}
