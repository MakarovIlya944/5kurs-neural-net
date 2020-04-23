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

        protected override Vector<double> f(Vector<double> x)
        {
            return x.Map(e => Math.Max(0, e) * scale);
        }

        protected override Vector<double> df(Vector<double> x)
        {
            return x.Map(e =>  e > 0 ? scale : 0);
        }
    }
}
