using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;

namespace Mnist.Functions
{
    class ScaleActivation : IActivationFunction<double> 
    {
        public Vector<double> backPropagation(Vector<double> v)
        {
            throw new NotImplementedException();
        }

        public Matrix<double> backPropagation(Matrix<double> v)
        {
            throw new NotImplementedException();
        }

        public Vector<double> call(Vector<double> v)
        {
            return f(v);
        }

        public Matrix<double> call(Matrix<double> m)
        {
            List<Vector<double>> a = new List<Vector<double>>(m.RowCount);
            foreach (Vector<double> v in m.EnumerateRows())
                a.Add(f(v));

            return  Matrix<double>.Build.DenseOfRowVectors(a);
        }

        private Vector<double> f(Vector<double> x)
        {
            double max = x[0];
            foreach (double e in x)
                max = Math.Max(max, e);
            return x.Map(e => e = e > 0 ? e / max : 0);
        }
    }
}
