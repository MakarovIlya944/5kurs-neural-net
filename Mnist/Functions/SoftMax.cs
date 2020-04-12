using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    public class SoftMax : BaseActivation
    {
        override protected Vector<double> f(Vector<double> x)
        {
            double s = x.Map(e => Math.Exp(e)).Sum();
            return x.Map(e => e = Math.Exp(e)/s);
        }

        override public Matrix<double> call(Matrix<double> m)
        {
            List<Vector<double>> a = new List<Vector<double>>(m.RowCount);
            foreach (Vector<double> v in m.EnumerateRows())
                a.Add(f(v));

            return Matrix<double>.Build.DenseOfRowVectors(a);
        }

        override public Matrix<double> backPropagation(Matrix<double> m)
        {
            List<Vector<double>> a = new List<Vector<double>>(m.RowCount);

            for (int i = 0, n = m.RowCount; i < n; i++)
            {
                
            }

            foreach (Vector<double> v in m.EnumerateRows())
                a.Add(df(v));

            return Matrix<double>.Build.DenseOfRowVectors(a);
        }
    }
}
