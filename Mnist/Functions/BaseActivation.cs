using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;

namespace Mnist.Functions
{
    public class BaseActivation : IActivationFunction<double> 
    {
        public Vector<double> backPropagation(Vector<double> v)
        {
            return df(v);
        }

        virtual public Matrix<double> backPropagation(Matrix<double> m)
        {
            List<Vector<double>> a = new List<Vector<double>>(m.RowCount);
            foreach (Vector<double> v in m.EnumerateRows())
                a.Add(df(v));

            return Matrix<double>.Build.DenseOfRowVectors(a);
        }

        public Vector<double> call(Vector<double> v)
        {
            return f(v);
        }

        virtual public Matrix<double> call(Matrix<double> m)
        {
            List<Vector<double>> a = new List<Vector<double>>(m.RowCount);
            foreach (Vector<double> v in m.EnumerateRows())
                a.Add(f(v));

            return Matrix<double>.Build.DenseOfRowVectors(a);
        }

        virtual protected Vector<double> f(Vector<double> x)
        {
            throw new NotImplementedException();
        }

        virtual protected Vector<double> df(Vector<double> x)
        {
            throw new NotImplementedException();
        }
    }
}
