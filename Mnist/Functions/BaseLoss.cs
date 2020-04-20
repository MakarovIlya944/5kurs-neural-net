using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;

namespace Mnist.Functions
{
    public class BaseLoss : ILossFunction<double> 
    {
        virtual public double call(Vector<double> calc, Vector<double> truly)
        {
            throw new NotImplementedException();
        }

        virtual public Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            throw new NotImplementedException();
        }

        public Vector<double> call(Matrix<double> calc, Matrix<double> truly)
        {
            var c = calc.EnumerateRows();
            var t = truly.EnumerateRows();
            List<double> answer = new List<double>();

            for (int i = 0, n = calc.RowCount; i < n; i++)
                answer.Add(call(c.ElementAt(i), t.ElementAt(i)));

            return Vector<double>.Build.DenseOfEnumerable(answer);
        }

        public Matrix<double> backPropagation(Matrix<double> calc, Matrix<double> truly)
        {
            var c = calc.EnumerateRows();
            var t = truly.EnumerateRows();
            List<Vector<double>> answer = new List<Vector<double>>(truly.RowCount);

            for (int i = 0, n = calc.RowCount; i < n; i++)
                answer.Add(backPropagation(c.ElementAt(i), t.ElementAt(i)));

            return Matrix<double>.Build.DenseOfRowVectors(answer);
        }
    }
}
