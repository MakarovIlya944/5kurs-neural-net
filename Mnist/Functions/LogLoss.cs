﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    /*
    Loss fuction:
    sum((y-x)^2)/2

    Derivative:
    -(y-x)
    */

    class LogLoss : ILossFunction<double>
    {
        public double call(Vector<double> calc, Vector<double> truly)
        {
            return -(calc.Map2((x, y) => y * Math.Log(x), truly).Sum());
        }

        public Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            return -(truly.PointwiseDivide(calc));
        }

        public Vector<double> call(Matrix<double> calc, Matrix<double> truly)
        {
            var c = calc.EnumerateRows();
            var t = truly.EnumerateRows();
            List<double> answer = new List<double>();

            for (int i = 0, n = calc.RowCount; i < n; i++)
                answer.Add((t.ElementAt(i) - c.ElementAt(i)).Map(x => x * x).Sum() / 2);

            return Vector<double>.Build.DenseOfEnumerable(answer);
        }

        public Matrix<double> backPropagation(Matrix<double> calc, Matrix<double> truly)
        {
            return -(truly - calc);
        }
    }
}
 