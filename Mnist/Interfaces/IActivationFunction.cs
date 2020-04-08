using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist
{
    public interface IActivationFunction<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Vector<T> call(Vector<T> v);
        public Vector<T> backPropagation(Vector<T> v);
    }
}
