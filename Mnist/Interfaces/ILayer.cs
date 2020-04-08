using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.Text;

namespace Mnist
{
    public interface ILayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Vector<T> forward(Vector<T> input);
    }
}
