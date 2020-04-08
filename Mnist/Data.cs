using System;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public class Data<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Vector<T> signal;
        public Vector<T> answer;

        public void Open(string filename)
        {

        }
    }
}
