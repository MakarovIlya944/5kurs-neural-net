using System;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public interface IModel<T> where T : struct, IEquatable<T>, IFormattable
    {
        public void save(string filename);
        public void load(string filename);
        public void train(Data data, int epochs, T rate, ILossFunction<T> loss);
        public Vector<T> predict(Data data);
    }
}
