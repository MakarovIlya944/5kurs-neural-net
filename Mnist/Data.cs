using System;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public class Data<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Vector<T>[] signal;
        public Vector<T>[] answer;

        public Matrix<T> AllSignal { get => Matrix<T>.Build.DenseOfColumnVectors(signal); }
        public Matrix<T> AllAnswer { get => Matrix<T>.Build.DenseOfColumnVectors(answer); }

        public int InputDataSize { get => input; }
        private int input;

        public Data(Vector<T>[] signal, Vector<T>[] answer)
        {
            this.signal = signal;
            this.answer = answer;
            this.input = signal.Length;
        }

        public void Open(string filename)
        {

        }
    }
}
