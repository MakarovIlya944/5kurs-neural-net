using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Mnist
{
    public class WeightsMatrix<T> : IWeightsMatrix<T>
    {
        public T[][] w;
        public int nodesCount;
        public int inputVectorSize;
        public T[] bias;

        public WeightsMatrix(int nodesCount, int inputVectorSize, T b)
        {
            this.nodesCount = nodesCount;
            this.inputVectorSize = inputVectorSize;
            w = Enumerable.Repeat(new T[inputVectorSize], nodesCount).ToArray();
            bias = Enumerable.Repeat(b, nodesCount).ToArray();
        }
    }
}
