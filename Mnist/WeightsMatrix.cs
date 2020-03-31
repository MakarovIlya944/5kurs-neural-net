using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public class WeightsMatrix<T> : IWeightsMatrix<T>
    {
        public T[][] w;
        public int nodesCount;
        public int inputVectorSize;
        public T[] bias;
    }
}
