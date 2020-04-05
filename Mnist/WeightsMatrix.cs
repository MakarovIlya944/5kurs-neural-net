using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Mnist
{
    public class WeightsMatrix : IWeightsMatrix<float>
    {
        public float[][] w;
        public int nodesCount;
        public int inputVectorSize;
        public float[] bias;

        public WeightsMatrix(int nodesCount, int inputVectorSize, float b)
        {
            this.nodesCount = nodesCount;
            this.inputVectorSize = inputVectorSize;
            float t = 1;
            this.w = Enumerable.Repeat(Enumerable.Repeat(t, inputVectorSize).ToArray(), nodesCount).ToArray();
            this.bias = Enumerable.Repeat(b, nodesCount).ToArray();
        }
    }
}
