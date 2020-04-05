using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public class Layer<T> : ILayer<T>
    {
        public int width;
        public IActivationFunction<T> activation;
        public IWeightsMatrix<T> matrix;
        public INode<T> nodes;
        public ILayerInfo<T> info;

        public T[] forward(T[] input)
        {
            throw new NotImplementedException();
        }

        public IWeightsMatrix<T> getWeights()
        {
            throw new NotImplementedException();
        }
    }
}
