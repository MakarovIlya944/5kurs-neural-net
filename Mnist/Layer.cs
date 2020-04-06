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
        public List<INode<T>> nodes;
        public ILayerInfo<T> info;

        public Layer(int nodesCount, int inputVectorSize, T bias, IActivationFunction<T> activation)
        {
            width = nodesCount;
            this.activation = activation;
            matrix = new WeightsMatrix<T>(nodesCount, inputVectorSize, bias);
            nodes = new List<INode<T>>(nodesCount);
            for (int i = 0; i < nodesCount; i++)
                nodes.Add(new Node<T>(activation));
        }

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
