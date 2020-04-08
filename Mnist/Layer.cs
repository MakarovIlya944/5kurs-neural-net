using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.Runtime.InteropServices.WindowsRuntime;

namespace Mnist
{
    public class Layer : ILayer<double>
    {
        public int width;

        public IActivationFunction<double> activation;

        public Matrix<double> matrix;
        public Vector<double> bias;

        private Vector<double> z; // v * matrix + bias

        //public List<INode<double>> nodes;

        public ILayerInfo<double> info;

        public Layer(int nodesCount, int inputVectorSize, double init, double b, IActivationFunction<double> activation)
        {
            width = nodesCount;
            this.activation = activation;
            matrix = Matrix<double>.Build.Dense(inputVectorSize, nodesCount, init);
            bias = Vector<double>.Build.Dense(nodesCount, b);
            //nodes = new List<INode<double>>(nodesCount);
            //for (int i = 0; i < nodesCount; i++)
            //    nodes.Add(new Node<double>(activation));
        }

        public Vector<double> forward(Vector<double> input)
        {
            z = matrix * input + bias;
            return activation.call(z);
        }

        /// <summary>
        /// Calculate back propagation and change weights matrix
        /// </summary>
        /// <param name="input"></param>
        /// <param name="rate"></param>
        /// <param name="layerNum">0 - input layer, 1 - hidden layer, 2 - output layer</param> TODO refactor
        /// <returns></returns>
        public Vector<double> backPropagation(Vector<double> input, double rate, int layerNum)
        {
            Matrix<double> ret;
            switch (layerNum)
            {
                case 0:

                    break;
                case 1:
                    break;
                case 2:
                    //ret = matrix.Transpose() * input.PointwiseMultiply(activation.backPropagation(z));
                    //matrix -= ret * rate;
                    break;
                default:
                    throw new NotImplementedException();
            }
            throw new NotImplementedException();
        }
    }
}
