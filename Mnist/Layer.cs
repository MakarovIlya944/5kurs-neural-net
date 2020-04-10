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

        public int InputDataSize { 
            set => SetUp(value); 
            get => B.ColumnCount; 
        }

        private Matrix<double> B; // bias as matrix
        private Vector<double> zVector; // v * matrix + bias
        private Matrix<double> zMatrix; // v * matrix + bias

        //public List<INode<double>> nodes;  // for non full-connected

        public ILayerInfo<double> info;

        public Layer(int nodesCount, int inputVectorSize, double init, double b, IActivationFunction<double> activation)
        {
            width = nodesCount;
            this.activation = activation;
            matrix = Matrix<double>.Build.Dense(inputVectorSize, nodesCount, init);
            bias = Vector<double>.Build.Dense(nodesCount, b);

            //nodes = new List<INode<double>>(nodesCount); // for non full-connected
            //for (int i = 0; i < nodesCount; i++)
            //    nodes.Add(new Node<double>(activation));
        }

        private void SetUp(int size)
        {
            Vector<double>[] tempBias = new Vector<double>[size];
            for (int i = 0; i < size; i++)
                tempBias[i] = Vector<double>.Build.DenseOfVector(bias);
            B = Matrix<double>.Build.DenseOfColumnVectors(tempBias);
        }

        public Vector<double> forward(Vector<double> input)
        {
            zVector = matrix * input + bias;
            return activation.call(zVector);
        }

        public Matrix<double> forward(Matrix<double> input)
        {
            zMatrix = matrix * input + B;
            Console.WriteLine($"Z: \n{zMatrix.ToString()}");
            Matrix<double> t = activation.call(zMatrix);
            Console.WriteLine($"A: \n{t.ToString()}");
            return t;
        }

        /// <summary>
        /// Calculate back propagation and change weights matrix
        /// </summary>
        /// <param name="input"></param>
        /// <param name="rate"></param>
        /// <param name="layerNum">0 - input layer, 1 - hidden layer, 2 - output layer</param> TODO refactor
        /// <returns></returns>
        public Vector<double> backPropagation(Vector<double> input, double rate)
        {
            //Matrix<double> ret;
            throw new NotImplementedException();
        }

        public Matrix<double> backPropagation(Matrix<double> input, double rate)
        {
            Matrix<double> ret = activation.backPropagation(zMatrix);
            matrix -= matrix.Transpose() * ret;
            return ret;
        }
    }
}
