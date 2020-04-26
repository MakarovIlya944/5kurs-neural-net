using Mnist.Functions;
using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using Microsoft.Extensions.Logging;

namespace Mnist
{
    public class Model : IModel<double>
    {
        public List<Layer> layers;
        private List<Layer> reverseLayers;
        public List<Matrix<double>> weights;

        public int Deep
        {
            get => layers.Count;
        }

        public int LogEpoch
        {
            get => _logEpoch;
            set => _logEpoch = value;
        }

        static private int _logEpoch = 5;

        public Model(int deep, int width, double init, double b, int inputSize = 2, int outputSize = 1)
        {
            layers = new List<Layer>(deep);
            reverseLayers = new List<Layer>(deep);
            ReLU f1 = new ReLU();
            SoftMax f2 = new SoftMax(10);
            Sigmoid f3 = new Sigmoid();
            if (deep < 2)
                throw new Exception("Too few layers!");
            else if (deep == 2)
            {
                layers.Add(new Layer(width, inputSize, init, b, f1));
                layers.Add(new Layer(outputSize, width, init, b, f2));
            }
            else if (deep == 3)
            {
                layers.Add(new Layer(width, inputSize, init, b, f1));
                layers.Add(new Layer(width, width, init, b, f3));
                layers.Add(new Layer(outputSize, width, init, b, f2));
            }
            else
            {
                layers.Add(new Layer(inputSize, width, init, b, f1));
                for (int i = 1; i < deep - 2; i++)
                    layers.Add(new Layer(width, width, init, b, f1));
                layers.Add(new Layer(width, outputSize, init, b, f2));
            }

            //int j = 0;
            //foreach (Layer layer in layers)
            //    {
            //        Console.WriteLine($"------------------------#{j++}---------------------");
            //        Console.WriteLine(layer.matrix.ToString());
            //        Console.WriteLine(layer.bias.ToString());
            //    }
            for (int i = deep - 1; i >= 0; i--)
                reverseLayers.Add(layers[i]);
        }

        public Model(int deep, int[] width, double[] init, double[] bias, int inputSize = 2, int outputSize = 1)
        {
            layers = new List<Layer>(deep);
            reverseLayers = new List<Layer>(deep);
            ReLU f1 = new ReLU();
            SoftMax f2 = new SoftMax(10);
            Sigmoid f3 = new Sigmoid();
            if (deep < 2)
                throw new Exception("Too few layers!");
            else if (deep == 2)
            {
                layers.Add(new Layer(width[0], inputSize, init[0], bias[0], f1));
                layers.Add(new Layer(outputSize, width[0], init[1], bias[1], f2));
            }
            else
            {
                layers.Add(new Layer(width[0], inputSize, init[0], bias[0], f1));
                for (int i = 1; i < deep - 1; i++)
                    layers.Add(new Layer(width[i], width[i - 1], init[i], bias[i], f1));
                layers.Add(new Layer(outputSize, width[deep - 2], init[deep - 1], bias[deep - 1], f2));
            }

            //int j = 0;
            //foreach (Layer layer in layers)
            //    {
            //        Console.WriteLine($"------------------------#{j++}---------------------");
            //        Console.WriteLine(layer.matrix.ToString());
            //        Console.WriteLine(layer.bias.ToString());
            //    }

            for (int i = 0; i < deep - 1; i++)
                layers[i].RandomMatrix(0, 1E-1);

            for (int i = deep - 1; i >= 0; i--)
                reverseLayers.Add(layers[i]);
        }

        public void load(string filename)
        {
            throw new NotImplementedException();
        }

        public void save(string filename)
        {
            throw new NotImplementedException();
        }

        public void train(Data data, int epoch, double rate, ILossFunction<double> loss)
        {
            double maxLoss = -1, currentLoss, prevLoss = -1;
            Vector<double> currentLossVector = Vector<double>.Build.Dense(data.InputDataSize, 0);
            Matrix<double> signal, currentLossVectorString = Matrix<double>.Build.DenseOfRowVectors(currentLossVector);
            Matrix<double> answer = data.AllAnswer;
            Matrix<double>[] matrixStorage = new Matrix<double>[2];
            int currentFreeMatrixStorage = 0;

            foreach (var layer in layers)
                layer.InputDataSize = data.InputDataSize;

            if (layers[layers.Count - 1].activation.GetType() == typeof(SoftMax))
                layers[layers.Count - 1].activation.SetAnswer(answer);

            for (int i = 0; i < epoch; i++)
            {
                currentFreeMatrixStorage = 0;
                signal = data.AllSignal;
                currentLossVector.Clear();

                if (i % _logEpoch == 0)
                {
                    Console.WriteLine($"==================================================");
                    Console.WriteLine($"Epoch #{i}\n");
                }

                //Console.WriteLine("Forward signal through layers");
                for (int k = 0, n = layers.Count; k < n; k++)
                {
                    //Console.WriteLine($"--------------------------------------#{k}--------------------------------\nSignal:");
                    //Console.WriteLine(signal.ToString());
                    signal = layers[k].forward(signal);
                }
                //Normalize(signal);
                //signal = layers[layers.Count - 1].forward(signal);

                if (i % _logEpoch == 0)
                {
                    //Console.WriteLine($"--------------------------------------#{layers.Count-1}--------------------------------\nSignal:");
                    Console.WriteLine(signal.Transpose().ToString());
                    Console.WriteLine($"----------------------------------------------------------------------");
                    Console.WriteLine(answer.Transpose().ToString());
                }

                    currentLossVector = loss.call(signal, answer);
                //Console.WriteLine($"Current loss-vector: \n{currentLossVector.ToString()}");


                if (layers[layers.Count - 1].activation.GetType() == typeof(SoftMax))
                {
                    signal = layers[layers.Count - 1].backPropagation(layers[layers.Count - 2].A, rate, out matrixStorage[currentFreeMatrixStorage]);
                }
                else
                {
                    signal = loss.backPropagation(signal, answer);
                    //Console.WriteLine($"Current error: \n{error.ToString()}");
                    signal = layers[layers.Count - 1].backPropagation(layers[layers.Count - 2].A.Transpose(), signal, rate, out matrixStorage[currentFreeMatrixStorage]);
                }
                currentFreeMatrixStorage = 1;
                //Console.WriteLine("Backward signal through layers");
                for (int k = layers.Count - 2; k > 0; k--)
                {
                    //Console.WriteLine($"#{k}");
                    signal = layers[k].backPropagation(layers[k - 1].A.Transpose(), signal * layers[k + 1].matrix.Transpose(), rate, out matrixStorage[currentFreeMatrixStorage++]);
                    currentFreeMatrixStorage %= 2;
                    layers[k + 1].matrix -= matrixStorage[currentFreeMatrixStorage];
                }

                layers[0].backPropagation(data.AllSignal.Transpose(), signal * layers[1].matrix.Transpose(), rate, out matrixStorage[currentFreeMatrixStorage++]);
                currentFreeMatrixStorage %= 2;
                layers[1].matrix -= matrixStorage[currentFreeMatrixStorage++];
                currentFreeMatrixStorage %= 2;
                layers[0].matrix -= matrixStorage[currentFreeMatrixStorage];


                //index = 0;
                //    Console.WriteLine($"---------------------------------------------------");
                //foreach (Layer layer in layers)
                //{
                //    Console.WriteLine($"Layer #{index}\nW:{layer.matrix}");
                //}
                //Console.WriteLine($"---------------------------------------------------");
                if (i % _logEpoch == 0)
                {
                    currentLoss = currentLossVector.L2Norm();
                    currentLossVectorString = Matrix<double>.Build.DenseOfRowVectors(currentLossVector);
                    maxLoss = (currentLoss < maxLoss) ? maxLoss : currentLoss;
                    Console.WriteLine($"Previous loss: {prevLoss}\nCurrent loss: {currentLoss}\nMaxLoss: {maxLoss}");
                    Console.WriteLine(currentLossVectorString.ToString());
                    Console.WriteLine($"==================================================\n");
                    prevLoss = currentLoss;
                }

            }
        }

        public Vector<double> predict(Data data)
        {
            throw new NotImplementedException();
        }
    }
}