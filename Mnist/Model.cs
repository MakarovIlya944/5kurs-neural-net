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

        public int Deep { get => layers.Count; }

        private int _logEpoch = 1;

        public Model(int deep, int width, double init, double b, int inputSize = 2, int outputSize = 1)
        {
            layers = new List<Layer>(deep);
            reverseLayers = new List<Layer>(deep);
            SimpleActivation<double> f = new SimpleActivation<double>();

            if (deep < 3)
                throw new Exception("Too few layers!");
            else if (deep == 3)
            {
                layers.Add(new Layer(inputSize, inputSize, init, b, f));
                layers.Add(new Layer(inputSize, outputSize, init, b, f));
                layers.Add(new Layer(outputSize, outputSize, init, b, f));
            }
            else
            {
                layers.Add(new Layer(inputSize, width, init, b, f));
                for (int i = 1; i < deep - 2; i++)
                    layers.Add(new Layer(width, width, init, b, f));
                layers.Add(new Layer(width, outputSize, init, b, f));
            }

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

        public void train(Data<double> data, int epoch, double rate, ILossFunction<double> loss)
        {
            int j = 0;

            double maxLoss = -1, currentLoss = -1;
            Vector<double> currentLossVector = Vector<double>.Build.Dense(data.InputDataSize, 0);
            Matrix<double> signal = data.AllSignal;
            Matrix<double> answer = data.AllAnswer;
            foreach (var layer in layers)
                layer.InputDataSize = data.InputDataSize;
            for (int i = 0; i < epoch; i++)
            {
                if (i % _logEpoch == 0)
                    Console.WriteLine($"Epoch #{i}\nMaxLoss {maxLoss}\nCurrentLoss {currentLoss}\n");
                currentLossVector.Clear();

                j = 0;
                Console.WriteLine("Forward signal through layers");
                foreach (var layer in layers)
                {
                    Console.WriteLine($"--------------------------------------#{++j}--------------------------------\nSignal:");
                    Console.WriteLine(signal.ToString());
                    signal = layer.forward(signal);
                }

                currentLossVector += loss.call(signal, answer);
                Console.WriteLine($"Current loss-vector: \n{currentLossVector.ToString()}");

                Matrix<double> error = loss.backPropagation(signal, answer);
                Console.WriteLine($"Current error: \n{error.ToString()}");

                j = 0;
                Console.WriteLine("Backward signal through layers");
                foreach (var layer in reverseLayers)
                { 
                    error = layer.backPropagation(error, rate);
                    Console.WriteLine($"#{++j}");
                    Console.WriteLine(error.ToString());
                }

                currentLoss = currentLossVector.L2Norm();
                maxLoss = (currentLoss < maxLoss) ? maxLoss : currentLoss;
                Console.WriteLine($"Current loss: {currentLoss}");
            }
        }

        public Vector<double> predict(Data<double> data)
        {
            throw new NotImplementedException();
        }
    }
}
