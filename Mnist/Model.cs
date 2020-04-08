using Mnist.Functions;
using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.Text;

namespace Mnist
{
    public class Model : IModel<double> 
    {
        public List<Layer> layers;
        private List<Layer> reverseLayers;
        public List<Matrix<double>> weights;

        public int Deep { get => layers.Count; }

        private int LogEpoch = 1;

        public Model(int deep, int width, double init, double b)
        {
            layers = new List<Layer>(deep);
            SimpleActivation<double> f = new SimpleActivation<double>();
            for (int i = 0; i < deep; i++)
                layers.Add(new Layer(width, width, init, b, f));
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

        public void train(Data<double>[] data, int epoch, double rate, ILossFunction<double> loss)
        {
            double maxLoss = -1;
            double currentLoss = -1;
            double sumCurrentLoss = 0;
            for (int i = 0; i < epoch; i++)
            {
                if (i % LogEpoch == 0)
                    Console.WriteLine($"Epoch #{i}\nMaxLoss {maxLoss}\nCurrentLoss {currentLoss}");
                sumCurrentLoss = 0;
                foreach (var e in data)
                {
                    Vector<double> d = e.signal;
                    foreach (var layer in layers)
                        d = layer.forward(d);

                    sumCurrentLoss += loss.call(d, e.answer);

                    Vector<double> v = loss.backPropagation(d, e.answer);
                }
                currentLoss = sumCurrentLoss / (double)data.Length;
                maxLoss = (currentLoss < maxLoss) ? maxLoss : currentLoss;
            }
        }

        public Vector<double> predict(Data<double> data)
        {
            throw new NotImplementedException();
        }
    }
}
