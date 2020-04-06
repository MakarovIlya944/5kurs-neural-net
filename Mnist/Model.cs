using Mnist.Functions;
using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public class Model<T> : IModel<T>
    {
        public List<Layer<T>> layers;
        public List<WeightsMatrix<T>> weights;

        public int Deep { get => layers.Count; }

        public Model(int deep, int width, T b)
        {
            layers = new List<Layer<T>>(deep);
            SimpleActivation<T> f = new SimpleActivation<T>();
            for (int i = 0; i < deep; i++)
                layers.Add(new Layer<T>(width, width, b, f));
            

        }

        public void load(string filename)
        {
            throw new NotImplementedException();
        }

        public T[] predict(IData<T> data)
        {
            throw new NotImplementedException();
        }

        public void save(string filename)
        {
            throw new NotImplementedException();
        }

        public void train(IData<T>[] data, int epochs, ILossFunction<T> loss, ILearningRate<T> rate)
        {
            throw new NotImplementedException();
        }
    }
}
