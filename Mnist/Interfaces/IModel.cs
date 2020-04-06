using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public interface IModel<T>
    {
        public void save(string filename);
        public void load(string filename);
        public void train(IData<T>[] data, int epochs, ILossFunction<T> loss, ILearningRate<T> rate);
        public T[] predict(IData<T> data);
    }
}
