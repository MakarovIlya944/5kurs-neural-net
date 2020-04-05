using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public interface ILayer<T>
    {
        public IWeightsMatrix<T> getWeights();
        public T[] forward(T[] input);
    }
}
