using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public interface IActivationFunction<T>
    {
        public T call(T v);
        public T backPropagation(T v);
    }
}
