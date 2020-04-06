using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist.Functions
{
    class SimpleActivation<T> : IActivationFunction<T>
    {
        public T backPropagation(T v)
        {
            throw new NotImplementedException();
        }

        public T call(T v)
        {
            return v;
        }
    }
}
