using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public interface ILossFunction<T>
    {
        public T call(T val);
    }
}
