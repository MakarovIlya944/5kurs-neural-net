using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    class Data<T> : IData<T>
    {
        public List<T> signal;
    }
}
