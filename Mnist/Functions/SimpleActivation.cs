using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist.Functions
{
    class SimpleActivation : IActivationFunction<float>
    {
        public float backPropagation(float v)
        {
            throw new NotImplementedException();
        }

        public float call(float v)
        {
            return v;
        }
    }
}
