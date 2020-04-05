using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public class Node<T> : INode<T>
    {
        public List<Node<T>> input;
        public List<Node<T>> output;
        public IActivationFunction<T> activation;
        public T forward()
        {
            throw new NotImplementedException();
        }
    }
}
