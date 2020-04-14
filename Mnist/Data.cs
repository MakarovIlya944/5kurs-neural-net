using System;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Linq;

namespace Mnist
{
    public class Data
    {
        public Vector<double>[] signal;
        public Vector<double>[] answer;

        public Matrix<double> AllSignal { get => Matrix<double>.Build.DenseOfRowVectors(signal); }
        public Matrix<double> AllAnswer { get => Matrix<double>.Build.DenseOfRowVectors(answer); }

        public int InputDataSize { get => input; }
        public int input, output;

        public Data(Vector<double>[] signal, Vector<double>[] answer)
        {
            this.signal = signal;
            this.answer = answer;
            this.input = signal.Length;
            this.output = answer.Length;
        }

        public Data(int input, int output)
        {
            this.input = input;
            this.output = output;
        }

        public void OpenMnist(string filenamelabel, string filenameimage, double percent = 1.0)
        {
            byte[] byteLabels = File.ReadAllBytes(filenamelabel), byteImages = File.ReadAllBytes(filenameimage);

            int len;
            
            len = BitConverter.ToInt32(byteLabels.Skip(4).Take(4).Reverse().ToArray());
            len = (int)(len * percent);

            List<Vector<double>> listImages = new List<Vector<double>>(len);
            List<Vector<double>> listLabels = new List<Vector<double>>(len);

            Vector<double> v = Vector<double>.Build.Dense(output);
            for (int i = 8, j = -1; i < len; i++)
            {
                j = byteLabels[i];
                v[j] = 1;
                listLabels.Add(v);
                v = Vector<double>.Build.Dense(output);
            }

            int rows = BitConverter.ToInt32(byteImages.Skip(8).Take(4).Reverse().ToArray());
            int columns = BitConverter.ToInt32(byteImages.Skip(12).Take(4).Reverse().ToArray());
            input = rows * columns;

            v = Vector<double>.Build.Dense(input);

            for (int i = 16; i < len; i += input)
            {
                listImages.Add(Vector<double>.Build.DenseOfEnumerable(
                    byteImages
                    .Skip(i).Take(input).Reverse()
                    .Select(x => (double)(int)x)));
            }

            answer = listLabels.ToArray();
            signal = listImages.ToArray();
        }
    }
}
