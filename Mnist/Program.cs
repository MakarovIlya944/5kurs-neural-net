using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using Mnist.Functions;
using Mnist.Pictures;
using NLog;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Mnist
{
    public class Program
    {
        private static Logger logger = LogManager.GetLogger("console");

        public static Data allData = MnistConverter.OpenMnist(@"D:\Projects\Mnist\data\train-labels.idx1-ubyte", @"D:\Projects\Mnist\data\train-images.idx3-ubyte", 1);
        public static string modelPath = @"D:\Projects\Mnist\NeuralNet\Ready\Models\Model2";
        public static int trainDataSize = 50000;
        public static int trainEpoch = 15;
        public static int trainBatch = 10;
        public static double trainTeachRate = 5;
        public static double trainMatrixRandomCenter = 0;
        public static double trainMatrixRandomOffset = 1E+1;
        public static double trainReLUCoef = 1;


        static void Main(string[] args)
        {
            Console.WriteLine("Hello World! Version 3\n");
            //Train();
            Predict();
            Console.WriteLine("Good bye World!");
            Console.ReadLine();
        }

        public static int PredictedIndex(Vector<double> v)
        {
            double max = v[0];
            int a = 0;
            v.MapIndexed((i, x) => {
                if (max < x) { a = i; max = x; }; 
                return x; });
            return a;
        }

        public static void Predict()
        {
            Model m = new Model();
            m.Load(modelPath);
            Data data = allData.Skip(trainDataSize);
            foreach (var item in m.layers)
            {
                item.InputDataSize = data.InputDataSize;
            }
            Matrix<double> pred = m.Predict(data), ans = data.AllAnswer;
            Vector<double> l2error = Vector<double>.Build.Dense((pred - ans).EnumerateRows().Select(x => x.L2Norm()).ToArray());

            logger.Error("Train:");
            logger.Error($"L2 errors: {l2error.ToString()}");
            logger.Error($"Max error: {l2error.Max()}:");

            int[] right = new int[10], falsive = new int[10];
            int a, b;
            for (int i = 0; i < pred.RowCount; i++)
            {
                a = PredictedIndex(ans.Row(i));
                b = PredictedIndex(pred.Row(i));
                if (a == b) right[a]++;
                else falsive[a]++;
            }
            logger.Error("i:\ttrue/false\t\ttrue/all\t\tfalse/all\t\ttrue/false");
            double truly, falsly, trfa, all;
            for (int i = 0; i < 10; i++)
            {
                all = right[i] + falsive[i];
                truly = right[i] / all;
                falsly = falsive[i] / all;
                trfa = right[i] / (double)falsive[i];
                logger.Error(
                    String.Format("{0}: {1:00}/{2:00}\t\t{3:0.00}%\t\t{4:0.00}%\t\t{5:0.00}", i, right[i], falsive[i], truly*100, falsly * 100, trfa));
            }
        }

        public static Vector<double> Predict(byte[] image)
        {
            Model m = new Model();
            m.Load(modelPath);
            Data data = new Data(new Vector<double>[1] { Vector<double>.Build.DenseOfArray(image.Select(x => 255.0-(double)(int)x).ToArray()) }, new Vector<double>[0]);
            MnistConverter.SavePicture(data, 0);
            foreach (var item in m.layers)
            {
                item.InputDataSize = data.InputDataSize;
            }
            return m.Predict(data).Row(0);
        }

        public static void Train()
        {
            Data data = allData.Take(trainDataSize);

            int inputSize = 28 * 28, outputSize = 10, deep = 5;
            List<int> width = new List<int>() { inputSize, inputSize / 2, inputSize / 4, inputSize / 8 };
            //double[] init = new double[5] { 1, 1, 1, 1, 1 }, bias = new double[5] { 1, 1, 1, 1, 1 };
            ILossFunction<double> loss = new LogLoss();

            Model m = new Model(deep, width.ToArray(), 1, 1, inputSize, outputSize, true, trainMatrixRandomCenter, trainMatrixRandomOffset, trainReLUCoef);

            m.LogEpoch = 2;

            m.Train(data, trainEpoch, trainBatch, trainTeachRate, loss);
            m.Save(modelPath);
        }
    }
}
