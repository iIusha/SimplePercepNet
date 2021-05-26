using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Drawing;

namespace NeuronNet
{
    public static class Misc
    {
        public static double Exp(double val)
        {
            long tmp = (long)(1512775 * val + 1072632447);
            return BitConverter.Int64BitsToDouble(tmp << 32);
        }
        public static double[] ToDouble(int[] b)
        {
            var c = new double[b.Length];
            for (int i = 0; i < b.Length; i++) c[i] = Convert.ToDouble(b[i]);
            return c;
        }
        public static double[] smp(double[] a, double[] b)
        {
            var c = new double[a.Length];
            for (int i = 0; i < a.Length; i++) c[i] = a[i] * b[i];
            return c;
        }
        public static double[,] nw(double[] a, double[] b)
        {
            var c = new double[a.Length, b.Length];

            for (int i = 0; i < a.Length; i++)
                for (int j = 0; j < b.Length; j++) c[i, j] = a[i] * b[j];
            return c;
        }
        public static double[] sigma(double[] z)
        {
            var r = new double[z.Length];
            for (int b = 0; b < z.Length; b++) r[b] = 1.0d / (1.0d + Exp(-z[b]));
            return r;
        }
        public static double[] cd(double[] a, int[] b)
        {
            var c = new double[a.Length];
            for (int i = 0; i < a.Length; i++) c[i] = a[i] - b[i];
            return c;
        }
        public static double[] sd(double[] z)
        {
            var sig = sigma(z);
            for (int i = 0; i < z.Length; i++) z[i] = sig[i] * (1 - sig[i]);
            return z;
        }
        public static double[,] Transpose(double[,] a)
        {
            var b = new double[a.GetLength(1), a.GetLength(0)];
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++) b[j, i] = a[i, j];
            return b;
        }
        public static double[] Vector(double[,] a, double[] b)
        {
            var c = new double[a.GetLength(0)];
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++) c[i] += a[i, j] * b[j];
            return c;
        }
        public static double[] imp(double[] a, double[] b)
        {
            var c = new double[a.Length];
            for (int i = 0; i < a.Length; i++) c[i] = a[i] + b[i];
            return c;
        }
        public static double[,] imp(double[,] a, double[,] b)
        {
            var c = new double[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++) c[i, j] = a[i, j] + b[i, j];
            return c;
        }
    }

    public partial class Samples
    {
        public int[] bitmap { get; set; }
        public int[] letter { get; set; }
    }

    public class Data
    {
        public List<Samples> samples;
        public Data(string dir)
        {
            samples = new List<Samples>();
            string[] paths = Directory.GetFiles(dir);
            foreach (string path in paths)
            {
                Bitmap bmp = new Bitmap(path);
                var bitmap = new int[bmp.Height * bmp.Width];
                for (int y = 0; y < bmp.Height; y++)
                    for (int x = 0; x < bmp.Width; x++)
                        bitmap[y * bmp.Width + x] = bmp.GetPixel(x, y).R > 0 ? 0 : 1;
                Samples smpl = new Samples{ bitmap = bitmap };
                string ltr = path.Split('_')[1].Split('.')[0];
                int index = Char.ToLower(ltr[0]) - 97;
                var arr = new int[26];
                arr[index] = 1;
                smpl.letter = arr;
                samples.Add(smpl);
            }
        }

        public List<Samples> ToBatch(int index, int count) { return this.samples.GetRange(index, count).ToList(); }
        public void Shuffle() { this.samples = this.samples.OrderBy(i => Guid.NewGuid()).ToList(); }
        public int GetLength() { return samples.Count; }

    }

    class Layer
    {
        public double[] activations;
        public double[,] weights;
        public double[,] nw;
        public double[,] dnw;
        public double[] biases;
        public double[] nb;
        public double[] dnb;
        public double[] z;
        public Layer(int[] sizes, int num)
        {
            Random r = new Random();
            double min = -1.1;
            double max = 1.1;
            activations = new double[sizes[num]];
            biases = new double[sizes[num]];
            nb = new double[sizes[num]];
            for (int b = 0; b < biases.Length; b++) biases[b] = r.NextDouble() * (max - min) + min;
            if (num > 0)
            {
                weights = new double[sizes[num], sizes[num - 1]];
                for (int w0 = 0; w0 < weights.GetLength(0); w0++)
                    for (int w1 = 0; w1 < weights.GetLength(1); w1++)
                        weights[w0, w1] = r.NextDouble() * (max - min) + min;
            }
            else weights = null;
            nw = weights;
        }

        public void NbZero()
        {
            if (this.weights != null)
            {
                this.dnw = new double[0, 0];
                this.dnb = new double[this.biases.Length];
            }

        }
        public void MergeNb()
        {
            this.nb = Misc.imp(this.nb, this.dnb);
            this.nw = Misc.imp(this.nw, this.dnw);
        }
        public void Mprov(int size, double lr)
        {
            for (int i = 0; i < this.weights.GetLength(0); i++)
                for (int j = 0; j < this.weights.GetLength(1); j++)
                    weights[i, j] = weights[i, j] - (lr / size) * nw[i, j];
            for (int k = 0; k < this.biases.Length; k++)
                biases[k] -= (double)(lr / size * nb[k]);
        }
    }

    class Network
    {
        public Layer[] layers;
        public int epochs { get; set; }
        public int batch_size { get; set; }
        public double learning_rate { get; set; }
        public Network(int[] sizes)
        {
            this.layers = new Layer[sizes.Length];
            for (int i = 0; i < sizes.Length; i++) layers[i] = new Layer(sizes, i);
        }

        public void Train(Data training_data, Data test_data)
        {
            int buffer = training_data.GetLength();

            for (int e = 0; e < this.epochs; e++)
            {
                training_data.Shuffle();
                for (int j = 0; j < buffer / batch_size; j++) this.Gradient(training_data.ToBatch((buffer / batch_size) * j, batch_size));
                var r = this.Evaluate(test_data);
                var t = test_data.samples.Count;
                Console.WriteLine(String.Format("Epoch {0}:\tRight/Total: {1}/{2}\t Acc: {3}", e, r, t, (float) r / t));
            }
        }
        public void Gradient(List<Samples> batch)
        {
            foreach (Samples sample in batch)
            {
                this.NbZero();
                this.Backprop(sample.bitmap, sample.letter);
                for (int i = 1; i < layers.Length; i++) this.layers[i].MergeNb();
            }
            for (int i = 1; i < layers.Length; i++) this.layers[i].Mprov(batch.Count, this.learning_rate);
        }

        public void NbZero() { for (int l = 0; l < layers.Length; l++) layers[l].NbZero(); }

        public void Backprop(int[] bitmap, int[] answer)
        {
            this.layers[0].activations = Misc.ToDouble(bitmap);
            for (int l = 1; l < layers.Length; l++)
            {
                var z = Misc.imp(Misc.Vector(this.layers[l].weights, this.layers[l - 1].activations), this.layers[l].biases);
                this.layers[l].activations = Misc.sigma(z);
                this.layers[l].z = z;
            }
            var delta = Misc.smp(Misc.cd(layers[layers.Length - 1].activations, answer), Misc.sd(layers[layers.Length - 1].activations));
            layers[layers.Length - 1].dnb = delta;
            layers[layers.Length - 1].dnw = Misc.nw(delta, layers[layers.Length - 2].activations);

            for (int l = layers.Length - 2; l > 0; l--)
            {
                delta = Misc.smp(Misc.Vector(Misc.Transpose(layers[l + 1].weights), delta), Misc.sd(layers[l].z));
                layers[l].dnb = delta;
                layers[l].dnw = Misc.nw(delta, layers[l - 1].activations);
            }
        }

        public int Evaluate(Data td)
        {
            var r = 0;
            var t = 0;
            foreach (Samples sample in td.samples)
            {
                var a = this.predict(sample.bitmap);
                t++;
                var max = a.ToList().IndexOf(a.Max());
                var answer = sample.letter.ToList().IndexOf(sample.letter.Max());
                if (max == answer) r++;
            }
            return r;
        }

        public double[] predict(int[] data)
        {
            layers[0].activations = Misc.ToDouble(data);
            for (int l = 1; l < this.layers.Length; l++)
                layers[l].activations = Misc.sigma(Misc.imp(Misc.Vector(this.layers[l].weights, this.layers[l - 1].activations), this.layers[l].biases));
            return layers[layers.Length - 1].activations;
        }
    }


    class Program
    {
        static void Main()
        {
            Network network = new Network(new int[] { 4096, 104, 26 })
            {
                batch_size = 10,
                epochs = 300,
                learning_rate = 0.2
            };
            Data test = new Data(@"E:\Shit\DATASET\");
            network.Train(test, test);
            Console.ReadKey();
        }
    }
}
