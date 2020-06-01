using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BL
{
    /// <summary>
    ///Нейроная сеть
    /// </summary>
    public class NeuralNetwork
    {
        

        /// <summary>
        /// Слои нейроной сети
        /// </summary>
        public List<Layer> Layers { get; }

        /// <summary>
        /// Топология сети
        /// </summary>
        public Topology Topology { get; }


        /// <summary>
        /// Создание нейронки
        /// </summary>
        /// <param name="topology">Топология нейронной сети</param>
        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();
            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();


        }


        /// <summary>
        /// Прогонка сети
        /// </summary>
        /// <param name="inputSignals"></param>
        /// <returns></returns>
        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount ==1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }
        /// <summary>
        /// обучение
        /// </summary>
        /// <param name="dataset"> на чем обучать соответственно 1-ожидаемый результат, 2- вх параметры</param>
        /// <param name="epoch"> кол-во эпох - сколько раз обучить на этом dataset</param>
        /// <returns></returns>
        public double Learn(double[] expected, double[,] inputs, int epoch)
        {
            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);

                    error += Backpropagation(output, input);
                    if (i % 100 == 0)
                    {
                        Debug.WriteLine(i);
                    }

                }
            }
            var result = error / epoch;
            return result;
        }

        /// <summary>
        /// Получить строку
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        public static double[] GetRow(double[,] matrix, int row)
        {
            var columns = matrix.GetLength(1);
            var array = new double[columns];
            for (int i = 0; i < columns; i++)
            {
                array[i] = matrix[row, i];
            }
            return array;
        }
        //TODO:надо описать
        /// <summary>
        /// масшабирование
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        private double[,] Scalling(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];
            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, column];
                    if (item < min) 
                    {
                        min = item;
                    }
                    if (item > max) 
                    {
                        max = item;
                    }
                }

                var divider = (max - min);
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - min) / divider;
                }
            }
            return result;
        }

        /// <summary>
        /// Алгоритм нормализации
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        private double[,] Normalization(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];
            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                //среднее значение сигнала нейрона
                var sum = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, column];
                }
                var average = sum/inputs.GetLength(0);

                //Стандарное квадратичное отклонение нейрона
                var error = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row, column] - average), 2);
                }
                var standartError = Math.Sqrt(error / inputs.GetLength(0));

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - average) / standartError;
                }
            }
            return result;
        }

        /// <summary>
        /// метод обратного распространения ошибки
        /// </summary>
        /// <param name="expected">реальный результат</param>
        /// <param name="inputs"> входные значения</param>
        /// <returns></returns>
        private double Backpropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;

            var difference = actual - expected;

            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j+1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];

                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];

                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }

            }
            var result = difference * difference;
            return result;
        }
        /// <summary>
        /// Проход по всем слоям кроме входного
        /// </summary>
        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousSignals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousSignals);
                }
            }
        }

        

        /// <summary>
        /// Отправить сигналы во входной слой
        /// </summary>
        /// <param name="inputSignals"></param>
        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            ///первоначальные данные - вх слой
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        /// <summary>
        /// Создание Входного слоя
        /// </summary>
        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }
        /// <summary>
        /// Создание скрытых слоев
        /// </summary>
        private void CreateHiddenLayers()
        {
            for (int j = 0; j < Topology.HiddenLayersCount.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayersCount[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }
                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }
        /// <summary>
        /// Создание Выходного слоя
        /// </summary>
        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }
    }
}
