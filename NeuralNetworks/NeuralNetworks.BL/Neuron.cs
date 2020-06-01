using System;
using System.Collections.Generic;

namespace NeuralNetworks.BL
{
    /// <summary>
    /// Нейрон
    /// </summary>
    public class Neuron
    {
      

        /// <summary>
        /// Веса 
        /// </summary>
        public List<double> Weights { get; }
        /// <summary>
        /// Входные значения
        /// </summary>
        public List<double> Inputs { get; }
        /// <summary>
        /// Тип нейрона
        /// </summary>
        public NeuronType NeuronType { get; }
        /// <summary>
        /// Сохранение состояния. Результат после действий всех коэфф
        /// </summary>
        public double Output { get; private set; }

        /// <summary>
        /// Дельта
        /// </summary>
        public double Delta { get; private set; }


        /// <summary>
        /// Создать нейрон
        /// </summary>
        /// <param name="inputCount">кол-во входных связей</param>
        /// <param name="neuronType">Тип нейрона</param>
        public Neuron(int inputCount, NeuronType neuronType = NeuronType.Normal)
        {
            NeuronType = neuronType;
            Weights = new List<double>();
            Inputs = new List<double>();
            InitWeightsRandomValue(inputCount);

        }


        /// <summary>
        /// заполнить случайными числами вес
        /// </summary>
        /// <param name="inputCount"></param>
        private void InitWeightsRandomValue(int inputCount)
        {
            Random rnd = new Random();
            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rnd.NextDouble());
                }
                Inputs.Add(0);
            }

        }
        /// <summary>
        /// Слева направо сеть.  и сохранение значений
        /// </summary>
        /// <param name="inputs">Список вход. сигналов - на 1 нейрон приходят, кол-во сигналов==кол-во весов</param>
        /// <returns>Сохранение состояния</returns>
        public double FeedForward(List<double> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }
            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }
            //Сигмойда для вычисления вых. рез
            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }
            return Output;
        }

        /// <summary>
        /// сигмоида для вычисления вых. рез
        /// </summary>
        /// <param name="x">от какого значения берется</param>
        /// <returns></returns>
        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        /// <summary>
        /// сигмоида для вычисления вых. рез
        /// </summary>
        /// <param name="x">от какого значения берется</param>
        /// <returns></returns>
        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }

        /// <summary>
        /// изменение нейрона.
        /// </summary>
        /// <param name="error"> шибка нейрона </param>
        /// <param name="learningRate"> сила измениения - скорость обучения</param>
        public void Learn(double error, double learningRate)
        {
            if (NeuronType==NeuronType.Input)
            {
                return;
            }
            Delta = error * SigmoidDx(Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
