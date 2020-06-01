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
        /// Тип нейрона
        /// </summary>
        public NeuronType NeuronType { get; }
        /// <summary>
        /// Сохранение состояния. Результат после действий всех коэфф
        /// </summary>
        public double Output { get; private set; }


        /// <summary>
        /// Создать нейрон
        /// </summary>
        /// <param name="inputCount">кол-во входных связей</param>
        /// <param name="neuronType">Тип нейрона</param>
        public Neuron(int inputCount, NeuronType neuronType = NeuronType.Normal)
        {
            NeuronType = neuronType;
            Weights = new List<double>();

            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(1);
            }
        }
        /// <summary>
        /// Слева направо сеть. 
        /// </summary>
        /// <param name="inputs">Список вход. сигналов - на 1 нейрон приходят, кол-во сигналов==кол-во весов</param>
        /// <returns>Сохранение состояния</returns>
        public double FeedForward(List<double> inputs)
        {
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
        /// Установка весов
        /// </summary>
        /// <param name="weights"></param>
        public void SetWights(params double[] weights)
        {
            ///TODO: удалить после возможности обучения сети
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
