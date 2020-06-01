using System.Collections.Generic;

namespace NeuralNetworks.BL
{
    /// <summary>
    /// Слой
    /// </summary>
    public class Layer
    {
        /// <summary>
        /// нейроны в слое
        /// </summary>
        public List<Neuron> Neurons { get; }
        /// <summary>
        /// Число нейронов в слое
        /// </summary>
        public int Count => Neurons?.Count ?? 0;


        /// <summary>
        /// 
        /// </summary>
        /// <param name="neurons">Список нейронов в слое</param>
        /// <param name="neuronType">Тип нейронов в слое. в слое один тип</param>
        public Layer(List<Neuron> neurons, NeuronType neuronType = NeuronType.Normal)
        {
            //Todo: проверить все вх нейроны на соотв типу
            Neurons = neurons;
        }


        /// <summary>
        /// Вывод всех выходных данных со всех нейронов
        /// </summary>
        /// <returns></returns>
        public List<double> GetSignals()
        {
            var result = new List<double>();
            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }
    }
}
