using System;
using System.Collections.Generic;
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
        public Neuron FeedForward(List<double> inputSignals)
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
        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            ///первоначальные данные - вх слой
            for (int i = 0; i < inputSignals.Count; i++)
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
                    var neuron = new Neuron(lastLayer.Count);
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
                var neuron = new Neuron(lastLayer.Count, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }
    }
}
