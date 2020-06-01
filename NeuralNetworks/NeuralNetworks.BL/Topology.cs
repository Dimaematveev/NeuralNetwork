using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BL
{
    /// <summary>
    /// Топология нейроной сети
    /// </summary>
    public class Topology
    {
        

        /// <summary>
        /// Кол-во  нейронов в  вх слое
        /// </summary>
        public int InputCount { get; }
        /// <summary>
        /// кол-во нейронов в  вых слое
        /// </summary>
        public int OutputCount { get; }
        /// <summary>
        /// скорость обучения
        /// </summary>
        public double LearningRate { get; }

        /// <summary>
        /// кол-во нейронов в каждом скрытом слое
        /// </summary>
        public List<int> HiddenLayersCount { get; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputCount"></param>
        /// <param name="outputCount"></param>
        /// <param name="learningRate">скорость обучения</param>
        /// <param name="hiddenLayersCount"></param>
        public Topology(int inputCount, int outputCount, double learningRate, params int[] hiddenLayersCount)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRate = learningRate;
            HiddenLayersCount = new List<int>();
            HiddenLayersCount.AddRange(hiddenLayersCount);

        }
    }
}
