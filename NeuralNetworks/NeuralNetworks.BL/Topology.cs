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
        /// кол-во нейронов в каждом скрытом слое
        /// </summary>
        public List<int> HiddenLayersCount { get; }


        public Topology(int inputCount, int outputCount, params int[] hiddenLayersCount)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            HiddenLayersCount = new List<int>();
            HiddenLayersCount.AddRange(hiddenLayersCount);

        }
    }
}
