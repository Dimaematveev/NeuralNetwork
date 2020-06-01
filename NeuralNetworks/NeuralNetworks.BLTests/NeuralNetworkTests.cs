using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks.BL;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.BL.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            // Результат - Пациент болен - 1
            //             Пациент Здоров - 0
            var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            var inputs = new double[,]
            {
                // Неправильная температура T
                // Хороший возраст A
                // Курит S
                // Правильно питается F
                // T  A  S  F
                 { 0, 0, 0, 0 },
                 { 0, 0, 0, 1 },
                 { 0, 0, 1, 0 },
                 { 0, 0, 1, 1 },
                 { 0, 1, 0, 0 },
                 { 0, 1, 0, 1 },
                 { 0, 1, 1, 0 },
                 { 0, 1, 1, 1 },
                 { 1, 0, 0, 0 },
                 { 1, 0, 0, 1 },
                 { 1, 0, 1, 0 },
                 { 1, 0, 1, 1 },
                 { 1, 1, 0, 0 },
                 { 1, 1, 0, 1 },
                 { 1, 1, 1, 0 },
                 { 1, 1, 1, 1 },
            };


            var topology = new Topology(4, 1, 0.1, 3);
            var neuralNetwork = new NeuralNetwork(topology);
            var difference = neuralNetwork.Learn(outputs, inputs, 100000);
            
            var results = new List<double>();
            for (int i = 0; i < outputs.Length; i++)
            {
                var row = NeuralNetwork.GetRow(inputs, i);
                var res = neuralNetwork.FeedForward(row).Output;
                results.Add(res);
            }
            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 3);
                var actual = Math.Round(results[i], 3);
            }
            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 3);
                var actual = Math.Round(results[i], 3);


                Assert.AreEqual(expected, actual);
            }


        }
    }
}