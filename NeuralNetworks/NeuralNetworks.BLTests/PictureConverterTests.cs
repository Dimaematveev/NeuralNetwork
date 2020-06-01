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
    public class PictureConverterTests
    {
        [TestMethod()]
        public void ConvertTest()
        {
            var converter = new PictureConverter();
            var inputs = converter.Convert(@"Images\Parasitized.png");
            converter.Save(@"Images\Parasitized_BW.png", inputs);

            inputs = converter.Convert(@"Images\Unparasitized.png");
            converter.Save(@"Images\Unparasitized_BW.png", inputs);
        }
    }
}