using Backprop;

namespace NeuralNetPredictor.Network;

/// <summary>
/// Thin wrapper around the course classes (INeuron / HNeuron / ONeuron / NeuralNet in the
/// Backprop namespace). It adds ONLY plumbing around them: the epoch loop with shuffling,
/// loss / accuracy helpers and a cancel hook. All learning (forward pass, error, derivatives,
/// weight and bias updates) is done by Backprop.NeuralNet.learn().
///
/// Output layer = one sigmoid neuron per class, trained on 1-of-N targets.
/// The predicted class is the output neuron with the highest activation.
/// </summary>
public sealed class BackpropClassifier
{
    private readonly NeuralNet _net;
    private readonly int _numInput;
    private readonly int _numOutput;
    private readonly Random _shuffleRng;

    public BackpropClassifier(int numInput, int numHidden, int numOutput, double learningRate, int shuffleSeed = 1)
    {
        _numInput = numInput;
        _numOutput = numOutput;
        _net = new NeuralNet(numInput, numHidden, numOutput);
        _net.setLearningRates(learningRate, learningRate);
        _shuffleRng = new Random(shuffleSeed);
    }

    /// <summary>Forward pass for one sample; returns a copy of the output activations (each 0..1).</summary>
    public double[] ComputeOutputs(double[] features)
    {
        for (int i = 0; i < _numInput; ++i) _net.setInputs(i, features[i]);
        _net.run();
        var result = new double[_numOutput];
        for (int k = 0; k < _numOutput; ++k) result[k] = _net.getOuputData(k);
        return result;
    }

    /// <summary>
    /// Online back-propagation. Rows are [features..., one-hot targets...].
    /// logInterval: report every N epochs (0 = about 10 reports); epoch 1 and the last epoch are always reported.
    /// onEpochLogged(epoch, trainLoss) runs after the epoch; shouldStop is polled after every epoch.
    /// </summary>
    public void Train(double[][] data, int maxEpochs, int logInterval = 0,
        Action<int, double> onEpochLogged = null, Func<bool> shouldStop = null)
    {
        if (logInterval <= 0) logInterval = Math.Max(1, maxEpochs / 10);
        int[] order = Enumerable.Range(0, data.Length).ToArray();

        for (int epoch = 1; epoch <= maxEpochs; ++epoch)
        {
            for (int i = 0; i < order.Length - 1; ++i)          // Fisher-Yates shuffle
            {
                int j = _shuffleRng.Next(i, order.Length);
                (order[i], order[j]) = (order[j], order[i]);
            }

            foreach (int idx in order)
            {
                double[] row = data[idx];
                for (int i = 0; i < _numInput; ++i) _net.setInputs(i, row[i]);
                for (int k = 0; k < _numOutput; ++k) _net.setDesiredOutput(k, row[_numInput + k]);
                _net.learn();                                     // the course back-propagation step
            }

            if (onEpochLogged != null && (epoch == 1 || epoch % logInterval == 0 || epoch == maxEpochs))
                onEpochLogged(epoch, Loss(data));

            if (shouldStop != null && shouldStop()) break;
        }
    }

    /// <summary>Mean loss per sample: 0.5 * sum_k (target_k - output_k)^2 (the E of the worksheet / slides).</summary>
    public double Loss(double[][] data)
    {
        double total = 0;
        foreach (double[] row in data)
        {
            double[] y = ComputeOutputs(row);
            for (int k = 0; k < _numOutput; ++k)
            {
                double e = row[_numInput + k] - y[k];
                total += 0.5 * e * e;
            }
        }
        return data.Length == 0 ? 0 : total / data.Length;
    }

    /// <summary>Fraction of rows whose highest-activation output matches the one-hot target.</summary>
    public double Accuracy(double[][] data)
    {
        int correct = 0;
        var t = new double[_numOutput];
        foreach (double[] row in data)
        {
            Array.Copy(row, _numInput, t, 0, _numOutput);
            if (ArgMax(ComputeOutputs(row)) == ArgMax(t)) ++correct;
        }
        return data.Length == 0 ? 0 : (double)correct / data.Length;
    }

    public static int ArgMax(double[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; ++i) if (v[i] > v[best]) best = i;
        return best;
    }
}
