using System.Diagnostics;
using System.Globalization;
using System.Text;
using NeuralNetPredictor.Network;
using NeuralNetPredictor.Pipeline;

namespace NeuralNetPredictor.Tuning;

/// <summary>One finished experiment: a (learning rate, hidden nodes, epochs) combination and its scores.</summary>
public sealed class TuningResult
{
    public double LearningRate { get; init; }
    public int HiddenNodes { get; init; }
    public int Epochs { get; init; }
    public double TrainAccuracy { get; init; }
    public double ValAccuracy { get; init; }
    public double TrainLoss { get; init; }
    public double ValLoss { get; init; }
    /// <summary>Accuracy gap train - validation. A large positive gap means over-fitting.</summary>
    public double Gap => TrainAccuracy - ValAccuracy;
    /// <summary>Average seconds of training until this epoch count was reached.</summary>
    public double Seconds { get; init; }
}

public sealed class TuningProgress
{
    public int RunsDone { get; init; }
    public int RunsTotal { get; init; }
    public string Message { get; init; }
}

/// <summary>
/// Grid search over learning rate x hidden nodes x epochs. Models are ranked on the VALIDATION
/// split so the test split stays untouched for the final, honest evaluation.
/// Trick: for each (learning rate, hidden) pair the network is trained once up to the largest
/// epoch value and scored when it passes each smaller epoch value (same result as separate runs, much faster).
/// The course classes start from random weights that cannot be seeded, so every pair is trained
/// <c>repeats</c> times and the scores are averaged - a single lucky or unlucky start does not decide the winner.
/// </summary>
public static class HyperparameterTuner
{
    public static List<TuningResult> RunGrid(
        Dataset ds,
        IReadOnlyList<double> learningRates,
        IReadOnlyList<int> hiddenCounts,
        IReadOnlyList<int> epochCheckpoints,
        int repeats,
        int seed,
        IProgress<TuningProgress> progress,
        Func<bool> cancel)
    {
        var results = new List<TuningResult>();
        int[] checkpoints = epochCheckpoints.Distinct().OrderBy(e => e).ToArray();
        int maxEpochs = checkpoints[checkpoints.Length - 1];
        int interval = checkpoints.Aggregate(Gcd);
        repeats = Math.Max(1, repeats);

        int total = learningRates.Count * hiddenCounts.Count;
        int done = 0;

        foreach (double lr in learningRates)
        {
            foreach (int hidden in hiddenCounts)
            {
                if (cancel != null && cancel()) return results;

                // sums over the repeats: [train acc, val acc, train loss, val loss, seconds]
                var sums = checkpoints.ToDictionary(c => c, _ => new double[5]);

                for (int rep = 0; rep < repeats; ++rep)
                {
                    var stopwatch = Stopwatch.StartNew();
                    var net = new BackpropClassifier(ds.NumFeatures, hidden, ds.NumClasses, lr, seed + rep);
                    net.Train(ds.Train, maxEpochs, interval,
                        (epoch, trainLoss) =>
                        {
                            if (!sums.TryGetValue(epoch, out double[] s)) return;
                            s[0] += net.Accuracy(ds.Train);
                            s[1] += net.Accuracy(ds.Validation);
                            s[2] += trainLoss;
                            s[3] += net.Loss(ds.Validation);
                            s[4] += stopwatch.Elapsed.TotalSeconds;
                        },
                        cancel);
                    if (cancel != null && cancel()) return results;
                }

                foreach (int epoch in checkpoints)
                {
                    double[] s = sums[epoch];
                    results.Add(new TuningResult
                    {
                        LearningRate = lr,
                        HiddenNodes = hidden,
                        Epochs = epoch,
                        TrainAccuracy = s[0] / repeats,
                        ValAccuracy = s[1] / repeats,
                        TrainLoss = s[2] / repeats,
                        ValLoss = s[3] / repeats,
                        Seconds = s[4] / repeats
                    });
                }

                done++;
                progress?.Report(new TuningProgress
                {
                    RunsDone = done,
                    RunsTotal = total,
                    Message = $"Finished lr={lr.ToString("0.####", CultureInfo.InvariantCulture)}, hidden={hidden}  ({done}/{total}, {repeats} runs each)"
                });
            }
        }
        return results;
    }

    /// <summary>Best = highest validation accuracy; ties are broken by lower validation loss.</summary>
    public static TuningResult PickBest(IEnumerable<TuningResult> results) =>
        results.Where(r => !double.IsNaN(r.ValLoss))
               .OrderByDescending(r => r.ValAccuracy)
               .ThenBy(r => r.ValLoss)
               .FirstOrDefault();

    public static void ExportCsv(IEnumerable<TuningResult> results, string path)
    {
        var sb = new StringBuilder();
        sb.AppendLine("learning_rate,hidden_nodes,epochs,train_accuracy,val_accuracy,train_loss,val_loss,gap,seconds");
        foreach (var r in results)
        {
            sb.AppendLine(string.Join(",", new[]
            {
                r.LearningRate.ToString("0.####", CultureInfo.InvariantCulture),
                r.HiddenNodes.ToString(CultureInfo.InvariantCulture),
                r.Epochs.ToString(CultureInfo.InvariantCulture),
                r.TrainAccuracy.ToString("F4", CultureInfo.InvariantCulture),
                r.ValAccuracy.ToString("F4", CultureInfo.InvariantCulture),
                r.TrainLoss.ToString("F4", CultureInfo.InvariantCulture),
                r.ValLoss.ToString("F4", CultureInfo.InvariantCulture),
                r.Gap.ToString("F4", CultureInfo.InvariantCulture),
                r.Seconds.ToString("F2", CultureInfo.InvariantCulture)
            }));
        }
        File.WriteAllText(path, sb.ToString());
    }

    private static int Gcd(int a, int b) => b == 0 ? a : Gcd(b, a % b);
}
