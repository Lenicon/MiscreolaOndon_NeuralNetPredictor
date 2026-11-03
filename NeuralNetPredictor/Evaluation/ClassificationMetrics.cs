using System.Text;

using NeuralNetPredictor.Network;

namespace NeuralNetPredictor.Evaluation;

/// <summary>Accuracy, confusion matrix and per-class precision / recall / F1 for one data split.</summary>
public sealed class ClassificationReport
{
    public int NumClasses { get; init; }
    public int Total { get; init; }
    /// <summary>Confusion[actual, predicted]</summary>
    public int[,] Confusion { get; init; }
    public double Accuracy { get; init; }
    /// <summary>Mean 0.5*squared error per sample (the network's training loss).</summary>
    public double Loss { get; init; }
    public double[] Precision { get; init; }
    public double[] Recall { get; init; }
    public double[] F1 { get; init; }
    public int[] Support { get; init; }
    public double MacroPrecision { get; init; }
    public double MacroRecall { get; init; }
    public double MacroF1 { get; init; }

    /// <summary>Accuracy of always predicting the most common class (the number to beat).</summary>
    public double MajorityBaseline => Total == 0 ? 0 : (double)Support.Max() / Total;
}

public static class ClassificationMetrics
{
    public static int ArgMax(double[] v) => BackpropClassifier.ArgMax(v);

    /// <summary>Runs the network over <paramref name="data"/> (rows = [features..., one-hot...]).</summary>
    public static ClassificationReport Evaluate(BackpropClassifier nn, double[][] data,
        int numInput, int numOutput)
    {
        var confusion = new int[numOutput, numOutput];
        var x = new double[numInput];
        var t = new double[numOutput];

        foreach (double[] row in data)
        {
            Array.Copy(row, x, numInput);
            Array.Copy(row, numInput, t, 0, numOutput);
            int predicted = ArgMax(nn.ComputeOutputs(x));
            int actual = ArgMax(t);
            confusion[actual, predicted]++;
        }

        var precision = new double[numOutput];
        var recall = new double[numOutput];
        var f1 = new double[numOutput];
        var support = new int[numOutput];
        int correct = 0;

        for (int c = 0; c < numOutput; ++c)
        {
            int tp = confusion[c, c];
            int rowSum = 0, colSum = 0;
            for (int k = 0; k < numOutput; ++k)
            {
                rowSum += confusion[c, k]; // actual = c
                colSum += confusion[k, c]; // predicted = c
            }
            support[c] = rowSum;
            correct += tp;
            precision[c] = colSum > 0 ? (double)tp / colSum : 0.0;
            recall[c] = rowSum > 0 ? (double)tp / rowSum : 0.0;
            f1[c] = (precision[c] + recall[c]) > 0
                ? 2.0 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0.0;
        }

        return new ClassificationReport
        {
            NumClasses = numOutput,
            Total = data.Length,
            Confusion = confusion,
            Accuracy = data.Length > 0 ? (double)correct / data.Length : 0.0,
            Loss = nn.Loss(data),
            Precision = precision,
            Recall = recall,
            F1 = f1,
            Support = support,
            MacroPrecision = precision.Average(),
            MacroRecall = recall.Average(),
            MacroF1 = f1.Average()
        };
    }

    /// <summary>Formats one report as monospace text (summary line, per-class table, confusion matrix).</summary>
    public static string Format(ClassificationReport r, string[] classNames, string title)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"=== {title} ({r.Total} samples) ===");
        sb.AppendLine($"Accuracy: {r.Accuracy:P2}    Loss (1/2 squared error): {r.Loss:F4}    " +
                      $"Majority-class baseline: {r.MajorityBaseline:P2}");
        sb.AppendLine();
        sb.AppendLine($"{"Class",-14}{"Precision",11}{"Recall",9}{"F1",9}{"Support",9}");
        for (int c = 0; c < r.NumClasses; ++c)
            sb.AppendLine($"{classNames[c],-14}{r.Precision[c],11:F3}{r.Recall[c],9:F3}{r.F1[c],9:F3}{r.Support[c],9}");
        sb.AppendLine($"{"Macro average",-14}{r.MacroPrecision,11:F3}{r.MacroRecall,9:F3}{r.MacroF1,9:F3}{r.Total,9}");
        sb.AppendLine();
        sb.AppendLine("Confusion matrix (rows = actual, columns = predicted):");
        sb.Append($"{"",-14}");
        for (int c = 0; c < r.NumClasses; ++c) sb.Append($"{Shorten(classNames[c]),13}");
        sb.AppendLine();
        for (int a = 0; a < r.NumClasses; ++a)
        {
            sb.Append($"{classNames[a],-14}");
            for (int p = 0; p < r.NumClasses; ++p) sb.Append($"{r.Confusion[a, p],13}");
            sb.AppendLine();
        }
        return sb.ToString();
    }

    private static string Shorten(string s) => s.Length <= 12 ? s : s.Substring(0, 12);
}
