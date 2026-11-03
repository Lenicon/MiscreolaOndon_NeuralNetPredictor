using System.Globalization;
using System.Text;

namespace NeuralNetPredictor.Pipeline;

/// <summary>Summary statistics of one input column (computed on the whole raw file, for display only).</summary>
public sealed class FeatureStat
{
    public string Name { get; init; }
    public double Min { get; init; }
    public double Max { get; init; }
    public double Mean { get; init; }
    public int MissingCount { get; init; }
}

/// <summary>
/// A fully pre-processed dataset. Every row in Train/Validation/Test is laid out exactly like
/// the rows the Backprop network wrapper expects: [scaled features ... , one-hot class ...].
/// </summary>
public sealed class Dataset
{
    public string SourcePath { get; init; }
    public string[] FeatureNames { get; init; }
    public string[] ClassNames { get; init; }

    public double[][] Train { get; init; }
    public double[][] Validation { get; init; }
    public double[][] Test { get; init; }

    /// <summary>Values used to fill blanks. Learned from the TRAINING split only (no leakage).</summary>
    public double[] ImputeValues { get; init; }
    /// <summary>MinMax scaler parameters. Learned from the TRAINING split only.</summary>
    public double[] FeatureMin { get; init; }
    public double[] FeatureMax { get; init; }

    public int RawRowCount { get; init; }
    public int SkippedRowCount { get; init; }
    public int MissingValueCount { get; init; }
    public int[] ClassCounts { get; init; }
    public FeatureStat[] Stats { get; init; }

    /// <summary>Header + first rows of the file as text, for the preview grid.</summary>
    public string[] Headers { get; init; }
    public List<string[]> PreviewRows { get; init; }

    public int NumFeatures => FeatureNames.Length;
    public int NumClasses => ClassNames.Length;

    /// <summary>Applies the SAME imputation + MinMax scaling used for training to one raw sample.
    /// Use NaN for a missing value.</summary>
    public double[] ScaleRaw(double[] raw)
    {
        var x = new double[NumFeatures];
        for (int i = 0; i < NumFeatures; ++i)
        {
            double v = double.IsNaN(raw[i]) ? ImputeValues[i] : raw[i];
            double range = FeatureMax[i] - FeatureMin[i];
            x[i] = range > 0.0 ? (v - FeatureMin[i]) / range : 0.0;
        }
        return x;
    }

    public string Describe()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"File            : {Path.GetFileName(SourcePath)}");
        sb.AppendLine($"Rows used       : {RawRowCount}   (rows skipped for missing label: {SkippedRowCount})");
        sb.AppendLine($"Missing cells   : {MissingValueCount}  -> filled with the TRAIN-set column mean");
        sb.AppendLine($"Split           : train {Train.Length} / validation {Validation.Length} / test {Test.Length}  (stratified, seed-fixed)");
        sb.AppendLine($"Scaling         : MinMax to [0,1], parameters fitted on the training split only");
        sb.AppendLine($"Network shape   : {NumFeatures} inputs -> hidden -> {NumClasses} outputs (one-hot, sigmoid)");
        sb.AppendLine();
        sb.AppendLine("Classes:");
        for (int c = 0; c < NumClasses; ++c)
            sb.AppendLine($"  {ClassNames[c],-14} {ClassCounts[c],5}  ({100.0 * ClassCounts[c] / RawRowCount:F1}%)");
        sb.AppendLine();
        sb.AppendLine($"{"Feature",-22}{"Min",9}{"Max",9}{"Mean",9}{"Missing",9}");
        foreach (var s in Stats)
            sb.AppendLine($"{s.Name,-22}{s.Min,9:F1}{s.Max,9:F1}{s.Mean,9:F1}{s.MissingCount,9}");
        return sb.ToString();
    }
}

/// <summary>Loads a CSV file and turns it into a <see cref="Dataset"/>.</summary>
public static class DataPipeline
{
    private static readonly string[] MissingTokens = { "", "na", "n/a", "nan", "null", "?", "none" };

    /// <summary>
    /// CSV format: first line = header, last column = class label (text), all other columns numeric.
    /// Blank / NA / ? cells in feature columns are treated as missing values.
    /// </summary>
    public static Dataset Load(string path, double trainFraction = 0.70,
        double validationFraction = 0.15, int seed = 1)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("CSV file not found.", path);

        string[] lines = File.ReadAllLines(path)
            .Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
        if (lines.Length < 11)
            throw new InvalidDataException("The file needs a header and at least 10 data rows.");

        string[] header = SplitLine(lines[0]);
        int numCols = header.Length;
        if (numCols < 3)
            throw new InvalidDataException("Need at least 2 feature columns plus 1 label column.");
        int numFeatures = numCols - 1;

        // ---- 1. parse ---------------------------------------------------------------------
        var features = new List<double[]>();   // NaN marks a missing value
        var labels = new List<string>();
        var preview = new List<string[]>();
        int skipped = 0;

        for (int r = 1; r < lines.Length; ++r)
        {
            string[] cells = SplitLine(lines[r]);
            if (cells.Length != numCols)
                throw new InvalidDataException(
                    $"Data row {r} has {cells.Length} columns but the header has {numCols}.");

            string label = cells[numCols - 1];
            if (IsMissing(label)) { skipped++; continue; }

            var f = new double[numFeatures];
            for (int c = 0; c < numFeatures; ++c)
            {
                if (IsMissing(cells[c])) f[c] = double.NaN;
                else if (!double.TryParse(cells[c], NumberStyles.Float, CultureInfo.InvariantCulture, out f[c]))
                    throw new InvalidDataException(
                        $"Data row {r}, column '{header[c]}': '{cells[c]}' is not a number.");
            }
            features.Add(f);
            labels.Add(label);
            if (preview.Count < 200) preview.Add(cells);
        }

        int n = features.Count;
        if (n < 10) throw new InvalidDataException("Fewer than 10 usable rows.");

        string[] classNames = labels.Distinct().OrderBy(x => x, StringComparer.OrdinalIgnoreCase).ToArray();
        if (classNames.Length < 2) throw new InvalidDataException("Need at least 2 different classes.");
        if (classNames.Length > 20) throw new InvalidDataException("More than 20 classes: is the last column really a label?");
        int numClasses = classNames.Length;
        int[] labelIndex = labels.Select(l => Array.IndexOf(classNames, l)).ToArray();

        // ---- 2. STRATIFIED shuffle + split (Fisher-Yates inside every class) ------------------
        // Each class is split in the same 70/15/15 proportions, so train / validation / test all
        // keep the original class balance (important with only ~150 rows).
        var rnd = new Random(seed);
        void Shuffle(int[] a)
        {
            for (int i = 0; i < a.Length - 1; ++i)
            {
                int j = rnd.Next(i, a.Length);
                (a[i], a[j]) = (a[j], a[i]);
            }
        }
        var trainIdx = new List<int>(); var valIdx = new List<int>(); var testIdx = new List<int>();
        for (int c = 0; c < numClasses; ++c)
        {
            int[] idx = Enumerable.Range(0, n).Where(i => labelIndex[i] == c).ToArray();
            Shuffle(idx);
            int cTrain = (int)Math.Round(idx.Length * trainFraction);
            int cVal = (int)Math.Round(idx.Length * validationFraction);
            for (int i = 0; i < idx.Length; ++i)
            {
                if (i < cTrain) trainIdx.Add(idx[i]);
                else if (i < cTrain + cVal) valIdx.Add(idx[i]);
                else testIdx.Add(idx[i]);
            }
        }
        int[] trainArr = trainIdx.ToArray(), valArr = valIdx.ToArray(), testArr = testIdx.ToArray();
        Shuffle(trainArr); Shuffle(valArr); Shuffle(testArr);   // mix classes within each subset
        int[] order = trainArr.Concat(valArr).Concat(testArr).ToArray();
        int nTrain = trainArr.Length, nVal = valArr.Length, nTest = testArr.Length;
        if (nTrain < 5 || nVal < 1 || nTest < 1)
            throw new InvalidDataException("Split fractions leave an empty subset.");

        // ---- 3. learn imputation + scaling from the TRAIN rows only -----------------------------
        var impute = new double[numFeatures];
        var fMin = new double[numFeatures];
        var fMax = new double[numFeatures];
        for (int c = 0; c < numFeatures; ++c)
        {
            double sum = 0, mn = double.MaxValue, mx = double.MinValue;
            int cnt = 0;
            for (int t = 0; t < nTrain; ++t)
            {
                double v = features[order[t]][c];
                if (double.IsNaN(v)) continue;
                sum += v; cnt++;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            impute[c] = cnt > 0 ? sum / cnt : 0.0;
            fMin[c] = cnt > 0 ? mn : 0.0;
            fMax[c] = cnt > 0 ? mx : 1.0;
        }

        // ---- 4. descriptive statistics of the raw file (display only) ---------------------------
        int totalMissing = 0;
        var stats = new FeatureStat[numFeatures];
        for (int c = 0; c < numFeatures; ++c)
        {
            double sum = 0, mn = double.MaxValue, mx = double.MinValue;
            int cnt = 0, miss = 0;
            for (int r = 0; r < n; ++r)
            {
                double v = features[r][c];
                if (double.IsNaN(v)) { miss++; continue; }
                sum += v; cnt++;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            totalMissing += miss;
            stats[c] = new FeatureStat
            {
                Name = header[c],
                Min = cnt > 0 ? mn : 0,
                Max = cnt > 0 ? mx : 0,
                Mean = cnt > 0 ? sum / cnt : 0,
                MissingCount = miss
            };
        }

        var classCounts = new int[numClasses];
        foreach (int li in labelIndex) classCounts[li]++;

        // ---- 5. build the network-ready matrices ------------------------------------------------
        double[][] BuildRows(int start, int count)
        {
            var result = new double[count][];
            for (int k = 0; k < count; ++k)
            {
                int src = order[start + k];
                var row = new double[numFeatures + numClasses];
                for (int c = 0; c < numFeatures; ++c)
                {
                    double v = double.IsNaN(features[src][c]) ? impute[c] : features[src][c];
                    double range = fMax[c] - fMin[c];
                    row[c] = range > 0.0 ? (v - fMin[c]) / range : 0.0;
                }
                row[numFeatures + labelIndex[src]] = 1.0; // 1-of-N target
                result[k] = row;
            }
            return result;
        }

        return new Dataset
        {
            SourcePath = path,
            FeatureNames = header.Take(numFeatures).ToArray(),
            ClassNames = classNames,
            Train = BuildRows(0, nTrain),
            Validation = BuildRows(nTrain, nVal),
            Test = BuildRows(nTrain + nVal, nTest),
            ImputeValues = impute,
            FeatureMin = fMin,
            FeatureMax = fMax,
            RawRowCount = n,
            SkippedRowCount = skipped,
            MissingValueCount = totalMissing,
            ClassCounts = classCounts,
            Stats = stats,
            Headers = header,
            PreviewRows = preview
        };
    }

    private static bool IsMissing(string s) =>
        Array.IndexOf(MissingTokens, s.Trim().ToLowerInvariant()) >= 0;

    private static string[] SplitLine(string line) =>
        line.Split(',').Select(s => s.Trim().Trim('"')).ToArray();
}
