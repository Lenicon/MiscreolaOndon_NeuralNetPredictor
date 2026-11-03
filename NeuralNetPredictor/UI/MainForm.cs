using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using NeuralNetPredictor.Evaluation;
using NeuralNetPredictor.Network;
using NeuralNetPredictor.Pipeline;
using NeuralNetPredictor.Tuning;

namespace NeuralNetPredictor.UI;

/// <summary>
/// Main window. Tabs: Data -> Train -> Tune -> Evaluate -> Predict.
/// All heavy work (training, grid search) runs on a background task; the UI is only touched
/// through IProgress callbacks, so the window never freezes and there are no cross-thread calls.
/// </summary>
public sealed class MainForm : Form
{
    private const int TuneRepeats = 3; // random starts averaged per grid cell (course weights cannot be seeded)
    private const int Seed = 1; // fixes data split and weight initialisation -> reproducible runs
    private readonly string _outputDir = Path.Combine(AppContext.BaseDirectory, "Output");

    // ---- state ----
    private Dataset _ds;
    private BackpropClassifier _nn;             // last successfully trained network (wraps Backprop.NeuralNet)
    private CancellationTokenSource _cts;
    private bool _busy;
    private List<TuningResult> _tuningResults = new();
    private TuningResult _bestTuning;
    private readonly List<string> _trainingLogRows = new();
    private int _lastHidden, _lastEpochs;
    private double _lastLr;

    // ---- shared controls ----
    private readonly TabControl _tabs = new() { Dock = DockStyle.Fill };
    private readonly StatusStrip _status = new();
    private readonly ToolStripStatusLabel _statusLabel = new() { Text = "Ready." };

    // data tab
    private readonly Label _lblPath = new() { AutoSize = true, Padding = new Padding(8, 8, 0, 0) };
    private readonly DataGridView _dgvRaw = MakeGrid();
    private readonly TextBox _txtDataInfo = MakeMonoBox();

    // train tab
    private readonly NumericUpDown _nudHidden = MakeNud(1, 256, 0, 1, 8);
    private readonly NumericUpDown _nudEpochs = MakeNud(10, 50000, 0, 10, 300);
    private readonly NumericUpDown _nudLr = MakeNud(0.0001m, 1m, 4, 0.01m, 0.2m);
    private readonly NumericUpDown _nudLogEvery = MakeNud(1, 5000, 0, 5, 10);
    private readonly Button _btnTrain = new() { Text = "Train", Height = 34, Dock = DockStyle.Fill };
    private readonly Button _btnStop = new() { Text = "Stop", Height = 34, Dock = DockStyle.Fill, Enabled = false };
    private readonly Button _btnOpenOutput = new() { Text = "Open output folder", Height = 30, Dock = DockStyle.Fill };
    private readonly ProgressBar _progress = new() { Dock = DockStyle.Fill, Minimum = 0, Maximum = 100 };
    private readonly LossChartControl _chart = new() { Dock = DockStyle.Fill };
    private readonly TextBox _txtLog = MakeMonoBox();

    // tune tab
    private readonly TextBox _txtLrList = new() { Text = "0.05, 0.1, 0.2, 0.3, 0.5", Width = 200 };
    private readonly TextBox _txtHiddenList = new() { Text = "4, 8, 12, 16", Width = 200 };
    private readonly TextBox _txtEpochList = new() { Text = "50, 100, 200, 300, 500", Width = 200 };
    private readonly Button _btnTune = new() { Text = "Run grid search", AutoSize = true };
    private readonly Button _btnTuneCancel = new() { Text = "Cancel", AutoSize = true, Enabled = false };
    private readonly Button _btnApplyBest = new() { Text = "Use best settings in Train tab", AutoSize = true, Enabled = false };
    private readonly Button _btnExportTune = new() { Text = "Export CSV...", AutoSize = true, Enabled = false };
    private readonly ProgressBar _tuneProgress = new() { Dock = DockStyle.Fill, Minimum = 0, Maximum = 100 };
    private readonly Label _lblBest = new() { AutoSize = true, Padding = new Padding(4, 6, 0, 0), Text = "No grid search run yet." };
    private readonly DataGridView _dgvTune = MakeGrid();

    // evaluate tab
    private readonly DataGridView _dgvConfusion = MakeGrid();
    private readonly TextBox _txtReport = MakeMonoBox();

    // predict tab
    private readonly Panel _pnlPredictInputs = new() { Dock = DockStyle.Fill, AutoScroll = true };
    private readonly Button _btnPredict = new() { Text = "Predict", Height = 34, Dock = DockStyle.Fill };
    private readonly TextBox _txtPrediction = MakeMonoBox();
    private readonly List<TextBox> _predictBoxes = new();

    public MainForm()
    {
        Text = "MLP Iris Species Classifier  -  custom C# neural network";
        Size = new Size(1200, 800);
        MinimumSize = new Size(1000, 680);
        StartPosition = FormStartPosition.CenterScreen;

        _tabs.TabPages.Add(BuildDataTab());
        _tabs.TabPages.Add(BuildTrainTab());
        _tabs.TabPages.Add(BuildTuneTab());
        _tabs.TabPages.Add(BuildEvaluateTab());
        _tabs.TabPages.Add(BuildPredictTab());

        _status.Items.Add(_statusLabel);
        Controls.Add(_tabs);
        Controls.Add(_status);

        _btnTrain.Click += async (_, _) => await TrainAsync();
        _btnStop.Click += (_, _) => _cts?.Cancel();
        _btnOpenOutput.Click += (_, _) => OpenOutputFolder();
        _btnTune.Click += async (_, _) => await TuneAsync();
        _btnTuneCancel.Click += (_, _) => _cts?.Cancel();
        _btnApplyBest.Click += (_, _) => ApplyBestSettings();
        _btnExportTune.Click += (_, _) => ExportTuning();
        _btnPredict.Click += (_, _) => Predict();

        Shown += (_, _) =>
        {
            string defaultCsv = Path.Combine(AppContext.BaseDirectory, "Data", "iris.csv");
            if (File.Exists(defaultCsv)) LoadDataset(defaultCsv);
            else SetStatus("iris.csv not found next to the program - use 'Load CSV...'.");
        };
    }

    // =====================================================================================
    //  Tab construction
    // =====================================================================================

    private TabPage BuildDataTab()
    {
        var page = new TabPage("1. Data");
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3 };
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 55));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 45));

        var bar = new FlowLayoutPanel { Dock = DockStyle.Fill, AutoSize = true, Padding = new Padding(4) };
        var btnLoad = new Button { Text = "Load CSV...", AutoSize = true };
        btnLoad.Click += (_, _) =>
        {
            using var dlg = new OpenFileDialog { Filter = "CSV files (*.csv)|*.csv|All files|*.*", Title = "Choose a dataset" };
            if (dlg.ShowDialog(this) == DialogResult.OK) LoadDataset(dlg.FileName);
        };
        var hint = new Label
        {
            AutoSize = true,
            Padding = new Padding(8, 8, 0, 0),
            Text = "Format: header row, numeric feature columns, LAST column = class label."
        };
        bar.Controls.Add(btnLoad);
        bar.Controls.Add(hint);
        bar.Controls.Add(_lblPath);

        layout.Controls.Add(bar, 0, 0);
        layout.Controls.Add(_dgvRaw, 0, 1);
        layout.Controls.Add(_txtDataInfo, 0, 2);
        page.Controls.Add(layout);
        return page;
    }

    private TabPage BuildTrainTab()
    {
        var page = new TabPage("2. Train");
        var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1 };
        root.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 290));
        root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

        // left: hyper-parameters
        var left = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, Padding = new Padding(8) };
        left.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 55));
        left.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 45));
        AddRow(left, "Hidden nodes", _nudHidden);
        AddRow(left, "Epochs", _nudEpochs);
        AddRow(left, "Learning rate", _nudLr);
        AddRow(left, "Log loss every N epochs", _nudLogEvery);

        var buttons = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, Height = 40 };
        buttons.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        buttons.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        buttons.Controls.Add(_btnTrain, 0, 0);
        buttons.Controls.Add(_btnStop, 1, 0);
        left.Controls.Add(buttons, 0, left.RowCount++);
        left.SetColumnSpan(buttons, 2);
        _progress.Height = 20;
        left.Controls.Add(_progress, 0, left.RowCount++);
        left.SetColumnSpan(_progress, 2);
        left.Controls.Add(_btnOpenOutput, 0, left.RowCount++);
        left.SetColumnSpan(_btnOpenOutput, 2);
        var note = new Label
        {
            Dock = DockStyle.Fill,
            AutoSize = true,
            MaximumSize = new Size(260, 0),
            ForeColor = Color.DimGray,
            Text = "Every run is also saved as a training-log CSV (epoch, losses, accuracies) in the Output folder."
        };
        left.Controls.Add(note, 0, left.RowCount++);
        left.SetColumnSpan(note, 2);

        // right: chart on top, text log below
        var right = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2 };
        right.RowStyles.Add(new RowStyle(SizeType.Percent, 58));
        right.RowStyles.Add(new RowStyle(SizeType.Percent, 42));
        right.Controls.Add(_chart, 0, 0);
        right.Controls.Add(_txtLog, 0, 1);

        root.Controls.Add(left, 0, 0);
        root.Controls.Add(right, 1, 0);
        page.Controls.Add(root);
        return page;
    }

    private TabPage BuildTuneTab()
    {
        var page = new TabPage("3. Tune");
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4 };
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 26));
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        var inputs = new FlowLayoutPanel { Dock = DockStyle.Fill, AutoSize = true, Padding = new Padding(4) };
        inputs.Controls.Add(new Label { Text = "Learning rates:", AutoSize = true, Padding = new Padding(0, 6, 0, 0) });
        inputs.Controls.Add(_txtLrList);
        inputs.Controls.Add(new Label { Text = "Hidden nodes:", AutoSize = true, Padding = new Padding(10, 6, 0, 0) });
        inputs.Controls.Add(_txtHiddenList);
        inputs.Controls.Add(new Label { Text = "Epochs:", AutoSize = true, Padding = new Padding(10, 6, 0, 0) });
        inputs.Controls.Add(_txtEpochList);

        var buttons = new FlowLayoutPanel { Dock = DockStyle.Fill, AutoSize = true, Padding = new Padding(4) };
        buttons.Controls.Add(_btnTune);
        buttons.Controls.Add(_btnTuneCancel);
        buttons.Controls.Add(_btnApplyBest);
        buttons.Controls.Add(_btnExportTune);
        buttons.Controls.Add(_lblBest);

        layout.Controls.Add(inputs, 0, 0);
        layout.Controls.Add(_tuneProgress, 0, 1);
        layout.Controls.Add(buttons, 0, 2);
        layout.Controls.Add(_dgvTune, 0, 3);
        page.Controls.Add(layout);
        return page;
    }

    private TabPage BuildEvaluateTab()
    {
        var page = new TabPage("4. Evaluate");
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3 };
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 24));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 30));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 70));
        layout.Controls.Add(new Label
        {
            Text = "Test-set confusion matrix (rows = actual, columns = predicted; darker = larger share of that actual class)",
            Dock = DockStyle.Fill,
            Padding = new Padding(4, 4, 0, 0)
        }, 0, 0);
        layout.Controls.Add(_dgvConfusion, 0, 1);
        layout.Controls.Add(_txtReport, 0, 2);
        page.Controls.Add(layout);
        return page;
    }

    private TabPage BuildPredictTab()
    {
        var page = new TabPage("5. Predict");
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1 };
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 420));
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

        var left = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3, Padding = new Padding(8) };
        left.RowStyles.Add(new RowStyle(SizeType.Absolute, 24));
        left.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        left.RowStyles.Add(new RowStyle(SizeType.Absolute, 44));
        left.Controls.Add(new Label { Text = "Enter raw values (leave blank = missing):", Dock = DockStyle.Fill }, 0, 0);
        left.Controls.Add(_pnlPredictInputs, 0, 1);
        left.Controls.Add(_btnPredict, 0, 2);

        layout.Controls.Add(left, 0, 0);
        layout.Controls.Add(_txtPrediction, 1, 0);
        page.Controls.Add(layout);
        return page;
    }

    // =====================================================================================
    //  Data
    // =====================================================================================

    private void LoadDataset(string path)
    {
        try
        {
            _ds = DataPipeline.Load(path, 0.70, 0.15, Seed);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Could not load dataset", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        _nn = null;
        _lblPath.Text = path;
        _txtDataInfo.Text = _ds.Describe();

        var table = new DataTable();
        foreach (string h in _ds.Headers) table.Columns.Add(h);
        foreach (string[] row in _ds.PreviewRows) table.Rows.Add((object[])row);
        _dgvRaw.DataSource = table;

        BuildPredictInputs();
        _chart.Clear();
        _txtLog.Clear();
        _txtReport.Text = "Train a model first (tab 2).";
        _dgvConfusion.Columns.Clear();
        _txtPrediction.Text = "Train a model first, then enter values on the left and press Predict.";
        SetStatus($"Loaded {_ds.RawRowCount} rows, {_ds.NumFeatures} features, {_ds.NumClasses} classes.");
    }

    // =====================================================================================
    //  Training
    // =====================================================================================

    private async Task TrainAsync()
    {
        if (_ds == null) { MessageBox.Show(this, "Load a dataset first (tab 1).", "No data"); return; }
        if (_busy) return;

        // read settings on the UI thread
        int hidden = (int)_nudHidden.Value;
        int epochs = (int)_nudEpochs.Value;
        double lr = (double)_nudLr.Value;
        int logEvery = (int)_nudLogEvery.Value;
        Dataset ds = _ds;

        SetBusy(true);
        _cts = new CancellationTokenSource();
        CancellationToken token = _cts.Token;
        _chart.Clear();
        _txtLog.Clear();
        _progress.Value = 0;
        _trainingLogRows.Clear();
        _trainingLogRows.Add("epoch,train_loss,val_loss,train_accuracy,val_accuracy");

        AppendLog($"Training {ds.NumFeatures}-{hidden}-{ds.NumClasses} network | epochs={epochs} lr={lr} (sigmoid neurons, online back-propagation)");
        AppendLog($"{"epoch",6} {"train loss",11} {"val loss",10} {"train acc",10} {"val acc",9}");

        var progress = new Progress<EpochLog>(entry =>
        {
            _chart.AddPoint(entry.Epoch, entry.TrainLoss, entry.ValLoss);
            AppendLog($"{entry.Epoch,6} {entry.TrainLoss,11:F4} {entry.ValLoss,10:F4} {entry.TrainAcc,10:P1} {entry.ValAcc,9:P1}");
            _trainingLogRows.Add(string.Join(",", new[]
            {
                entry.Epoch.ToString(CultureInfo.InvariantCulture),
                F(entry.TrainLoss), F(entry.ValLoss),
                F(entry.TrainAcc), F(entry.ValAcc)
            }));
            _progress.Value = Math.Min(100, (int)(100.0 * entry.Epoch / epochs));
            SetStatus($"Training... epoch {entry.Epoch}/{epochs}");
        });

        BackpropClassifier net = null;
        var sw = Stopwatch.StartNew();
        try
        {
            net = await Task.Run(() =>
            {
                var n = new BackpropClassifier(ds.NumFeatures, hidden, ds.NumClasses, lr, Seed);
                n.Train(ds.Train, epochs, logEvery,
                    (epoch, trainLoss) =>
                    {
                        // this runs on the worker thread; only IProgress.Report touches the UI
                        ((IProgress<EpochLog>)progress).Report(new EpochLog
                        {
                            Epoch = epoch,
                            TrainLoss = trainLoss,
                            ValLoss = n.Loss(ds.Validation),
                            TrainAcc = n.Accuracy(ds.Train),
                            ValAcc = n.Accuracy(ds.Validation)
                        });
                    },
                    () => token.IsCancellationRequested);
                return n;
            });
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Training failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        sw.Stop();

        if (net != null)
        {
            _nn = net;
            _lastHidden = hidden; _lastEpochs = epochs; _lastLr = lr;

            string logPath = SaveTrainingLog();
            AppendLog(token.IsCancellationRequested
                ? $"Stopped early after {sw.Elapsed.TotalSeconds:F1}s."
                : $"Done in {sw.Elapsed.TotalSeconds:F1}s.");
            AppendLog($"Training log saved: {logPath}");
            UpdateEvaluation();
            SetStatus("Training finished. See the Evaluate tab.");
        }
        else
        {
            SetStatus("Training did not finish.");
        }

        _progress.Value = 100;
        SetBusy(false);
    }

    private string SaveTrainingLog()
    {
        Directory.CreateDirectory(_outputDir);
        string path = Path.Combine(_outputDir, $"training_log_{DateTime.Now:yyyyMMdd_HHmmss}.csv");
        File.WriteAllLines(path, _trainingLogRows);
        return path;
    }

    // =====================================================================================
    //  Hyper-parameter tuning
    // =====================================================================================

    private async Task TuneAsync()
    {
        if (_ds == null) { MessageBox.Show(this, "Load a dataset first (tab 1).", "No data"); return; }
        if (_busy) return;

        double[] lrs; int[] hiddens; int[] epochList;
        try
        {
            lrs = ParseDoubles(_txtLrList.Text);
            hiddens = ParseInts(_txtHiddenList.Text);
            epochList = ParseInts(_txtEpochList.Text);
            if (lrs.Length == 0 || hiddens.Length == 0 || epochList.Length == 0)
                throw new FormatException("Each list needs at least one value.");
            if (lrs.Any(v => v <= 0) || hiddens.Any(v => v < 1) || epochList.Any(v => v < 1))
                throw new FormatException("All values must be positive.");
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, "Please enter comma-separated numbers.\n\n" + ex.Message, "Invalid grid");
            return;
        }

        Dataset ds = _ds;

        SetBusy(true);
        _cts = new CancellationTokenSource();
        CancellationToken token = _cts.Token;
        _tuneProgress.Value = 0;
        _btnApplyBest.Enabled = false;
        _btnExportTune.Enabled = false;
        _lblBest.Text = "Running grid search...";

        var progress = new Progress<TuningProgress>(p =>
        {
            _tuneProgress.Value = Math.Min(100, (int)(100.0 * p.RunsDone / p.RunsTotal));
            SetStatus(p.Message);
        });

        List<TuningResult> results = null;
        try
        {
            results = await Task.Run(() => HyperparameterTuner.RunGrid(
                ds, lrs, hiddens, epochList, TuneRepeats, Seed, progress, () => token.IsCancellationRequested));
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Grid search failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        if (results != null && results.Count > 0)
        {
            _tuningResults = results;
            _bestTuning = HyperparameterTuner.PickBest(results);
            ShowTuningResults();

            Directory.CreateDirectory(_outputDir);
            string path = Path.Combine(_outputDir, $"tuning_results_{DateTime.Now:yyyyMMdd_HHmmss}.csv");
            HyperparameterTuner.ExportCsv(results, path);
            SetStatus($"Grid search done: {results.Count} experiments. Saved {Path.GetFileName(path)}");
        }
        else
        {
            SetStatus("Grid search produced no results.");
            _lblBest.Text = "No results.";
        }

        _tuneProgress.Value = 100;
        SetBusy(false);
    }

    private void ShowTuningResults()
    {
        // sort best-first so the table reads like a leaderboard
        var sorted = _tuningResults
            .OrderByDescending(r => r.ValAccuracy)
            .ThenBy(r => r.ValLoss)
            .ToList();
        _dgvTune.DataSource = new BindingList<TuningResult>(sorted);

        foreach (DataGridViewColumn col in _dgvTune.Columns)
        {
            if (col.ValueType == typeof(double)) col.DefaultCellStyle.Format = "F4";
        }
        if (_dgvTune.Columns["Seconds"] != null) _dgvTune.Columns["Seconds"].DefaultCellStyle.Format = "F2";
        if (_dgvTune.Columns["LearningRate"] != null) _dgvTune.Columns["LearningRate"].DefaultCellStyle.Format = "0.####";
        if (_dgvTune.Rows.Count > 0)
        {
            _dgvTune.Rows[0].DefaultCellStyle.BackColor = Color.Honeydew;
            _dgvTune.Rows[0].DefaultCellStyle.Font = new Font(_dgvTune.Font, FontStyle.Bold);
        }

        if (_bestTuning != null)
        {
            _lblBest.Text = $"Best (by validation accuracy): lr={_bestTuning.LearningRate}, hidden={_bestTuning.HiddenNodes}, " +
                            $"epochs={_bestTuning.Epochs}  ->  val acc {_bestTuning.ValAccuracy:P2}, val loss {_bestTuning.ValLoss:F4}";
            _btnApplyBest.Enabled = true;
        }
        _btnExportTune.Enabled = true;
    }

    private void ApplyBestSettings()
    {
        if (_bestTuning == null) return;
        _nudLr.Value = Clamp(_nudLr, (decimal)_bestTuning.LearningRate);
        _nudHidden.Value = Clamp(_nudHidden, _bestTuning.HiddenNodes);
        _nudEpochs.Value = Clamp(_nudEpochs, _bestTuning.Epochs);
        _nudLogEvery.Value = Clamp(_nudLogEvery, Math.Max(1, _bestTuning.Epochs / 30));
        _tabs.SelectedIndex = 1;
        SetStatus("Best grid-search settings copied to the Train tab. Press Train.");
    }

    private void ExportTuning()
    {
        if (_tuningResults.Count == 0) return;
        using var dlg = new SaveFileDialog { Filter = "CSV file (*.csv)|*.csv", FileName = "tuning_results.csv" };
        if (dlg.ShowDialog(this) != DialogResult.OK) return;
        HyperparameterTuner.ExportCsv(_tuningResults.OrderByDescending(r => r.ValAccuracy).ThenBy(r => r.ValLoss), dlg.FileName);
        SetStatus("Exported " + dlg.FileName);
    }

    // =====================================================================================
    //  Evaluation
    // =====================================================================================

    private void UpdateEvaluation()
    {
        if (_nn == null || _ds == null) return;
        int f = _ds.NumFeatures, c = _ds.NumClasses;

        ClassificationReport train = ClassificationMetrics.Evaluate(_nn, _ds.Train, f, c);
        ClassificationReport val = ClassificationMetrics.Evaluate(_nn, _ds.Validation, f, c);
        ClassificationReport test = ClassificationMetrics.Evaluate(_nn, _ds.Test, f, c);

        var sb = new StringBuilder();
        sb.AppendLine($"Model: {f}-{_lastHidden}-{c} MLP | epochs {_lastEpochs} | learning rate {_lastLr}");
        sb.AppendLine();
        sb.AppendLine(ClassificationMetrics.Format(test, _ds.ClassNames, "TEST SET  (final, untouched during tuning)"));
        sb.AppendLine(ClassificationMetrics.Format(val, _ds.ClassNames, "VALIDATION SET  (used to choose hyper-parameters)"));
        sb.AppendLine(ClassificationMetrics.Format(train, _ds.ClassNames, "TRAINING SET"));
        sb.AppendLine("=== ANALYSIS ===");
        foreach (string line in BuildAnalysis(train, test)) sb.AppendLine("- " + line);
        _txtReport.Text = sb.ToString();

        FillConfusionGrid(test);
    }

    private IEnumerable<string> BuildAnalysis(ClassificationReport train, ClassificationReport test)
    {
        var lines = new List<string>();
        double lift = test.Accuracy - test.MajorityBaseline;
        lines.Add($"Test accuracy {test.Accuracy:P1} beats always guessing the biggest class ({test.MajorityBaseline:P1}) by {lift * 100:F1} points.");

        double gap = train.Accuracy - test.Accuracy;
        if (gap > 0.07) lines.Add($"Train-test accuracy gap is {gap * 100:F1} points: the network is over-fitting. Try fewer hidden nodes or fewer epochs.");
        else if (gap < -0.02) lines.Add($"Test accuracy is above training accuracy by {-gap * 100:F1} points; this is normal sampling noise on a small test set.");
        else lines.Add($"Train-test accuracy gap is only {gap * 100:F1} points, so the model generalises well.");

        int worst = 0;
        for (int i = 1; i < test.NumClasses; ++i) if (test.F1[i] < test.F1[worst]) worst = i;
        lines.Add($"Hardest class: '{_ds.ClassNames[worst]}' (F1 {test.F1[worst]:F3}, precision {test.Precision[worst]:F3}, recall {test.Recall[worst]:F3}).");

        int bestA = -1, bestP = -1, bestCount = 0;
        for (int a = 0; a < test.NumClasses; ++a)
            for (int p = 0; p < test.NumClasses; ++p)
                if (a != p && test.Confusion[a, p] > bestCount) { bestCount = test.Confusion[a, p]; bestA = a; bestP = p; }
        if (bestCount > 0)
            lines.Add($"Most common mistake: actual '{_ds.ClassNames[bestA]}' predicted as '{_ds.ClassNames[bestP]}' ({bestCount} times).");

        lines.Add($"Macro F1 {test.MacroF1:F3} treats all classes equally, so it is a fairer summary than accuracy if the classes are unbalanced.");
        return lines;
    }

    private void FillConfusionGrid(ClassificationReport r)
    {
        _dgvConfusion.Columns.Clear();
        _dgvConfusion.Columns.Add("actual", "Actual \\ Predicted");
        foreach (string name in _ds.ClassNames) _dgvConfusion.Columns.Add(name, name);
        _dgvConfusion.Rows.Clear();

        for (int a = 0; a < r.NumClasses; ++a)
        {
            var cells = new object[r.NumClasses + 1];
            cells[0] = _ds.ClassNames[a];
            for (int p = 0; p < r.NumClasses; ++p) cells[p + 1] = r.Confusion[a, p];
            int rowIndex = _dgvConfusion.Rows.Add(cells);

            for (int p = 0; p < r.NumClasses; ++p)
            {
                double share = r.Support[a] > 0 ? (double)r.Confusion[a, p] / r.Support[a] : 0.0;
                var cell = _dgvConfusion.Rows[rowIndex].Cells[p + 1];
                cell.Style.BackColor = Color.FromArgb(255 - (int)(150 * share), 255 - (int)(95 * share), 255);
                cell.Style.ForeColor = Color.Black;
                if (a == p) cell.Style.Font = new Font(_dgvConfusion.Font, FontStyle.Bold);
            }
            _dgvConfusion.Rows[rowIndex].Cells[0].Style.Font = new Font(_dgvConfusion.Font, FontStyle.Bold);
        }
    }

    // =====================================================================================
    //  Single prediction
    // =====================================================================================

    private void BuildPredictInputs()
    {
        _pnlPredictInputs.Controls.Clear();
        _predictBoxes.Clear();
        var table = new TableLayoutPanel { Dock = DockStyle.Top, AutoSize = true, ColumnCount = 2 };
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 62));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 38));

        for (int i = 0; i < _ds.NumFeatures; ++i)
        {
            FeatureStat s = _ds.Stats[i];
            table.Controls.Add(new Label
            {
                Text = $"{s.Name}  ({s.Min:F1}-{s.Max:F1})",
                AutoSize = true,
                Padding = new Padding(0, 6, 4, 0)
            }, 0, i);
            var box = new TextBox { Dock = DockStyle.Fill, Text = _ds.ImputeValues[i].ToString("F1", CultureInfo.InvariantCulture) };
            _predictBoxes.Add(box);
            table.Controls.Add(box, 1, i);
        }
        _pnlPredictInputs.Controls.Add(table);
    }

    private void Predict()
    {
        if (_ds == null || _nn == null)
        {
            MessageBox.Show(this, "Train a model first (tab 2).", "No trained model");
            return;
        }

        var raw = new double[_ds.NumFeatures];
        for (int i = 0; i < raw.Length; ++i)
        {
            string text = _predictBoxes[i].Text.Trim();
            if (text.Length == 0) { raw[i] = double.NaN; continue; }
            if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out raw[i]))
            {
                MessageBox.Show(this, $"'{text}' is not a number (field: {_ds.FeatureNames[i]}). Use a dot as decimal separator.", "Invalid input");
                return;
            }
        }

        double[] probs = _nn.ComputeOutputs(_ds.ScaleRaw(raw));
        int best = ClassificationMetrics.ArgMax(probs);

        var sb = new StringBuilder();
        sb.AppendLine($"Predicted outcome:  {_ds.ClassNames[best]}   (output activation {probs[best]:F3})");
        sb.AppendLine();
        sb.AppendLine("Output neuron activations (0-1, highest wins; they are not forced to sum to 1):");
        for (int c = 0; c < probs.Length; ++c)
            sb.AppendLine($"  {_ds.ClassNames[c],-14} {probs[c],7:F3}  {new string('#', (int)Math.Round(probs[c] * 40))}");
        sb.AppendLine();
        sb.AppendLine("Input after missing-value fill and MinMax scaling:");
        double[] scaled = _ds.ScaleRaw(raw);
        for (int i = 0; i < scaled.Length; ++i)
            sb.AppendLine($"  {_ds.FeatureNames[i],-22} {scaled[i],7:F3}");
        _txtPrediction.Text = sb.ToString();
    }

    // =====================================================================================
    //  Helpers
    // =====================================================================================

    private sealed class EpochLog
    {
        public int Epoch { get; init; }
        public double TrainLoss { get; init; }
        public double ValLoss { get; init; }
        public double TrainAcc { get; init; }
        public double ValAcc { get; init; }
    }

    private void SetBusy(bool busy)
    {
        _busy = busy;
        _btnTrain.Enabled = !busy;
        _btnStop.Enabled = busy;
        _btnTune.Enabled = !busy;
        _btnTuneCancel.Enabled = busy;
    }

    private void AppendLog(string line) => _txtLog.AppendText(line + Environment.NewLine);

    private void SetStatus(string text) => _statusLabel.Text = text;

    private void OpenOutputFolder()
    {
        Directory.CreateDirectory(_outputDir);
        Process.Start(new ProcessStartInfo(_outputDir) { UseShellExecute = true });
    }

    private static string F(double v) => v.ToString("F6", CultureInfo.InvariantCulture);

    private static decimal Clamp(NumericUpDown nud, decimal v) => Math.Max(nud.Minimum, Math.Min(nud.Maximum, v));

    private static double[] ParseDoubles(string text) =>
        text.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(s => double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture)).ToArray();

    private static int[] ParseInts(string text) =>
        text.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();

    private static void AddRow(TableLayoutPanel table, string label, Control control)
    {
        int row = table.RowCount++;
        table.Controls.Add(new Label { Text = label, AutoSize = true, Padding = new Padding(0, 6, 0, 0) }, 0, row);
        control.Dock = DockStyle.Fill;
        table.Controls.Add(control, 1, row);
    }

    private static NumericUpDown MakeNud(decimal min, decimal max, int decimals, decimal step, decimal value) =>
        new()
        {
            Minimum = min,
            Maximum = max,
            DecimalPlaces = decimals,
            Increment = step,
            Value = value
        };

    private static DataGridView MakeGrid() =>
        new()
        {
            Dock = DockStyle.Fill,
            ReadOnly = true,
            AllowUserToAddRows = false,
            AllowUserToDeleteRows = false,
            RowHeadersVisible = false,
            AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
            SelectionMode = DataGridViewSelectionMode.FullRowSelect,
            BackgroundColor = Color.White
        };

    private static TextBox MakeMonoBox() =>
        new()
        {
            Dock = DockStyle.Fill,
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Both,
            WordWrap = false,
            Font = new Font("Consolas", 9.5f),
            BackColor = Color.White
        };
}
