using System.ComponentModel;
using System.Drawing.Drawing2D;
using System.Globalization;

namespace NeuralNetPredictor.UI;

/// <summary>Small GDI+ line chart showing training loss and validation loss per logged epoch.</summary>
public sealed class LossChartControl : Control
{
    private readonly List<(int Epoch, double Train, double Val)> _points = new();

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public string Title { get; set; } = "Loss per epoch (1/2 squared error)";

    public LossChartControl()
    {
        DoubleBuffered = true;
        ResizeRedraw = true;
        BackColor = Color.White;
        MinimumSize = new Size(200, 120);
    }

    public void Clear()
    {
        _points.Clear();
        Invalidate();
    }

    public void AddPoint(int epoch, double trainLoss, double valLoss)
    {
        _points.Add((epoch, trainLoss, valLoss));
        Invalidate();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        var g = e.Graphics;
        g.SmoothingMode = SmoothingMode.AntiAlias;
        g.Clear(BackColor);

        const int left = 58, right = 18, top = 34, bottom = 38;
        var plot = new Rectangle(left, top, Width - left - right, Height - top - bottom);
        if (plot.Width < 20 || plot.Height < 20) return;

        using var titleFont = new Font(Font.FontFamily, 10f, FontStyle.Bold);
        using var smallFont = new Font(Font.FontFamily, 8f);
        g.DrawString(Title, titleFont, Brushes.Black, 8, 6);

        using var axisPen = new Pen(Color.Gray);
        using var gridPen = new Pen(Color.Gainsboro);
        g.DrawRectangle(axisPen, plot);

        if (_points.Count == 0)
        {
            g.DrawString("Train a model to see the loss curve.", smallFont, Brushes.Gray, plot.Left + 10, plot.Top + 10);
            return;
        }

        int xMin = _points[0].Epoch, xMax = _points[_points.Count - 1].Epoch;
        if (xMax == xMin) xMax = xMin + 1;
        double yMax = 0.0;
        foreach (var p in _points)
        {
            if (!double.IsNaN(p.Train) && !double.IsInfinity(p.Train)) yMax = Math.Max(yMax, p.Train);
            if (!double.IsNaN(p.Val) && !double.IsInfinity(p.Val)) yMax = Math.Max(yMax, p.Val);
        }
        if (yMax <= 0.0) yMax = 1.0;
        yMax *= 1.08;

        float X(int epoch) => plot.Left + (float)(epoch - xMin) / (xMax - xMin) * plot.Width;
        float Y(double v) => plot.Bottom - (float)(Math.Min(Math.Max(v, 0.0), yMax) / yMax) * plot.Height;

        // grid + y labels
        for (int i = 0; i <= 5; ++i)
        {
            double v = yMax * i / 5.0;
            float y = Y(v);
            g.DrawLine(gridPen, plot.Left, y, plot.Right, y);
            g.DrawString(v.ToString("F2", CultureInfo.InvariantCulture), smallFont, Brushes.DimGray, 8, y - 7);
        }
        // x labels
        for (int i = 0; i <= 5; ++i)
        {
            int epoch = xMin + (int)Math.Round((xMax - xMin) * i / 5.0);
            float x = X(epoch);
            g.DrawLine(axisPen, x, plot.Bottom, x, plot.Bottom + 4);
            g.DrawString(epoch.ToString(CultureInfo.InvariantCulture), smallFont, Brushes.DimGray, x - 10, plot.Bottom + 6);
        }
        g.DrawString("epoch", smallFont, Brushes.DimGray, plot.Right - 34, plot.Bottom + 20);

        DrawSeries(g, plot, Color.RoyalBlue, p => p.Train, X, Y);
        DrawSeries(g, plot, Color.OrangeRed, p => p.Val, X, Y);

        // legend + latest values
        var last = _points[_points.Count - 1];
        using var trainBrush = new SolidBrush(Color.RoyalBlue);
        using var valBrush = new SolidBrush(Color.OrangeRed);
        g.FillRectangle(trainBrush, plot.Right - 250, 10, 10, 10);
        g.DrawString($"train {last.Train:F4}", smallFont, Brushes.Black, plot.Right - 236, 8);
        g.FillRectangle(valBrush, plot.Right - 130, 10, 10, 10);
        g.DrawString($"validation {last.Val:F4}", smallFont, Brushes.Black, plot.Right - 116, 8);
    }

    private void DrawSeries(Graphics g, Rectangle plot, Color color,
        Func<(int Epoch, double Train, double Val), double> selector,
        Func<int, float> toX, Func<double, float> toY)
    {
        using var pen = new Pen(color, 2f);
        using var brush = new SolidBrush(color);
        PointF? prev = null;
        foreach (var p in _points)
        {
            double v = selector(p);
            if (double.IsNaN(v) || double.IsInfinity(v)) { prev = null; continue; }
            var pt = new PointF(toX(p.Epoch), toY(v));
            if (prev.HasValue) g.DrawLine(pen, prev.Value, pt);
            g.FillEllipse(brush, pt.X - 2.5f, pt.Y - 2.5f, 5f, 5f);
            prev = pt;
        }
    }
}
