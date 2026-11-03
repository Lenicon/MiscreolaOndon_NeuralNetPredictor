namespace NeuralNetPredictor;

internal static class Program
{
    /// <summary>Application entry point.</summary>
    [STAThread]
    private static void Main()
    {
        ApplicationConfiguration.Initialize();
        Application.Run(new UI.MainForm());
    }
}
