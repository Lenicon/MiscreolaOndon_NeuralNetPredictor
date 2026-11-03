# Iris Species Classifier - Custom C# Multi-Layer Perceptron (WinForms)

CS345 activity: *Real-World Prediction with a Custom Multi-Layer Perceptron in C#*.
Track: **Multi-class classification**. The only machine-learning code is the course's input -> hidden -> output
back-propagation classes (`INeuron`, `HNeuron`, `ONeuron`, `NeuralNet`, namespace `Backprop`, in the `Backprop/` folder).
No ML libraries, no NuGet packages.

## How to run

1. Install **Visual Studio 2022 17.13+** or **Visual Studio 2026** with the *.NET desktop development* workload
   (`.slnx` files need one of these; the .NET 9+ CLI also opens them).
2. Open `NeuralNetPredictor.slnx`, press **F5**.
3. The dataset loads automatically. Then:
   * **1. Data** - preview the CSV, class balance, missing values, per-feature statistics.
   * **2. Train** - set hidden nodes, epochs and learning rate, press *Train*. Loss/accuracy are logged live, charted, and saved to
     `Output/training_log_<time>.csv` (next to the .exe, `bin\Debug\net8.0-windows\Output`).
   * **3. Tune** - grid search over learning rate x hidden nodes x epochs; the best row is highlighted and can be
     copied to the Train tab. Results are saved to `Output/tuning_results_<time>.csv`.
   * **4. Evaluate** - accuracy, precision / recall / F1 per class, confusion matrix, and written analysis.
   * **5. Predict** - enter raw values for one flower and get the output-neuron activations.

## Data description

`NeuralNetPredictor/Data/iris.csv` is the classic **Iris flower dataset** (Fisher, 1936; UCI Machine Learning Repository):
150 flowers, 3 species, 50 each. Four measurements in centimetres are used to predict the species.

| Column | Meaning | Range (cm) |
|---|---|---|
| SepalLengthCm | sepal length | 4.3 - 7.9 |
| SepalWidthCm | sepal width | 2.0 - 4.4 |
| PetalLengthCm | petal length | 1.0 - 6.9 |
| PetalWidthCm | petal width | 0.1 - 2.5 |
| **Species** (label) | `setosa`, `versicolor`, `virginica` | 50 / 50 / 50 rows |

`setosa` is linearly separable from the other two; `versicolor` and `virginica` overlap slightly, so a few
mistakes between those two are expected. The file has no missing values (the pipeline still handles blanks).
With only 150 rows the stratified 70/15/15 split gives 105 / 24 / 21 samples, so accuracy moves in steps of roughly 4-5 %
on the validation and test sets; judge settings by trends, not tiny differences. To use another dataset, press
*Load CSV...*: header row, numeric feature columns, **last column = class label**.

## Pipeline (`Pipeline/DataPipeline.cs`)

1. **Parse** the CSV (invariant culture, quoted cells, clear error message with row / column on bad data).
2. **Missing values** (`""`, `NA`, `?`, ...) are replaced by the column **mean of the training split**.
   Rows with a missing label are skipped.
3. **Stratified shuffle** (Fisher-Yates, fixed seed) and **split 70 % train / 15 % validation / 15 % test**; every class is split in the same proportions.
4. **MinMax scale** every feature to [0, 1]. Min / max / means are learned from the **training split only**
   so no information leaks from validation / test data.
5. **One-hot encode** the label (1-of-N), with one sigmoid output neuron per class.

Validation data is used to choose hyper-parameters; the test split is touched only for the final report.

## Network (`Backprop/` + `Network/BackpropClassifier.cs`)

`inputs -> sigmoid hidden neurons -> sigmoid output neurons` using **only the course classes**:
`INeuron` (input layer + input->hidden weights), `HNeuron` (hidden), `ONeuron` (output) and `NeuralNet`
(forward pass, error component, derivative, weight and bias updates in `learn()` - the 10 worksheet steps).

* One output neuron per class, trained on 1-of-N targets; the predicted class is the neuron with the highest activation.
* `Network/BackpropClassifier.cs` is a thin wrapper that adds **no learning code**: it feeds a sample to `NeuralNet`,
  calls `learn()`, shuffles the order every epoch, and computes loss / accuracy / cancel support for the UI.
* Loss = mean of 0.5 x sum(target - output)^2 per sample (the E of the slides).
* **Only change to the course classes:** `LRPOUT` and `LRPIN` in `NeuralNet.cs` were `const`; they are now plain fields with a
  `setLearningRates()` method so the learning rate can be tuned (defaults 0.2 / 0.15 unchanged). The UI's single "learning rate" sets both.
* The course neurons start from random weights using an unseeded `Random`, so results differ slightly from run to run.
  The tuner therefore trains every grid cell **3 times and averages** the scores.
