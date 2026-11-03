# NeuralNetPredictor - Iris Species Classifier - Custom C# Multi-Layer Perceptron (WinForms)

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

## Hyper-parameter tuning (`Tuning/HyperparameterTuner.cs`)

The grid defaults to learning rate {0.05, 0.1, 0.2, 0.3, 0.5} x hidden nodes {4, 8, 12, 16} x epochs {50, 100, 200, 300, 500}
(100 experiments x 3 random starts; every pair is trained once and scored at each epoch checkpoint, which equals separate runs but is much faster).
Experiments are ranked by **validation accuracy** (ties: lower validation loss). The table also shows the train-validation gap
to spot over-fitting. Edit the three text boxes to try other grids.

### Tuning results (example run - yours will differ slightly because of the random starts)

Averages over the whole grid (validation accuracy / validation loss):

| learning rate | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| val accuracy | 0.914 | 0.947 | 0.954 | 0.956 | 0.953 |
| val loss | 0.097 | 0.065 | 0.045 | 0.037 | 0.033 |

| hidden nodes | 4 | 8 | 12 | 16 |
|---|---|---|---|---|
| val accuracy | 0.939 | 0.946 | 0.947 | 0.948 |

| epochs | 50 | 100 | 200 | 300 | 500 |
|---|---|---|---|---|---|
| val accuracy | 0.901 | 0.952 | 0.956 | 0.956 | 0.958 |
| val loss | 0.100 | 0.065 | 0.043 | 0.036 | 0.033 |

* Learning rate 0.05 is clearly too slow for this online sigmoid network; accuracy levels off from 0.2 upward and loss keeps falling to 0.5.
* More hidden nodes help only slightly (4 -> 8 nodes); beyond 8 nothing changes, so a small network is enough.
* Most of the gain arrives by about 100-200 epochs; after that the validation loss creeps down but accuracy barely moves.
* The top configuration was lr 0.1, 16 hidden nodes, 100 epochs (validation accuracy 98.6 % averaged over 3 starts).
  Several others (e.g. lr 0.2, 8-16 nodes, 200 epochs) are within one validation sample of it, so the choice is not sharp.
  Re-run the grid in the *Tune* tab and paste **your** top rows here.

### Final model (example run)

lr 0.1, 16 hidden nodes, 100 epochs. Test accuracy 95.2 % (20 of 21), macro F1 0.952, majority baseline 33.3 %.
The single error is a `versicolor` predicted as `virginica`, which is the known overlap between those two species.

## Project layout

```
NeuralNetPredictor.slnx
NeuralNetPredictor/
  Backprop/                      the course classes: INeuron, HNeuron, ONeuron, NeuralNet
  Network/BackpropClassifier.cs  wrapper: epoch loop, shuffling, loss, accuracy (no learning code)
  Pipeline/DataPipeline.cs       CSV parsing, missing values, stratified split, scaling
  Evaluation/ClassificationMetrics.cs   accuracy, precision, recall, F1, confusion matrix
  Tuning/HyperparameterTuner.cs  grid search
  UI/MainForm.cs, LossChartControl.cs   WinForms front end (no designer files)
  Data/iris.csv                  dataset
```

