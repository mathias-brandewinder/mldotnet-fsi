#r "./packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Core.dll"
#r "./packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Data.dll"
#r "./packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.StandardTrainers.dll"
#r "./packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Transforms.dll"

#r "./packages/Microsoft.ML.DataView/lib/netstandard2.0/Microsoft.ML.DataView.dll"

#r "./packages/Microsoft.ML.CpuMath/lib/netstandard2.0/Microsoft.ML.CpuMath.dll"

#r "./packages/Microsoft.ML.FastTree/lib/netstandard2.0/Microsoft.ML.FastTree.dll"

#r "./packages/Newtonsoft.Json/lib/netstandard2.0/Newtonsoft.Json.dll"

open System
open System.IO

open Microsoft.ML
open Microsoft.ML.Data

// native libraries
let path = Environment.GetEnvironmentVariable("PATH")

let nativeDllsPaths = 
    [
        "Microsoft.ML/runtimes/win-x64/native"
        "Microsoft.ML.CpuMath/runtimes/win-x64/nativeassets/netstandard2.0"
        "Microsoft.ML.FastTree/runtimes/win-x64/native"
    ]
    |> List.map (fun subPath -> 
        Path.Combine(__SOURCE_DIRECTORY__, "packages", subPath)
        |> Path.GetFullPath
        )

path :: nativeDllsPaths
|> String.concat ";"
|> fun expandedPathReferences -> 
    Environment.SetEnvironmentVariable("PATH", expandedPathReferences)

type WineDescription = {
    [<LoadColumn(0)>]
    Sulphates: float32
    [<LoadColumn(1)>]
    Alcohol: float32
    [<LoadColumn(2)>]
    Quality: float32 
    }

let context = MLContext (seed = Nullable 0)

let dataPath = 
    Path.Combine (__SOURCE_DIRECTORY__, "dataset.csv")

let dataView = 
    context.Data.LoadFromTextFile<WineDescription>(
        path = dataPath,
        hasHeader = true,
        separatorChar = ','
        )

let testTrain = 
    context.Data.TrainTestSplit(
        data = dataView, 
        testFraction = 0.5, 
        seed = Nullable 0
        )

let train = testTrain.TrainSet
let test = testTrain.TestSet

let dataProcessPipeline =
    EstimatorChain()
        .Append(context.Transforms.CopyColumns("Label", "Quality"))
        .Append(
            context.Transforms.Concatenate(
                "Features",
                "Sulphates", 
                "Alcohol"
                )
            )

let trainer = 
    context.Regression.Trainers.FastForest(
        labelColumnName = "Label", 
        featureColumnName = "Features"
        )

let modelBuilder = dataProcessPipeline.Append trainer
let trainedModel = modelBuilder.Fit train

let trainEvaluation = 
    let predictions = trainedModel.Transform train
    context.Regression.Evaluate(predictions, "Label", "Score")

printfn "MAE: %f" trainEvaluation.MeanAbsoluteError
printfn "MSE: %f" trainEvaluation.MeanSquaredError

let testEvaluation = 
    let predictions = trainedModel.Transform test
    context.Regression.Evaluate(predictions, "Label", "Score")

printfn "MAE: %f" testEvaluation.MeanAbsoluteError
printfn "MSE: %f" testEvaluation.MeanSquaredError
