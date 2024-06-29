#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Determine the directory of the script or the notebook *)
dir = DirectoryName@$InputFileName;
If[dir == "", dir = NotebookDirectory[]];

(* Load necessary packages *)
Get[FileNameJoin@{dir, "KymoButler.wl"}];
Get[FileNameJoin@{dir, "KymoButlerPProc.wl"}];

(* Load the default neural networks models *)
models = Quiet[loadDefaultNets[dir <> ".."]];

(* Process each image file passed as command line argument *)
Do[
    Print["Processing file: ", imageFile];
    
    (* Example of unidirectional processing *)
    (* resultUNI = UniKymoButler[Import[imageFile], .2, "CPU", models["uninet"], 3, 3]; *)
    (* Print["Result for ", imageFile, ": ", resultUNI]; *)
    
    (* Extend this block to include bidirectional processing or any additional analysis you need *)
    resultBI = BiKymoButler[Import[imageFile], .2, .5, "CPU", models["binet"], models["decnet"], 10, 10];
    Print["Bidirectional Result for ", imageFile, ": ", resultBI];
    
    (* Add post-processing or testing code as needed *)
    
    (* Verification test example *)
    #testResult = VerificationTest[Length@resultUNI[[-1]], ExpectedValue];  (* Replace ExpectedValue with actual expected value *)
    #Print["Test Result for ", imageFile, ": ", testResult];
    
    (* Additional handling like saving results, plotting, etc. *)
,
    {imageFile, Rest[$ScriptCommandLine]}
];

