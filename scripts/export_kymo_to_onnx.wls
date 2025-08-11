(* scripts/export_kymo_to_onnx.wls *)
ClearAll["Global`*"];

(* ---------------- Args & defaults ---------------- *)
args      = Rest @ $ScriptCommandLine;
modelsDir = If[Length[args] >= 1, args[[1]], "models"];
outDir    = If[Length[args] >= 2, args[[2]], "export"];
segSize   = If[Length[args] >= 3, ToExpression @ args[[3]], 256];  (* square; multiple of 16 *)

If[!DirectoryQ[modelsDir], Print["ERROR: Models dir not found: ", modelsDir]; Exit[1]];
If[!IntegerQ@segSize || segSize <= 0 || Mod[segSize, 16] =!= 0,
  Print["WARN: segSize should be a positive multiple of 16. Using 256."]; segSize = 256;
];

Quiet @ CreateDirectory[outDir, CreateIntermediateDirectories -> True];

$wlVersion = $VersionNumber;
Print["Wolfram version: ", $wlVersion];

(* ---------------- Model specs ---------------- *)
(* dims are {C,H,W}; encSpec drives how we attach encoders for export *)
specs = {
  <|"file"->"Unidrectional_Segmentation_Module_V1_0", "nick"->"uni_seg",
    "dims"->{1, segSize, segSize}, "enc"->{"Image","Grayscale"}|>,
  <|"file"->"Bidirectional_Segmentation_Module_V1_1", "nick"->"bi_seg",
    "dims"->{1, segSize, segSize}, "enc"->{"Image","Grayscale"}|>,
  <|"file"->"Classification_Module_V1_0", "nick"->"classifier",
    "dims"->{1, 64, 64},           "enc"->{"Image","Grayscale"}|>,
  <|"file"->"Decision_Module_V1_0", "nick"->"decision",
    "dims"->{3, 48, 48},           "enc"->{"Image","RGB"}|>  (* convenient 3-ch wrapper *)
};

(* ---------------- Helpers ---------------- *)

getInputKey[net_] := Module[{ports = Quiet @ NetInformation[net, "InputPorts"]},
  Which[
    AssociationQ @ ports && KeyExistsQ[ports, "Input"], "Input",
    AssociationQ @ ports && Length @ ports >= 1, First @ Keys @ ports,
    True, None
  ]
];

attachEncoder[net_, dims:{c_,h_,w_}, encSpec:{encType_,encColor_}] := Module[
  {key = getInputKey[net], enc},
  If[key === None, Return[$Failed]];
  enc = Which[
    encType === "Image" && encColor === "Grayscale" && c == 1,
      NetEncoder[{"Image", {h, w}, "Grayscale"}],
    encType === "Image" && encColor === "RGB" && c == 3,
      NetEncoder[{"Image", {h, w}, "RGB"}],
    True, Return[$Failed]
  ];
  Quiet @ Check[ NetReplacePart[net, key -> enc], $Failed ]
];

validateFile[path_] := FileExistsQ[path] && Quiet@Check[FileByteCount[path] > 0, False];

(* Try ONNX export *)
tryExportONNX[net_, tgt_] := Quiet @ Check[ Export[tgt, net, "ONNX"], $Failed ];

(* Fallback: MXNet export pair *)
tryExportMXNet[net_, prefix_] := Module[{ok = $Failed, base},
  base = FileNameJoin[{DirectoryName[prefix], FileBaseName[prefix]}];
  ok = Quiet @ Check[ Export[base, net, "MXNet"], $Failed ];
  If[ok === $Failed, Return[$Failed]];
  If[ !validateFile[prefix <> "-symbol.json"] || !validateFile[prefix <> "-0000.params"], Return[$Failed]];
  ok
];

(* After ONNX export: re-import and run a dummy forward *)
forwardTestONNX[onnxPath_, dims:{c_,h_,w_}] := Module[
  {net = Quiet @ Check[ Import[onnxPath], $Failed ], key, x, y, outShapes},
  If[net === $Failed, Return[<|"ok"->False, "error"->"onnx_import_failed"|>]];
  key = getInputKey[net];
  If[key === None, Return[<|"ok"->False, "error"->"no_input_port"|>]];
  x = RandomReal[{0,1}, {1, c, h, w}];  (* NCHW *)
  y = Quiet @ Check[ net[x], $Failed ];
  If[y === $Failed, y = Quiet @ Check[ net[Association[key -> x]], $Failed ]];
  If[y === $Failed, Return[<|"ok"->False, "error"->"forward_failed"|>]];
  outShapes =
    Which[
      AssociationQ[y], KeyValueMap[#1 -> If[ArrayQ[#2], Dimensions[#2], "nonArray"] &, y],
      ArrayQ[y], Dimensions[y],
      True, "unknown"
    ];
  <|"ok"->True, "out_shapes"->outShapes|>
];

(* --- Fixed: robust 'add' helper (avoids ___Rule parser issue) --- *)
add[entry_] := entry;
add[entry_, kv__Rule] := Join[entry, <|kv|>];

report = <|"version" -> $wlVersion, "exports" -> {}|>;

(* ---------------- Main loop ---------------- *)
Do[
  Module[{src = FileNameJoin[{modelsDir, spec["file"]}],
          nick = spec["nick"], dims = spec["dims"], encSpec = spec["enc"],
          onnxPath, mxPrefix, net, netEnc, ok, entry, fwd},
    entry = <|"name"->nick, "source"->src, "dims"->dims, "status"->"start"|>;

    If[!FileExistsQ[src],
      entry = add[entry, "error"->"missing source"]; report["exports"] = Append[report["exports"], entry]; Continue[];
    ];

    Print["Exporting: ", spec["file"], " -> ", nick, " with dims ", dims];

    net = Quiet @ Check[ Import[src], $Failed ];
    If[net === $Failed,
      entry = add[entry, "error"->"import failed"]; report["exports"] = Append[report["exports"], entry]; Continue[];
    ];

    netEnc = attachEncoder[net, dims, encSpec];
    If[netEnc === $Failed,
      entry = add[entry, "error"->"attachEncoder failed"]; report["exports"] = Append[report["exports"], entry]; Continue[];
    ];

    onnxPath = FileNameJoin[{outDir, nick <> ".onnx"}];
    ok = tryExportONNX[netEnc, onnxPath];

    If[ok =!= $Failed && validateFile[onnxPath],
      entry = add[entry, "onnx"->onnxPath, "status"->"onnx_ok"];
      (* Re-import ONNX and run a dummy forward *)
      fwd = forwardTestONNX[onnxPath, dims];
      entry = add[entry, "onnx_forward_ok" -> Lookup[fwd, "ok", False]];
      If[TrueQ[fwd["ok"]], 
        entry = add[entry, "onnx_out_shapes" -> fwd["out_shapes"]],
        entry = add[entry, "onnx_forward_err" -> Lookup[fwd, "error", "unknown"]]
      ];
      ,
      (* Fallback to MXNet *)
      mxPrefix = FileNameJoin[{outDir, nick}];
      ok = tryExportMXNet[netEnc, mxPrefix];
      If[ok === $Failed,
        entry = add[entry, "status"->"failed_both", "error"->"ONNX and MXNet export failed"],
        entry = add[entry, "status"->"mxnet_ok",
                          "mxnet_symbol"->(mxPrefix <> "-symbol.json"),
                          "mxnet_params"->(mxPrefix <> "-0000.params")]
      ];
    ];

    report["exports"] = Append[report["exports"], entry];
  ],
  {spec, specs}
];

(* ---------------- Write report ---------------- *)
reportPath = FileNameJoin[{outDir, "export_report.json"}];
Export[reportPath, report, "JSON"];
Print["Report written: ", reportPath];
Print["Done. Saved to ", outDir];
Exit[0];
