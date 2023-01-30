# From 2D to 3D: Re-thinking Benchmarking of Monocular Depth Prediction]

This directory consists of 3D point cloud metrics for evaluating monocular depth prediction proposed in [From 2D to 3D: Re-thinking Benchmarking of Monocular Depth Prediction](https://arxiv.org/pdf/2111.14673.pdf). We also provide the point cloud utilization functions and commonly used 2D and iBims occlusion boundary metrics.


# Additional repositories

- Along with this repo, Kaolin library provides nice utilities: https://github.com/NVIDIAGameWorks/kaolin/
- Occlusion boundary metric adopted from: [iBims](https://arxiv.org/pdf/1805.01328v1.pdf) and https://github.com/MichaelRamamonjisoa/SharpNet 
- Other Chamfer Distance implementations:
	- https://github.com/ThibaultGROUEIX/ChamferDistancePytorch	
	- https://github.com/chrdiller/pyTorchChamferDistance


## References
If you use proposed 3d metrics in monodepth evaluation or wish to refer our arxiv paper, please cite
```
@inproceedings{ornek2022from2dto3d,
  title={From 2D to 3D: Re-thinking Benchmarking of Monocular Depth Prediction},
  author={{\"O}rnek, Evin P{\i}nar and Mudgal, Shristi and Wald, Johanna and Wang, Yida and Navab, Nassir and Tombari, Federico},
  publisher = {arXiv},
  year = {2022},
  url = {https://arxiv.org/abs/2203.08122}
}
```