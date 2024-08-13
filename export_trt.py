import torch
import time
from trt_utilities import Engine


def export_trt(trt_path: str, onnx_path: str, use_fp16: bool):
    engine = Engine(trt_path)

    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")

    return ret


export_trt(trt_path="./yolox_l.engine",
           onnx_path="./yolox_l.onnx", use_fp16=True)
export_trt(trt_path="./dw-ll_ucoco_384.engine",
           onnx_path="./dw-ll_ucoco_384.onnx", use_fp16=True)
