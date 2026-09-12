"""Native TensorRT/CUDA worker for JP5 Python 3.8 and JP6 Python 3.10.
Private pipes only; no networking, DDS or actuator commands.
"""

import ctypes as ct
import ctypes.util
import json
import sys
import numpy as np
import tensorrt as trt


class Engine:
    def __init__(self, path):
        self.cuda = ct.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with open(path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Cannot deserialize " + path)
        self.context = self.engine.create_execution_context()
        self.stream = ct.c_void_p()
        self.check(self.cuda.cudaStreamCreate(ct.byref(self.stream)))
        self.v3 = int(trt.__version__.split(".")[0]) >= 10
        self.inputs, self.outputs, self.bindings, self.allocations = [], [], [], []
        count = self.engine.num_io_tensors if self.v3 else self.engine.num_bindings
        for i in range(count):
            name = (
                self.engine.get_tensor_name(i)
                if self.v3
                else self.engine.get_binding_name(i)
            )
            shape = tuple(
                self.engine.get_tensor_shape(name)
                if self.v3
                else self.engine.get_binding_shape(i)
            )
            dtype = (
                self.engine.get_tensor_dtype(name)
                if self.v3
                else self.engine.get_binding_dtype(i)
            )
            if min(shape) <= 0 or dtype != trt.float32:
                raise ValueError("Only static FP32 I/O supported")
            a = np.empty(shape, dtype=np.float32)
            ptr = ct.c_void_p()
            self.check(self.cuda.cudaMalloc(ct.byref(ptr), ct.c_size_t(a.nbytes)))
            self.allocations.append(ptr)
            self.bindings.append(ptr.value)
            inp = (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if self.v3
                else self.engine.binding_is_input(i)
            )
            (self.inputs if inp else self.outputs).append((name, a, ptr))
            if self.v3:
                self.context.set_tensor_address(name, ptr.value)
        order = {"prop": 0, "actions": 1, "task": 2, "observation": 0}
        self.inputs.sort(key=lambda x: order[x[0]])
        if len(self.outputs) != 1:
            raise ValueError("Expected one output")

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError("CUDA runtime error " + str(code))

    def infer(self, data):
        offset = 0
        for _, a, ptr in self.inputs:
            a[:] = np.frombuffer(
                data, dtype=np.float32, count=a.size, offset=offset
            ).reshape(a.shape)
            offset += a.nbytes
            self.check(
                self.cuda.cudaMemcpyAsync(
                    ptr,
                    ct.c_void_p(a.ctypes.data),
                    ct.c_size_t(a.nbytes),
                    1,
                    self.stream,
                )
            )
        ok = (
            self.context.execute_async_v3(self.stream.value)
            if self.v3
            else self.context.execute_async_v2(self.bindings, self.stream.value)
        )
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        _, out, ptr = self.outputs[0]
        self.check(
            self.cuda.cudaMemcpyAsync(
                ct.c_void_p(out.ctypes.data),
                ptr,
                ct.c_size_t(out.nbytes),
                2,
                self.stream,
            )
        )
        self.check(self.cuda.cudaStreamSynchronize(self.stream))
        if not np.isfinite(out).all():
            raise RuntimeError("non-finite TensorRT output")
        return out.tobytes()

    def close(self):
        self.cuda.cudaStreamSynchronize(self.stream)
        for p in self.allocations:
            self.cuda.cudaFree(p)
        self.cuda.cudaStreamDestroy(self.stream)


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError("incomplete request")
        data.extend(chunk)
    return data


def main():
    engines = []
    try:
        for path in sys.argv[1:]:
            engines.append(Engine(path))
        print(
            json.dumps(
                dict(
                    tensorrt=trt.__version__,
                    models=[
                        dict(
                            inputs=[list(a.shape) for _, a, _ in e.inputs],
                            output=list(e.outputs[0][1].shape),
                        )
                        for e in engines
                    ],
                )
            ),
            flush=True,
        )
        while True:
            index = sys.stdin.buffer.read(1)
            if not index:
                break
            engine = engines[index[0]]
            data = read_exact(
                sys.stdin.buffer, sum(a.nbytes for _, a, _ in engine.inputs)
            )
            sys.stdout.buffer.write(engine.infer(data))
            sys.stdout.buffer.flush()
    finally:
        for e in engines:
            e.close()


if __name__ == "__main__":
    main()
