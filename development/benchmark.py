"""
Run benchmarks
--------------
1. Install pandas tabulate dependency `python -m pip install tabulate`
2. Run benchmark script, eg: `python benchmark.py --model_type vit_h --device cpu`

Line profiling
--------------
1. Install line profiler: `python -m pip install line_profiler`
2. Add `@profile` decorator to any function in the call stack
3. Run `kernprof -lv benchmark.py --model_type vit_h --device cpu`

Snakeviz visualization
----------------------
https://jiffyclub.github.io/snakeviz/
1. Install snakeviz: `python -m pip install snakeviz`
2. Generate profile file: `python -m cProfile -o program.prof benchmark.py --model_type vit_h --device cpu`
3. Visualize profile file: `snakeviz program.prof`
"""
import argparse
import time

import imageio.v3 as imageio
import micro_sam.instance_segmentation as instance_seg
import micro_sam.prompt_based_segmentation as seg
import micro_sam.util as util
import numpy as np
import pandas as pd

from micro_sam.sample_data import fetch_livecell_example_data


def _get_image_and_predictor(model_type, device, checkpoint_path, image_path):
    #example_data = fetch_livecell_example_data("../examples/data")
    example_data = image_path
    image = imageio.imread(example_data)
    predictor = util.get_custom_sam_model(checkpoint_path, model_type, device)
    return image, predictor


def _add_result(benchmark_results, name, runtimes):
    nres = len(name)
    assert len(name) == len(runtimes)
    res = {
        "benchmark": name,
        "mean runtime": runtimes
    }
    tab = pd.DataFrame(res)
    benchmark_results.append(tab)
    return benchmark_results


def benchmark_embeddings(image, predictor, n):
    print("Running benchmark_embeddings ...")
    n = 3 if n is None else n
    times = []
    for _ in range(n):
        t0 = time.time()
        util.precompute_image_embeddings(predictor, image)
        times.append(time.time() - t0)
    runtime = np.mean(times[1:])

    return ["embeddings"], [runtime]


def benchmark_prompts(image, predictor, n):
    print("Running benchmark_prompts ...")
    n = 10 if n is None else n
    names, runtimes = [], []

    np.random.seed(42)

    names, runtimes = [], []

    # from random single point
    times = []
    for _ in range(n):
        t0 = time.time()
        embeddings = util.precompute_image_embeddings(predictor, image)
        points = np.array([
            np.random.randint(0, image.shape[0]),
            np.random.randint(0, image.shape[1]),
        ])[None]
        labels = np.array([1])
        seg.segment_from_points(predictor, points, labels, embeddings)
        times.append(time.time() - t0)
    names.append("prompt-p1n0")
    runtimes.append(np.mean(times[1:]))

    """
    # from random 2p4n
    times = []
    for _ in range(n):
        t0 = time.time()
        embeddings = util.precompute_image_embeddings(predictor, image)
        points = np.concatenate([
            np.random.randint(0, image.shape[0], size=6)[:, None],
            np.random.randint(0, image.shape[1], size=6)[:, None],
        ], axis=1)
        labels = np.array([1, 1, 0, 0, 0, 0])
        seg.segment_from_points(predictor, points, labels, embeddings)
        times.append(time.time() - t0)
    names.append("prompt-p2n4")
    runtimes.append(np.mean(times))
    runtimes_wo.append(np.mean(times[1:]))
    """

    # from bounding box
    times = []
    for _ in range(n):
        t0 = time.time()
        embeddings = util.precompute_image_embeddings(predictor, image)
        box_size = np.random.randint(20, 100, size=2)
        box_start = [
            np.random.randint(0, image.shape[0] - box_size[0]),
            np.random.randint(0, image.shape[1] - box_size[1]),
        ]
        box = np.array([
            box_start[0], box_start[1],
            box_start[0] + box_size[0], box_start[1] + box_size[1],
        ])
        seg.segment_from_box(predictor, box, embeddings)
        times.append(time.time() - t0)
    names.append("prompt-box")
    runtimes.append(np.mean(times[1:]))

    return names, runtimes
"""
    # from bounding box and points
    times = []
    for _ in range(n):
        t0 = time.time()
        points = np.concatenate([
            np.random.randint(0, image.shape[0], size=6)[:, None],
            np.random.randint(0, image.shape[1], size=6)[:, None],
        ], axis=1)
        labels = np.array([1, 1, 0, 0, 0, 0])
        box_size = np.random.randint(20, 100, size=2)
        box_start = [
            np.random.randint(0, image.shape[0] - box_size[0]),
            np.random.randint(0, image.shape[1] - box_size[1]),
        ]
        box = np.array([
            box_start[0], box_start[1],
            box_start[0] + box_size[0], box_start[1] + box_size[1],
        ])
        seg.segment_from_box_and_points(predictor, box, points, labels, embeddings)
        times.append(time.time() - t0)
    names.append("prompt-box-and-points")
    #runtimes.append(np.mean(times))
    #errors.append(np.std(times)/np.sqrt(n))
    #runtimes.append(np.min(times))
    runtimes.append(times)
"""


def benchmark_amg(image, predictor, n):
    print("Running benchmark_amg ...")
    n = 1 if n is None else n
    amg = instance_seg.AutomaticMaskGenerator(predictor)
    times = []
    for _ in range(n):
        t0 = time.time()
        embeddings = util.precompute_image_embeddings(predictor, image)
        amg.initialize(image, embeddings)
        amg.generate()
        times.append(time.time() - t0)
    runtime = np.mean(times[1:])
    return ["amg"], [runtime]

def benchmark_ais(image, segmenter, n):
    print("Running benchmark_ais")
    predictor = segmenter._predictor
    n = 1 if n is None else n
    times = []
    for _ in range(n):
        t0 = time.time()
        embeddings = util.precompute_image_embeddings(predictor, image)
        segmenter.initialize(image, embeddings)
        segmenter.generate()
        times.append(time.time() - t0)
    runtime = np.mean(times[1:])
    return ["ais"], [runtime]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d",
                        choices=['cpu', 'cuda', 'mps'],
                        help="Which PyTorch backend device to use (REQUIRED)")
    parser.add_argument("--model_type", "-m", default="vit_h",
                        choices=list(util._MODEL_TYPES),
                        help="Which deep learning model to use")
    parser.add_argument("--benchmark_embeddings", "-e", action="store_false",
                        help="Skip embedding benchmark test, do not run")
    parser.add_argument("--benchmark_prompts", "-p", action="store_false",
                        help="Skip prompt benchmark test, do not run")
    parser.add_argument("--benchmark_amg", "-a", action="store_false",
                        help="Skip automatic mask generation (amg) benchmark test, do not run")
    parser.add_argument("--benchmark_ais", "-ai", action="store_false",
                        help="Skip automatic instance segmentation (ais) benchmark test, do not run")
    
    parser.add_argument("-n", "--n", type=int, default=None,
                        help="Number of times to repeat benchmark tests")
 
    parser.add_argument("-i", "--image", help="Path to test image")
    parser.add_argument("-c", "--checkpoint", help="Checkpoint path")
    parser.add_argument("-s", "--save_path", help='Path where benchmark results will be saved')

    args = parser.parse_args()

    model_type = args.model_type
    device = util.get_device(args.device)
    checkpoint_path = args.checkpoint
    image_path = args.image


    print("Running benchmarks for", model_type)
    print("with device:", device)

    image, predictor = _get_image_and_predictor(model_type, device, checkpoint_path, image_path)

    benchmark_results = []
    if args.benchmark_embeddings:
        name, rt = benchmark_embeddings(image, predictor, args.n)
        benchmark_results = _add_result(benchmark_results, name, rt)

    if args.benchmark_prompts:
        name, rt = benchmark_prompts(image, predictor, args.n)
        benchmark_results = _add_result(benchmark_results, name, rt)
        
    if args.benchmark_amg:
        name, rt = benchmark_amg(image, predictor, args.n)
        benchmark_results = _add_result(benchmark_results, name, rt)
    
    if args.benchmark_ais:
        segmenter = instance_seg.load_instance_segmentation_with_decoder_from_checkpoint(
            args.checkpoint, args.model_type
        )
        name, rt = benchmark_ais(image, segmenter, args.n)
        benchmark_results = _add_result(benchmark_results, name, rt)

    benchmark_results = pd.concat(benchmark_results)
    print(benchmark_results.to_markdown(index=False))
    benchmark_results.to_csv(args.save_path)


if __name__ == "__main__":
    main()
