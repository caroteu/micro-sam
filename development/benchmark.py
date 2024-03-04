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
from PIL import Image
import os
import shutil

import imageio.v3 as imageio
import micro_sam.instance_segmentation as instance_seg
from micro_sam.evaluation.instance_segmentation import run_instance_segmentation_inference
import micro_sam.prompt_based_segmentation as seg
import micro_sam.util as util
import numpy as np
import pandas as pd

from micro_sam.sample_data import fetch_livecell_example_data

def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): # Uncomment this if you want to delete subdirectories as well
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def convert_tif_to_png(tif_path, png_path):
    try:
        # Open the .tif file
        with Image.open(tif_path) as img:
            # Save as .png
            img.save(png_path, format="PNG")
    except Exception as e:
        print(f"Error converting {tif_path} to PNG: {e}")

def _get_image_and_predictor(model_type, device, checkpoint_path,image_path):
    #example_data = fetch_livecell_example_data("../examples/data")
    images = []
    image_paths = []
    for i, filename in enumerate(os.listdir(image_path)):
        filepath = os.path.join(image_path, filename)
        #os.rename(filepath, f"example_image_{i}.png")
        convert_tif_to_png(filepath, f"example_image_{i}.png")
        try:
            img = imageio.imread(filepath)
            images.append(img)
            image_paths.append(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    if checkpoint_path.endswith('.pt'):
        predictor = util.get_custom_sam_model(checkpoint_path, model_type, device)
    else:
        predictor = util.get_sam_model(model_type, device, checkpoint_path)
    
    return images, image_paths, predictor


def _add_result(benchmark_results, name, runtimes, error):
    nres = len(name)
    assert len(name) == len(runtimes)
    res = {
        "benchmark": name,
        "runtimes": runtimes, 
        "error": error
    }
    tab = pd.DataFrame(res)
    benchmark_results.append(tab)
    return benchmark_results


def benchmark_embeddings(images, predictor, n):
    print("Running benchmark_embeddings ...")
    n = 3 if n is None else n
    times = []
    for image in images:
        for _ in range(n):
            t0 = time.time()
            util.precompute_image_embeddings(predictor, image)
            times.append(time.time() - t0)
    runtime = np.mean(times[1:])
    error = np.std(times[1:])


    return ["embeddings"], [runtime], [error]


def benchmark_prompts(images, predictor, n):
    print("Running benchmark_prompts ...")
    n = 10 if n is None else n

    np.random.seed(42)

    names, runtimes, errors = [], [], []

    # from random single point
    times = []
    for image in images:
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
    errors.append(np.std(times[1:]))

   # from bounding box
    times = []
    for image in images:
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
    errors.append(np.std(times[1:]))

    return names, runtimes, errors

def benchmark_amg(image_paths, predictor, n):
    print("Running benchmark amg ...")
    
    amg = instance_seg.AutomaticMaskGenerator(predictor)
    prediction_dir = "/scratch/usr/nimcarot/sam/experiments/benchmark/predictions/"
    embedding_dir = "/scratch/usr/nimcarot/sam/experiments/benchmark/embeddings/"
    times = []
    for _ in range(n):
        t0 = time.time()

        run_instance_segmentation_inference(
            amg, image_paths, embedding_dir, prediction_dir)
        t1 = time.time()
        times.append((t1-t0)/len(image_paths))
        delete_files_in_directory(embedding_dir)
        delete_files_in_directory(prediction_dir)
        
    runtime = np.mean(times)
    error = np.std(times)
    return ["amg"], [runtime], [error]

def benchmark_ais(image_paths, segmenter, n):
    print("Running benchmark_ais")
    prediction_dir = "/scratch/usr/nimcarot/sam/experiments/benchmark/predictions/"
    embedding_dir = "/scratch/usr/nimcarot/sam/experiments/benchmark/embeddings/"
    times = []
    n = 1 if n is None else n
    for _ in range(n):
        t0 = time.time()

        run_instance_segmentation_inference(
            segmenter, image_paths, embedding_dir, prediction_dir)

        t1 = time.time()
        times.append((t1-t0)/len(image_paths))
        delete_files_in_directory(embedding_dir)
        delete_files_in_directory(prediction_dir)
        
    runtime = np.mean(times)
    error = np.std(times)
    return ["amg"], [runtime], [error]


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
 
    parser.add_argument("-i", "--image", help="Path to test images")
    parser.add_argument("-c", "--checkpoint", help="Checkpoint path")
    parser.add_argument("-s", "--save_path", help='Path where benchmark results will be saved')

    args = parser.parse_args()

    model_type = args.model_type
    device = util.get_device(args.device)
    checkpoint_path = args.checkpoint
    image_path = args.image


    print("Running benchmarks for", model_type)
    print("with device:", device)

    images, image_paths, predictor = _get_image_and_predictor(model_type, device, checkpoint_path, image_path)

    benchmark_results = []
    if args.benchmark_embeddings:
        name, rt, err = benchmark_embeddings(images, predictor, args.n)
        benchmark_results = _add_result(benchmark_results, name, rt, err)

    if args.benchmark_prompts:
        name, rt, err = benchmark_prompts(images, predictor, args.n)
        benchmark_results = _add_result(benchmark_results, name, rt, err)
        
    if args.benchmark_amg:
        name, rt, err = benchmark_amg(image_paths, predictor, args.n)
        benchmark_results = _add_result(benchmark_results, name, rt, err)
    
    if args.benchmark_ais:
        segmenter = instance_seg.load_instance_segmentation_with_decoder_from_checkpoint(
            args.checkpoint, args.model_type, is_custom_checkpoint=False
        )
        name, rt, err = benchmark_ais(image_paths, segmenter, args.n)
        benchmark_results = _add_result(benchmark_results, name, rt, err)

    benchmark_results = pd.concat(benchmark_results)
    print(benchmark_results.to_markdown(index=False))
    benchmark_results.to_csv(args.save_path)


if __name__ == "__main__":
    main()
