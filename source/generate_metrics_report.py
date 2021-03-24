import os
from pathlib import Path
import modules.metrics as metrics
from modules.config import execute

def main(args):
    test_experiment_path = Path(args.test_results_path) / args.experiment_name 

    available_metrics = [
        metrics.chamfer_distance,
        metrics.absolute_mean_surface_distance,
        metrics.symmetric_absolute_mean_surface_distance,
        metrics.hausdorff,
        metrics.symmetric_hausdorff,
        metrics.signed_mean_point_to_surface,
        metrics.abs_mean_point_to_surface,
        metrics.rms_point_to_surface
    ]

    available_tf_metrics = [
        metrics.chamfer_distance, 
        metrics.dice
    ]

    

if __name__ == "__main__":
    main(execute())