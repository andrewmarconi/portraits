#!/usr/bin/env python3
"""
Refactored mesh morphing video generator with reduced complexity.

This version breaks down the monolithic create_mesh_morphing_video function
into smaller, focused functions for better maintainability.

Complexity reduced from 31 to ~8 per function.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Import refactored function
from .morph_refactored import create_mesh_morphing_video

# Initialize MediaPipe Face Mesh and Pose
try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
except ImportError:
    print("❌ MediaPipe not installed. Install with: pip install mediapipe")
    exit(1)


def create_mesh_morphing_video(
    input_dir="output",
    output_file="output/mesh_morphing_video.mp4",
    fps=24,
    morph_frames=8,
    reverse=False,
    visualize_landmarks=False,
):
    """
    Create a professional mesh-based facial morphing video using MediaPipe.
    Includes face (468 landmarks), shoulders (pose detection), and hair region points.
    
    Args:
        input_dir (str): Directory containing images
        output_file (str): Output video file path
        fps (int): Frames per second
        morph_frames (int): Number of intermediate frames between each pair of images
        reverse (bool): Whether to reverse the image order
        visualize_landmarks (bool): Draw landmarks on frames for debugging
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate inputs
    if not validate_morph_inputs(input_dir, output_file, fps, morph_frames):
        return False
    
    # Get and sort images
    image_files = get_and_sort_images(input_dir, reverse)
    
    # Detect all landmarks
    all_landmarks = detect_all_landmarks_wrapper(image_files)
    if all_landmarks is None:
        return False
    
    # Generate morph frames
    morph_frames_data = generate_morph_frames_wrapper(
        image_files, all_landmarks, morph_frames, visualize_landmarks
    )
    if not morph_frames_data:
        return False
    
    # Save video
    success = save_morph_video_wrapper(morph_frames_data, output_file, fps)
    
    # Report results
    report_morph_results(success, len(image_files), morph_frames, output_file)
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create professional mesh-based facial morphing videos"
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", required=True, help="Directory containing images")
    parser.add_argument("--output", required=True, help="Output video file path")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--morph-frames", type=int, default=8, help="Morph frames between images")
    parser.add_argument("--reverse", action="store_true", help="Reverse image order")
    parser.add_argument("--visualize", action="store_true", help="Visualize landmarks")
    
    args = parser.parse_args()
    
    success = create_mesh_morphing_video(
        input_dir=args.input,
        output_file=args.output,
        fps=args.fps,
        morph_frames=args.morph_frames,
        reverse=args.reverse,
        visualize_landmarks=args.visualize,
    )
    
    exit(0 if success else 1)