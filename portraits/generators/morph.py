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

# Import helper functions
from .morph_helpers import (
    validate_morph_inputs,
    get_and_sort_images,
    detect_all_landmarks_wrapper,
    generate_morph_frames_wrapper,
    save_morph_video_wrapper,
    report_morph_results,
)

# Initialize MediaPipe Face Mesh and Pose
try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
except ImportError:
    print("❌ MediaPipe not installed. Install with: pip install mediapipe")
    exit(1)


def detect_face_landmarks(image_path):
    """Detect facial landmarks using MediaPipe Face Mesh."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            landmarks = []
            h, w = image.shape[:2]
            for landmark in results.multi_face_landmarks[0].landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            return np.array(landmarks)
    return None


def detect_pose_landmarks(image_path):
    """Detect pose landmarks using MediaPipe Pose."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True, model_complexity=1, min_detection_confidence=0.5
    ) as pose:
        results = pose.process(rgb_image)

        if results.pose_landmarks:
            landmarks = []
            h, w = image.shape[:2]
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            return np.array(landmarks)
    return None


def add_hair_region_points(face_landmarks, height, width):
    """Add strategic points for hair region coverage."""
    hair_points = []

    # Top hairline points
    for i in range(10, 20):
        x = int(face_landmarks[i][0] + (i - 15) * 10)
        y = max(0, int(face_landmarks[i][1] - 50))
        hair_points.append([x, y])

    # Side hair points
    for i in range(5):
        # Left side
        x = max(0, int(face_landmarks[0][0] - 30 - i * 15))
        y = int(face_landmarks[0][1] + i * 20)
        hair_points.append([x, y])

        # Right side
        x = min(width, int(face_landmarks[16][0] + 30 + i * 15))
        y = int(face_landmarks[16][1] + i * 20)
        hair_points.append([x, y])

    return np.array(hair_points)


def add_image_corners(height, width):
    """Add image corner points to ensure full coverage."""
    corners = [
        [0, 0],  # Top-left
        [width - 1, 0],  # Top-right
        [0, height - 1],  # Bottom-left
        [width - 1, height - 1],  # Bottom-right
    ]
    return np.array(corners)


def combine_all_landmarks(face_landmarks, pose_landmarks, hair_points, corner_points=None):
    """Combine all landmark types into single array."""
    all_points = []

    if face_landmarks is not None:
        all_points.extend(face_landmarks.tolist())

    if pose_landmarks is not None:
        all_points.extend(pose_landmarks.tolist())

    if hair_points is not None:
        all_points.extend(hair_points.tolist())

    if corner_points is not None:
        all_points.extend(corner_points.tolist())

    return np.array(all_points)


def create_delaunay_triangulation(points, height, width):
    """Create Delaunay triangulation for mesh warping."""
    try:
        from scipy.spatial import Delaunay

        return Delaunay(points)
    except ImportError:
        raise ImportError("scipy is required for morphing. Install with: uv sync --extra morph")


def warp_triangle(img, src_tri, dst_tri):
    """Warp a single triangle from source to destination using affine transform."""
    # Get bounding rectangles for both triangles
    src_rect = cv2.boundingRect(np.array([src_tri], dtype=np.float32))
    dst_rect = cv2.boundingRect(np.array([dst_tri], dtype=np.float32))

    # Offset points by left top corner of bounding rectangle
    src_tri_offset = []
    dst_tri_offset = []

    for i in range(3):
        src_tri_offset.append(((src_tri[i][0] - src_rect[0]), (src_tri[i][1] - src_rect[1])))
        dst_tri_offset.append(((dst_tri[i][0] - dst_rect[0]), (dst_tri[i][1] - dst_rect[1])))

    # Create mask for destination triangle
    mask = np.zeros((dst_rect[3], dst_rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_offset), (1.0, 1.0, 1.0), cv2.LINE_AA)

    # Extract source rectangle from image
    src_rect_img = img[src_rect[1]:src_rect[1] + src_rect[3],
                       src_rect[0]:src_rect[0] + src_rect[2]]

    # If source rectangle is empty, return None
    if src_rect_img.size == 0:
        return None, dst_rect

    # Get affine transform from source to destination triangle
    warp_mat = cv2.getAffineTransform(np.float32(src_tri_offset), np.float32(dst_tri_offset))

    # Apply affine transform to source rectangle
    dst_rect_img = cv2.warpAffine(
        src_rect_img,
        warp_mat,
        (dst_rect[2], dst_rect[3]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Apply mask to get only triangle region
    dst_rect_img = dst_rect_img * mask

    return dst_rect_img, dst_rect


def create_morphed_frame(img1, img2, landmarks1, landmarks2, tri, alpha, height, width):
    """Create a morphed frame using proper triangle-based mesh morphing."""
    # Calculate interpolated landmarks for the morphed position
    interpolated_landmarks = (1.0 - alpha) * landmarks1 + alpha * landmarks2

    # Start with a blended base for areas not covered by triangles
    beta = 1.0 - alpha
    morphed = cv2.addWeighted(img1, beta, img2, alpha, 0).astype(np.float32)

    # Get triangles from the Delaunay triangulation
    triangles = tri.simplices

    # Process each triangle
    for triangle_indices in triangles:
        # Get the three vertex coordinates for this triangle
        src_tri = landmarks1[triangle_indices].astype(np.float32)
        tgt_tri = landmarks2[triangle_indices].astype(np.float32)
        int_tri = interpolated_landmarks[triangle_indices].astype(np.float32)

        # Warp triangle from source image to interpolated position
        warped1, dst_rect1 = warp_triangle(img1, src_tri, int_tri)

        # Warp triangle from target image to interpolated position
        warped2, dst_rect2 = warp_triangle(img2, tgt_tri, int_tri)

        # Skip if warping failed
        if warped1 is None or warped2 is None:
            continue

        # Blend the two warped triangles
        blended = (1.0 - alpha) * warped1 + alpha * warped2

        # Add blended triangle to the output image at the correct position
        y1, y2 = dst_rect1[1], dst_rect1[1] + dst_rect1[3]
        x1, x2 = dst_rect1[0], dst_rect1[0] + dst_rect1[2]

        # Ensure coordinates are within bounds
        y1_orig, y2_orig = y1, y2
        x1_orig, x2_orig = x1, x2

        y1 = max(0, y1)
        y2 = min(height, y2)
        x1 = max(0, x1)
        x2 = min(width, x2)

        # Adjust blended region if coordinates were clamped
        by1 = y1 - y1_orig
        by2 = dst_rect1[3] - (y2_orig - y2)
        bx1 = x1 - x1_orig
        bx2 = dst_rect1[2] - (x2_orig - x2)

        # Skip if region is empty after clamping
        if y2 <= y1 or x2 <= x1 or by2 <= by1 or bx2 <= bx1:
            continue

        # Extract regions
        region = morphed[y1:y2, x1:x2]
        blended_region = blended[by1:by2, bx1:bx2]

        # Ensure shapes match
        if region.shape != blended_region.shape:
            continue

        # Add blended triangle using mask-based composition
        mask = (blended_region.sum(axis=2) > 0).astype(np.float32)
        if mask.sum() > 0:
            mask_3d = np.stack([mask, mask, mask], axis=2)
            morphed[y1:y2, x1:x2] = region * (1 - mask_3d) + blended_region

    return morphed.astype(np.uint8)


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
        description="Create professional mesh-based facial morphing videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
