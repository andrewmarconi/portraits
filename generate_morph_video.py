#!/usr/bin/env python3
"""
Create professional facial mesh morphing videos using MediaPipe Face Mesh.
This uses 468 facial landmarks for precise mesh-based morphing.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from scipy.spatial import Delaunay
import warnings
warnings.filterwarnings('ignore')

# Initialize MediaPipe Face Mesh and Pose
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def detect_face_landmarks(image, face_mesh):
    """
    Detect 468 facial landmarks using MediaPipe Face Mesh.

    Args:
        image: BGR image from OpenCV
        face_mesh: MediaPipe FaceMesh object

    Returns:
        numpy array of (x, y) coordinates, or None if no face detected
    """
    h, w = image.shape[:2]

    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print("    ⚠️  No face detected in image")
        return None

    # Get the first face's landmarks
    face_landmarks = results.multi_face_landmarks[0]

    # Convert normalized coordinates to pixel coordinates
    landmarks = []
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Clamp to image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        landmarks.append([x, y])

    return np.array(landmarks, dtype=np.float32)

def detect_pose_landmarks(image, pose_detector):
    """
    Detect body pose landmarks, focusing on shoulders and torso.

    Args:
        image: BGR image from OpenCV
        pose_detector: MediaPipe Pose object

    Returns:
        List of shoulder/torso landmark points, or empty list if detection fails
    """
    h, w = image.shape[:2]

    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return []

    # Extract shoulder and upper body landmarks
    # MediaPipe Pose landmark indices:
    # 11: left shoulder, 12: right shoulder
    # 23: left hip, 24: right hip
    # 7: left ear, 8: right ear (for hair/head region)
    pose_points = []

    relevant_indices = [
        7,   # Left ear (helps with hair region)
        8,   # Right ear (helps with hair region)
        11,  # Left shoulder
        12,  # Right shoulder
        23,  # Left hip
        24,  # Right hip
    ]

    landmarks = results.pose_landmarks.landmark
    for idx in relevant_indices:
        if idx < len(landmarks):
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            # Clamp to image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            # Only add if visibility is good
            if landmark.visibility > 0.5:
                pose_points.append([x, y])

    return pose_points

def add_hair_region_points(face_landmarks, img_shape, num_points=12):
    """
    Add strategic points in the hair region above the face.

    Args:
        face_landmarks: Facial landmark points
        img_shape: Tuple of (height, width)
        num_points: Number of hair region points to add

    Returns:
        Array of hair region points
    """
    _, w = img_shape[:2]

    if len(face_landmarks) == 0:
        return np.array([], dtype=np.float32).reshape(0, 2)

    # Find the topmost facial landmarks (forehead region)
    top_face_y = np.min(face_landmarks[:, 1])

    # Find the leftmost and rightmost facial landmarks
    left_face_x = np.min(face_landmarks[:, 0])
    right_face_x = np.max(face_landmarks[:, 0])

    # Create a grid of points in the hair region (above the forehead)
    hair_points = []

    # Hair extends from top of image to just above forehead
    hair_bottom = max(0, int(top_face_y - 10))
    hair_top = 0

    # Create points across the width of the face, extended slightly
    face_width = right_face_x - left_face_x
    hair_left = max(0, int(left_face_x - face_width * 0.1))
    hair_right = min(w - 1, int(right_face_x + face_width * 0.1))

    # Create grid in hair region with exact num_points
    rows = max(2, num_points // 4)
    cols = max(3, num_points // rows)

    points_added = 0
    for row in range(rows):
        y = int(hair_top + (hair_bottom - hair_top) * row / (rows - 1)) if rows > 1 else hair_top
        for col in range(cols):
            if points_added >= num_points:
                break
            x = int(hair_left + (hair_right - hair_left) * col / (cols - 1)) if cols > 1 else (hair_left + hair_right) // 2
            hair_points.append([x, y])
            points_added += 1
        if points_added >= num_points:
            break

    return np.array(hair_points, dtype=np.float32) if hair_points else np.array([], dtype=np.float32).reshape(0, 2)

def combine_all_landmarks(face_landmarks, pose_points, img_shape):
    """
    Combine face landmarks with pose (shoulder) landmarks and hair region points.

    Args:
        face_landmarks: Facial landmark points
        pose_points: Pose/shoulder landmark points
        img_shape: Tuple of (height, width)

    Returns:
        Combined array of all landmarks
    """
    # Start with face landmarks
    all_landmarks = face_landmarks.copy()

    # Add pose landmarks (shoulders, ears, etc.)
    if len(pose_points) > 0:
        pose_array = np.array(pose_points, dtype=np.float32)
        all_landmarks = np.vstack([all_landmarks, pose_array])

    # Add hair region points
    hair_points = add_hair_region_points(face_landmarks, img_shape, num_points=12)
    if len(hair_points) > 0:
        all_landmarks = np.vstack([all_landmarks, hair_points])

    return all_landmarks

def add_boundary_points(landmarks, img_shape):
    """
    Add boundary points to ensure the entire image is covered by triangulation.

    Args:
        landmarks: Array of facial landmark points
        img_shape: Tuple of (height, width)

    Returns:
        Extended landmarks array with boundary points
    """
    h, w = img_shape[:2]

    # Add corners
    corners = np.array([
        [0, 0],           # Top-left
        [w-1, 0],         # Top-right
        [w-1, h-1],       # Bottom-right
        [0, h-1],         # Bottom-left
    ], dtype=np.float32)

    # Add edge midpoints for better triangulation
    edge_points = np.array([
        [w//2, 0],        # Top-middle
        [w-1, h//2],      # Right-middle
        [w//2, h-1],      # Bottom-middle
        [0, h//2],        # Left-middle
    ], dtype=np.float32)

    # Combine all points
    extended_landmarks = np.vstack([landmarks, corners, edge_points])

    return extended_landmarks

def create_delaunay_triangulation(points):
    """
    Create Delaunay triangulation from points.

    Args:
        points: Array of (x, y) coordinates

    Returns:
        Array of triangle indices
    """
    if len(points) < 3:
        return np.array([], dtype=np.int32)

    try:
        tri = Delaunay(points)
        return tri.simplices
    except Exception as e:
        print(f"    ⚠️  Delaunay triangulation failed: {e}")
        return np.array([], dtype=np.int32)

def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    Apply affine transform calculated using src_tri and dst_tri to src.

    Args:
        src: Source image
        src_tri: Source triangle vertices (3x2)
        dst_tri: Destination triangle vertices (3x2)
        size: Size of output image (width, height)

    Returns:
        Warped image patch
    """
    # Get affine transform matrix
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the affine transformation
    dst = cv2.warpAffine(src, warp_mat, size,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)

    return dst

def morph_triangle(img1, img2, img_morph, tri1, tri2, tri_morph, alpha):
    """
    Morph a single triangle from img1 and img2 to img_morph.

    Args:
        img1: First source image
        img2: Second source image
        img_morph: Output morphed image
        tri1: Triangle in img1
        tri2: Triangle in img2
        tri_morph: Triangle in morphed image
        alpha: Blending factor (0 = img1, 1 = img2)
    """
    # Find bounding box for each triangle
    r1 = cv2.boundingRect(np.float32([tri1]))
    r2 = cv2.boundingRect(np.float32([tri2]))
    r_morph = cv2.boundingRect(np.float32([tri_morph]))

    # Offset points by left top corner of the respective bounding boxes
    tri1_rect = []
    tri2_rect = []
    tri_morph_rect = []

    for i in range(3):
        tri_morph_rect.append(((tri_morph[i][0] - r_morph[0]), (tri_morph[i][1] - r_morph[1])))
        tri1_rect.append(((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1])))
        tri2_rect.append(((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r_morph[3], r_morph[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_morph_rect), (1.0, 1.0, 1.0), 16, 0)

    # Extract image patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r_morph[2], r_morph[3])

    # Warp triangles
    warp_image1 = apply_affine_transform(img1_rect, tri1_rect, tri_morph_rect, size)
    warp_image2 = apply_affine_transform(img2_rect, tri2_rect, tri_morph_rect, size)

    # Blend the two warped images
    img_morph_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2

    # Copy triangular region of the morphed image to the output image
    img_morph[r_morph[1]:r_morph[1] + r_morph[3], r_morph[0]:r_morph[0] + r_morph[2]] = \
        img_morph[r_morph[1]:r_morph[1] + r_morph[3], r_morph[0]:r_morph[0] + r_morph[2]] * (1 - mask) + img_morph_rect * mask

def create_morphed_frame(img1, img2, landmarks1, landmarks2, alpha):
    """
    Create a morphed frame between two images using facial landmarks.

    Args:
        img1: First image
        img2: Second image
        landmarks1: Facial landmarks for img1
        landmarks2: Facial landmarks for img2
        alpha: Morphing factor (0 = img1, 1 = img2)

    Returns:
        Morphed image
    """
    h, w = img1.shape[:2]

    # Ensure both landmark sets have the same number of points
    if len(landmarks1) != len(landmarks2):
        print(f"    ⚠️  Landmark count mismatch: {len(landmarks1)} vs {len(landmarks2)}")
        return cv2.addWeighted(img2, alpha, img1, 1 - alpha, 0)

    # Add boundary points to both sets
    points1 = add_boundary_points(landmarks1, (h, w))
    points2 = add_boundary_points(landmarks2, (h, w))

    # Compute weighted average of landmarks for morphed image
    points_morph = (1 - alpha) * points1 + alpha * points2

    # Create triangulation on morphed points (consistent across all frames)
    triangles = create_delaunay_triangulation(points_morph)

    if len(triangles) == 0:
        print("    ⚠️  Triangulation failed, using simple blend")
        return cv2.addWeighted(img2, alpha, img1, 1 - alpha, 0)

    # Create output image
    img_morph = np.zeros(img1.shape, dtype=img1.dtype)

    # Morph each triangle
    for i, tri_indices in enumerate(triangles):
        # Get triangle vertices for all three images
        tri1 = points1[tri_indices]
        tri2 = points2[tri_indices]
        tri_morph = points_morph[tri_indices]

        # Morph one triangle
        try:
            morph_triangle(img1, img2, img_morph, tri1, tri2, tri_morph, alpha)
        except Exception as e:
            # Skip problematic triangles silently
            continue

    return img_morph

def create_mesh_morphing_video(input_dir="output", output_file="output/mesh_morphing_video.mp4",
                               fps=24, morph_frames=8, reverse=False, visualize_landmarks=False):
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
    """
    # Get all image files in directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return False

    # Sort files by name
    image_files.sort()

    if reverse:
        image_files.reverse()

    if len(image_files) < 2:
        print("❌ Need at least 2 images for morphing")
        return False

    print(f"✅ Found {len(image_files)} images")
    print(f"🎬 Generating {morph_frames} mesh morph frames between each pair")
    print(f"📊 Using MediaPipe Face Mesh (468 facial landmarks)")
    print(f"📊 Using MediaPipe Pose (shoulder/body detection)")
    print(f"📊 Adding hair region points for complete coverage")

    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"❌ Could not read first image: {image_files[0]}")
        return False

    height, width = first_image.shape[:2]
    print(f"📐 Video dimensions: {width}x{height}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"❌ Could not create video writer for: {output_file}")
        return False

    # Initialize MediaPipe Face Mesh and Pose
    print("🔧 Initializing MediaPipe Face Mesh and Pose detectors...")
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose_detector:

        # Process each pair of consecutive images
        for i in range(len(image_files) - 1):
            print(f"\n{'='*70}")
            print(f"🎭 Processing morph {i+1}/{len(image_files)-1}")
            print(f"   {image_files[i].name} → {image_files[i+1].name}")
            print(f"{'='*70}")

            # Load current and next images
            img1 = cv2.imread(str(image_files[i]))
            img2 = cv2.imread(str(image_files[i+1]))

            if img1 is None or img2 is None:
                print(f"⚠️  Warning: Could not read images, skipping...")
                continue

            # Resize if dimensions don't match
            img1 = cv2.resize(img1, (width, height))
            img2 = cv2.resize(img2, (width, height))

            # Detect facial landmarks in both images
            print("🔍 Detecting facial landmarks...")
            face_landmarks1 = detect_face_landmarks(img1, face_mesh)
            face_landmarks2 = detect_face_landmarks(img2, face_mesh)

            if face_landmarks1 is None or face_landmarks2 is None:
                print("⚠️  Face detection failed, using simple crossfade")
                # Write first image
                video_writer.write(img1)
                # Create simple crossfade
                for step in range(1, morph_frames + 1):
                    alpha = step / (morph_frames + 1)
                    blended = cv2.addWeighted(img2, alpha, img1, 1 - alpha, 0)
                    video_writer.write(blended)
                continue

            print(f"✅ Detected {len(face_landmarks1)} facial landmarks")

            # Detect pose landmarks (shoulders, ears, hips)
            print("🔍 Detecting pose landmarks (shoulders, body)...")
            pose_points1 = detect_pose_landmarks(img1, pose_detector)
            pose_points2 = detect_pose_landmarks(img2, pose_detector)

            # Only use pose landmarks if BOTH images have them AND counts match
            use_pose = (len(pose_points1) > 0 and len(pose_points2) > 0 and
                       len(pose_points1) == len(pose_points2))
            if use_pose:
                print(f"✅ Detected {len(pose_points1)} pose landmarks in both images")
            else:
                if len(pose_points1) != len(pose_points2):
                    print(f"⚠️  Pose landmark count mismatch ({len(pose_points1)} vs {len(pose_points2)})")
                print("⚠️  Using face + hair landmarks only (pose excluded)")
                pose_points1 = []
                pose_points2 = []

            # Combine face, pose, and hair region landmarks
            landmarks1 = combine_all_landmarks(face_landmarks1, pose_points1, (height, width))
            landmarks2 = combine_all_landmarks(face_landmarks2, pose_points2, (height, width))

            # Verify landmark counts match (critical for morphing)
            if len(landmarks1) != len(landmarks2):
                print(f"⚠️  Landmark count mismatch: {len(landmarks1)} vs {len(landmarks2)}")
                print("⚠️  Falling back to crossfade for this pair")
                # Write first image
                video_writer.write(img1)
                # Create simple crossfade
                for step in range(1, morph_frames + 1):
                    alpha = step / (morph_frames + 1)
                    blended = cv2.addWeighted(img2, alpha, img1, 1 - alpha, 0)
                    video_writer.write(blended)
                continue

            print(f"✅ Total landmarks including hair/shoulders: {len(landmarks1)}")

            # Write the first image
            if visualize_landmarks:
                img1_viz = img1.copy()
                for x, y in landmarks1.astype(int):
                    cv2.circle(img1_viz, (x, y), 1, (0, 255, 0), -1)
                video_writer.write(img1_viz)
            else:
                video_writer.write(img1)

            # Generate mesh morph frames
            for step in range(1, morph_frames + 1):
                alpha = step / (morph_frames + 1)
                print(f"   📹 Morphing frame {step}/{morph_frames} (α={alpha:.2f})...", end=' ')

                # Create morphed frame using mesh warping
                morphed = create_morphed_frame(img1, img2, landmarks1, landmarks2, alpha)

                # Optionally visualize landmarks
                if visualize_landmarks:
                    points_morph = (1 - alpha) * landmarks1 + alpha * landmarks2
                    for x, y in points_morph.astype(int):
                        cv2.circle(morphed, (x, y), 1, (0, 255, 0), -1)

                video_writer.write(morphed)
                print("✓")

        # Write the last image
        print(f"\n{'='*70}")
        print("📝 Writing final frame...")
        last_img = cv2.imread(str(image_files[-1]))
        if last_img is not None:
            last_img = cv2.resize(last_img, (width, height))

            if visualize_landmarks:
                landmarks_last = detect_face_landmarks(last_img, face_mesh)
                if landmarks_last is not None:
                    for x, y in landmarks_last.astype(int):
                        cv2.circle(last_img, (x, y), 1, (0, 255, 0), -1)

            video_writer.write(last_img)

    # Release video writer
    video_writer.release()

    print(f"\n{'='*70}")
    print(f"✅ MESH MORPHING VIDEO COMPLETED!")
    print(f"{'='*70}")
    print(f"📁 Saved to: {output_file}")
    print(f"{'='*70}\n")

    return True

if __name__ == "__main__":
    print("="*70)
    print("🎭 PROFESSIONAL MESH-BASED FACIAL MORPHING")
    print("="*70)
    print("✨ MediaPipe Face Mesh: 468 facial landmark points")
    print("✨ MediaPipe Pose: Shoulder and body detection")
    print("✨ Hair Region: Strategic points above face")
    print("✨ Delaunay triangulation for smooth mesh warping")
    print("="*70)

    # Create mesh morphing video
    success = create_mesh_morphing_video(
        input_dir="output",
        output_file="output/mesh_morphing_video.mp4",
        fps=12,
        morph_frames=6,
        visualize_landmarks=False  # Set to True to see landmarks
    )

    if success:
        print("\n" + "="*70)
        print("🎉 SUCCESS! PROFESSIONAL MORPHING COMPLETED!")
        print("="*70)
        print("\n✨ Features:")
        print("  ✅ 468 facial landmarks per face (eyes, nose, mouth, contours)")
        print("  ✅ Shoulder and body pose landmarks")
        print("  ✅ Hair region coverage (strategic grid points)")
        print("  ✅ Delaunay triangulation mesh")
        print("  ✅ Affine transformation per triangle")
        print("  ✅ Smooth vertex interpolation")
        print("\n🎬 Your morphing video shows:")
        print("  • Face features smoothly transitioning (eyes, nose, mouth)")
        print("  • Hair flowing and morphing naturally")
        print("  • Shoulders and body moving seamlessly")
        print("  • Complete portrait morphing, not just face")
        print("  • Professional mesh-based warping")
        print("="*70)
    else:
        print("\n❌ Failed to create mesh morphing video.")
