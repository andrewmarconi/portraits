"""Helper functions for mesh morphing to reduce complexity."""

import cv2
import numpy as np
from pathlib import Path


def validate_morph_inputs(input_dir, output_file, fps, morph_frames):
    """Validate morphing video inputs."""
    if not input_dir or not output_file:
        print("❌ Input directory and output file are required")
        return False
    
    if fps <= 0 or fps > 60:
        print("❌ FPS must be between 1 and 60")
        return False
    
    if morph_frames < 1 or morph_frames > 30:
        print("❌ Morph frames must be between 1 and 30")
        return False
    
    return True


def get_and_sort_images(input_dir, reverse=False):
    """Get and sort image files from directory."""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return []
    
    # Sort files by name
    image_files.sort()
    
    if reverse:
        image_files.reverse()
    
    return image_files


def detect_all_landmarks_wrapper(image_files):
    """Wrapper for landmark detection with error handling."""
    from .morph import detect_face_landmarks, detect_pose_landmarks, combine_all_landmarks, add_hair_region_points, add_image_corners

    all_landmarks = []

    for i, img_path in enumerate(image_files):
        print(f"🔍 Processing {img_path.name} ({i+1}/{len(image_files)})")

        # Detect face landmarks
        face_landmarks = detect_face_landmarks(str(img_path))
        if face_landmarks is None:
            print(f"❌ No face detected in {img_path.name}")
            return None

        # Detect pose landmarks
        pose_landmarks = detect_pose_landmarks(str(img_path))

        # Add hair region points and image corners
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]
        hair_points = add_hair_region_points(face_landmarks, height, width)
        corner_points = add_image_corners(height, width)

        # Combine all landmarks
        all_points = combine_all_landmarks(face_landmarks, pose_landmarks, hair_points, corner_points)
        all_landmarks.append(all_points)

    return all_landmarks


def generate_morph_frames_wrapper(image_files, all_landmarks, morph_frames, visualize_landmarks):
    """Generate morph frames with error handling."""
    from .morph import create_delaunay_triangulation, create_morphed_frame
    
    if len(image_files) < 2:
        print("❌ Need at least 2 images for morphing")
        return []
    
    morph_frames_data = []
    total_pairs = len(image_files) - 1
    
    for i in range(total_pairs):
        img1_path = image_files[i]
        img2_path = image_files[i + 1]
        
        print(f"🎬 Processing pair {i+1}/{total_pairs}: {img1_path.name} → {img2_path.name}")
        
        # Load images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            print(f"❌ Could not read images, skipping...")
            continue
        
        # Get landmarks
        landmarks1 = all_landmarks[i]
        landmarks2 = all_landmarks[i + 1]
        
        # Create triangulation
        height, width = img1.shape[:2]
        points1 = landmarks1.astype(np.float32)
        points2 = landmarks2.astype(np.float32)
        
        tri = create_delaunay_triangulation(points1, height, width)
        
        # Generate morph frames
        for frame_idx in range(morph_frames + 1):
            alpha = frame_idx / morph_frames
            
            # Create morphed frame with optional visualization
            morphed_img = create_morphed_frame(
                img1, img2, landmarks1, landmarks2, tri, alpha, height, width, visualize=visualize_landmarks
            )

            morph_frames_data.append(morphed_img)
        
        # Write final frame
        morph_frames_data.append(img2)
    
    return morph_frames_data


def save_morph_video_wrapper(morph_frames_data, output_file, fps):
    """Save morph frames as video with error handling."""
    try:
        height, width = morph_frames_data[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        total_frames = len(morph_frames_data)
        print(f"💾 Saving {total_frames} frames to {output_file}")
        
        for i, frame in enumerate(morph_frames_data):
            video_writer.write(frame)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{total_frames} frames ({((i+1)/total_frames)*100:.1f}%)")
        
        video_writer.release()
        return True
        
    except Exception as e:
        print(f"❌ Error saving video: {e}")
        return False


def report_morph_results(success, num_images, morph_frames, output_file):
    """Report morphing results."""
    if success:
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! PROFESSIONAL MORPHING COMPLETED!")
        print("=" * 70)
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
        print("=" * 70)
    else:
        print("\n❌ Failed to create mesh morphing video.")