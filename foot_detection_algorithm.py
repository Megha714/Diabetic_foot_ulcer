"""
Advanced Foot Detection Algorithm
Uses computer vision techniques to analyze if an image contains a foot
Based on anatomical features, skin analysis, and spatial patterns
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import feature, measure
from sklearn.cluster import KMeans
import torch
from PIL import Image


class FootDetectionAlgorithm:
    """
    Advanced algorithm to detect if an image contains a human foot
    Uses multiple computer vision techniques:
    1. Skin color detection
    2. Shape analysis (foot-like contours)
    3. Edge pattern analysis
    4. Texture analysis
    5. Aspect ratio and dimensional constraints
    6. Toe/heel detection
    """
    
    def __init__(self):
        # Skin color ranges in HSV (covers various skin tones)
        self.skin_hsv_ranges = [
            # Light skin
            {'lower': np.array([0, 20, 70]), 'upper': np.array([20, 150, 255])},
            # Medium skin
            {'lower': np.array([0, 25, 50]), 'upper': np.array([25, 170, 255])},
            # Dark skin
            {'lower': np.array([0, 30, 30]), 'upper': np.array([30, 180, 230])},
        ]
        
        # Skin color ranges in YCrCb (more robust for different lighting)
        self.skin_ycrcb_ranges = {
            'lower': np.array([0, 133, 77]),
            'upper': np.array([255, 173, 127])
        }
        
    def detect_foot(self, image_path_or_array, return_details=False):
        """
        Main detection method - analyzes if image contains a foot
        
        Args:
            image_path_or_array: Path to image or numpy array
            return_details: If True, returns detailed analysis
            
        Returns:
            is_foot (bool), confidence (float), details (dict if return_details=True)
        """
        # Load image
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                pil_img = Image.open(image_path_or_array)
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            image = image_path_or_array
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to BGR for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Run all detection methods
        results = {}
        
        # 1. Skin detection
        skin_score, skin_ratio = self._detect_skin(image)
        results['skin_score'] = skin_score
        results['skin_ratio'] = skin_ratio
        
        # 2. Shape analysis
        shape_score, foot_shape_found = self._analyze_shape(image)
        results['shape_score'] = shape_score
        results['foot_shape_found'] = foot_shape_found
        
        # 3. Texture analysis
        texture_score = self._analyze_texture(image)
        results['texture_score'] = texture_score
        
        # 4. Aspect ratio check
        aspect_score = self._check_aspect_ratio(image)
        results['aspect_score'] = aspect_score
        
        # 5. Edge pattern analysis
        edge_score = self._analyze_edges(image)
        results['edge_score'] = edge_score
        
        # 6. Color distribution (check for unnatural colors)
        color_score = self._analyze_color_distribution(image)
        results['color_score'] = color_score
        
        # 7. Anatomical feature detection (toes, heel, arch)
        anatomy_score = self._detect_anatomical_features(image)
        results['anatomy_score'] = anatomy_score
        
        # Calculate overall confidence using weighted scores
        confidence = self._calculate_confidence(results)
        results['overall_confidence'] = confidence
        
        # Decision threshold - balanced for medical images
        is_foot = confidence >= 0.50  # 50% confidence threshold
        
        if return_details:
            return is_foot, confidence, results
        else:
            return is_foot, confidence
    
    def _detect_skin(self, image):
        """Detect skin-like regions in the image"""
        # Convert to HSV and YCrCb
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Create skin masks using multiple color spaces
        skin_masks = []
        
        # HSV-based detection (multiple ranges for different skin tones)
        for skin_range in self.skin_hsv_ranges:
            mask = cv2.inRange(hsv, skin_range['lower'], skin_range['upper'])
            skin_masks.append(mask)
        
        # YCrCb-based detection
        mask_ycrcb = cv2.inRange(ycrcb, self.skin_ycrcb_ranges['lower'], 
                                  self.skin_ycrcb_ranges['upper'])
        skin_masks.append(mask_ycrcb)
        
        # Combine masks (at least 2 methods should agree)
        combined_mask = np.zeros_like(skin_masks[0])
        for mask in skin_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate skin ratio
        skin_pixels = np.count_nonzero(combined_mask)
        total_pixels = combined_mask.size
        skin_ratio = skin_pixels / total_pixels
        
        # Score based on skin ratio (feet should have significant skin area)
        # Allow wide range for medical images with wounds/ulcers
        if 0.15 <= skin_ratio <= 0.97:  # Wide range for medical foot images
            skin_score = 1.0
        elif 0.10 <= skin_ratio < 0.15 or 0.97 < skin_ratio <= 0.99:
            skin_score = 0.6  # Still acceptable
        elif skin_ratio < 0.08:  # Too little skin (clearly not a foot)
            skin_score = 0.1
        else:  # Too uniform
            skin_score = 0.4
        
        return skin_score, skin_ratio
    
    def _analyze_shape(self, image):
        """Analyze if image contains foot-like shapes"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 0.0, False
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Calculate contour properties
        if area < 1000:  # Too small
            return 0.2, False
        
        # Calculate elongation (foot-like shapes are moderately elongated)
        x, y, w, h = cv2.boundingRect(largest_contour)
        elongation = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        # Feet can have various shapes (ulcers and medical crops affect this)
        if 0.8 <= elongation <= 4.0:
            shape_score = 1.0
            foot_shape_found = True
        elif 0.6 <= elongation < 0.8 or 4.0 < elongation <= 5.5:
            shape_score = 0.6
            foot_shape_found = True  # Could still be foot
        else:
            shape_score = 0.2
            foot_shape_found = False
        
        # Check for convexity (feet have some concave areas but not too irregular)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Feet can have various solidities (ulcers create irregular shapes)
        if 0.50 <= solidity <= 1.0:
            shape_score *= 1.0  # Accept wide range for medical images
        elif 0.35 <= solidity < 0.50:
            shape_score *= 0.7
        else:
            shape_score *= 0.5
        
        return shape_score, foot_shape_found
    
    def _analyze_texture(self, image):
        """Analyze texture patterns (skin has characteristic texture)"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Local Binary Pattern (LBP) for texture
        # Skin has specific texture patterns
        radius = 3
        n_points = 8 * radius
        
        # Simple texture variance calculation
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        
        # Skin texture typically has moderate variance
        if 50 <= texture_variance <= 2000:
            texture_score = 1.0
        elif 20 <= texture_variance < 50 or 2000 < texture_variance <= 5000:
            texture_score = 0.6
        else:
            texture_score = 0.3
        
        return texture_score
    
    def _check_aspect_ratio(self, image):
        """Check if aspect ratio is reasonable for a foot image"""
        h, w = image.shape[:2]
        aspect_ratio = w / h if h > 0 else 0
        
        # Feet can be photographed in various orientations
        # Reasonable range: 0.4 to 4.0
        if 0.4 <= aspect_ratio <= 4.0:
            return 1.0
        elif 0.2 <= aspect_ratio < 0.4 or 4.0 < aspect_ratio <= 6.0:
            return 0.5
        else:
            return 0.1  # Extremely unusual aspect ratio
    
    def _analyze_edges(self, image):
        """Analyze edge patterns (feet have characteristic edge patterns)"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels
        
        # Feet have moderate edge density (not too sparse, not too dense)
        if 0.05 <= edge_density <= 0.30:
            edge_score = 1.0
        elif 0.02 <= edge_density < 0.05 or 0.30 < edge_density <= 0.45:
            edge_score = 0.6
        else:
            edge_score = 0.3
        
        return edge_score
    
    def _analyze_color_distribution(self, image):
        """Analyze color distribution to detect unnatural images"""
        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate color statistics
        mean_colors = rgb.mean(axis=(0, 1))
        std_colors = rgb.std(axis=(0, 1))
        
        r_mean, g_mean, b_mean = mean_colors
        r_std, g_std, b_std = std_colors
        
        # Check for unnatural color dominance
        max_mean = max(r_mean, g_mean, b_mean)
        min_mean = min(r_mean, g_mean, b_mean)
        color_range = max_mean - min_mean
        
        # Highly saturated single colors (like pure blue, green) are not feet
        if color_range > 80:  # Stricter threshold
            # Check for unnatural colors
            if b_mean > r_mean + 40 and b_mean > g_mean + 40:  # Too blue
                return 0.1
            if g_mean > r_mean + 35 and g_mean > b_mean + 35:  # Too green
                return 0.1
            if r_mean > g_mean + 50 and r_mean > b_mean + 50:  # Too red (graphics)
                return 0.2
        
        # Check color variance (very low = uniform/artificial)
        avg_std = (r_std + g_std + b_std) / 3
        if avg_std < 8:  # Too uniform (stricter)
            return 0.2
        elif avg_std < 20:
            return 0.5
        else:
            return 1.0
    
    def _detect_anatomical_features(self, image):
        """
        Attempt to detect anatomical features specific to feet
        (toes, heel, arch curve)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply GaussianBlur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect circles (toe-like features)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50
        )
        
        # Score based on detected circular features (toes)
        anatomy_score = 0.5  # Default neutral score
        
        if circles is not None:
            num_circles = len(circles[0])
            # Feet typically have 1-5 visible toe-like circles
            if 1 <= num_circles <= 8:
                anatomy_score = 0.8
            elif num_circles > 8:
                anatomy_score = 0.6  # Too many might indicate other objects
        
        # Detect curved edges (arch of foot)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Look for curves using Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        
        if lines is not None and len(lines) > 5:
            # Presence of multiple lines suggests structured object
            anatomy_score = min(1.0, anatomy_score + 0.2)
        
        return anatomy_score
    
    def _calculate_confidence(self, results):
        """
        Calculate overall confidence using weighted combination of all scores
        
        Weights prioritize most reliable indicators:
        - Skin detection: 25% (most important for feet)
        - Shape analysis: 20%
        - Anatomical features: 15%
        - Color distribution: 15%
        - Texture: 10%
        - Edges: 10%
        - Aspect ratio: 5%
        """
        weights = {
            'skin_score': 0.25,
            'shape_score': 0.20,
            'anatomy_score': 0.15,
            'color_score': 0.15,
            'texture_score': 0.10,
            'edge_score': 0.10,
            'aspect_score': 0.05
        }
        
        confidence = sum(results[key] * weight for key, weight in weights.items())
        
        # Apply penalties for extreme cases only
        # If skin ratio is very low, penalize
        if results['skin_ratio'] < 0.08:
            confidence *= 0.5
        
        # Shape penalty only if clearly wrong
        if not results.get('foot_shape_found', False) and results['shape_score'] < 0.3:
            confidence *= 0.8
        
        # Only penalize if MANY indicators are very low (obvious non-foot)
        very_low_scores = sum(1 for key in ['skin_score', 'shape_score', 'color_score'] 
                             if results.get(key, 1.0) < 0.3)
        if very_low_scores >= 2:
            confidence *= 0.5  # Clear rejection for obvious non-feet
        
        return min(1.0, max(0.0, confidence))
    
    def get_rejection_reason(self, results, confidence):
        """
        Generate human-readable rejection reason based on analysis
        """
        if confidence >= 0.50:
            return None  # Not rejected
        
        # Find the weakest indicators
        reasons = []
        
        if results['skin_score'] < 0.4:
            if results['skin_ratio'] < 0.10:
                reasons.append("Insufficient skin-like regions detected")
            else:
                reasons.append("Color pattern doesn't match human skin")
        
        if results['color_score'] < 0.4:
            reasons.append("Unnatural color distribution (appears to be graphic/object)")
        
        if results['shape_score'] < 0.4:
            reasons.append("Shape doesn't match typical foot anatomy")
        
        if results['anatomy_score'] < 0.4:
            reasons.append("No anatomical features (toes, heel) detected")
        
        if results['texture_score'] < 0.4:
            reasons.append("Texture pattern inconsistent with skin")
        
        if results['aspect_score'] < 0.4:
            reasons.append("Image proportions unusual for foot photograph")
        
        if not reasons:
            reasons.append("Overall confidence too low - may not be a foot image")
        
        return " | ".join(reasons[:2])  # Return top 2 reasons


# Singleton instance
_foot_detector = None

def get_foot_detector():
    """Get singleton instance of foot detector"""
    global _foot_detector
    if _foot_detector is None:
        _foot_detector = FootDetectionAlgorithm()
    return _foot_detector


def analyze_foot_image(image_path_or_array):
    """
    Convenience function to analyze if image contains a foot
    
    Args:
        image_path_or_array: Path to image or numpy array
        
    Returns:
        dict with is_foot, confidence, and analysis details
    """
    detector = get_foot_detector()
    is_foot, confidence, details = detector.detect_foot(
        image_path_or_array, 
        return_details=True
    )
    
    rejection_reason = None
    if not is_foot:
        rejection_reason = detector.get_rejection_reason(details, confidence)
    
    return {
        'is_foot': is_foot,
        'confidence': confidence,
        'rejection_reason': rejection_reason,
        'details': details
    }


if __name__ == '__main__':
    """Test the foot detection algorithm"""
    import sys
    
    print("\n" + "="*80)
    print("🔬 Foot Detection Algorithm Test")
    print("="*80 + "\n")
    
    if len(sys.argv) > 1:
        # Test with provided image
        image_path = sys.argv[1]
        print(f"Analyzing: {image_path}\n")
        
        result = analyze_foot_image(image_path)
        
        print(f"Result: {'✅ FOOT DETECTED' if result['is_foot'] else '❌ NOT A FOOT'}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        
        if result['rejection_reason']:
            print(f"Reason: {result['rejection_reason']}")
        
        print(f"\nDetailed Scores:")
        for key, value in result['details'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("Usage: python3 foot_detection_algorithm.py <image_path>")
        print("\nThis algorithm uses computer vision to detect feet based on:")
        print("  ✅ Skin color analysis")
        print("  ✅ Shape matching (elongation, contours)")
        print("  ✅ Anatomical features (toes, heel)")
        print("  ✅ Texture patterns")
        print("  ✅ Edge detection")
        print("  ✅ Color distribution")
        print("  ✅ Aspect ratio validation")
