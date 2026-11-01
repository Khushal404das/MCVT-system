"""
Vehicle Feature Extraction Module
Extracts deep appearance features + color histogram + shape descriptors
for cross-camera vehicle re-identification.

Inspired by the CVPR AIC Track 2 approach (ResNet50 + Metadata).
Uses a simpler but effective approach suitable for real-time processing.
"""

import cv2
import numpy as np
from collections import OrderedDict


class FeatureExtractor:
    """
    Extracts multi-modal features from vehicle crops for re-identification.
    
    Features include:
    1. Color histogram (HSV space) - captures vehicle color
    2. Spatial color distribution - captures color layout
    3. Shape/aspect ratio features
    4. Edge/texture features (HOG-like)
    5. Deep CNN features (if model available)
    """

    def __init__(self, feature_dim=256):
        self.feature_dim = feature_dim
        # Color bins for HSV histogram
        self.h_bins = 30
        self.s_bins = 32
        self.v_bins = 32
        
        # Spatial grid for color distribution
        self.grid_rows = 3
        self.grid_cols = 4
        
        # Standard resize for feature extraction
        self.standard_size = (128, 128)
        
        # Cache projection matrices for fast real-time projection
        self.proj_matrices = {}

        # Try to load a deep feature model
        self.deep_model = None
        self._init_deep_model()

        print(f"[FeatureExtractor] Initialized with dim={feature_dim}")

    def _init_deep_model(self):
        """Try to initialize a deep CNN for feature extraction."""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Use MobileNetV2 for efficiency (good balance of speed/accuracy)
            model = models.mobilenet_v2(pretrained=True)
            # Remove classifier, keep feature extractor
            self.deep_model = torch.nn.Sequential(*list(model.children())[:-1])
            self.deep_model.eval()
            self.deep_model.to(self.device)
            
            self.deep_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            
            print("[FeatureExtractor] Deep CNN (MobileNetV2) loaded successfully")
        except Exception as e:
            print(f"[FeatureExtractor] No deep model available: {e}")
            print("[FeatureExtractor] Using handcrafted features only")
            self.deep_model = None

    def extract(self, vehicle_crop):
        """
        Extract a comprehensive feature vector from a vehicle crop.
        
        Args:
            vehicle_crop: BGR image (numpy array) of a detected vehicle
            
        Returns:
            numpy array: L2-normalized feature vector
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        features = []

        # 1. Global color histogram (HSV)
        color_hist = self._extract_color_histogram(vehicle_crop)
        features.append(color_hist)

        # 2. Spatial color distribution
        spatial_color = self._extract_spatial_color(vehicle_crop)
        features.append(spatial_color)

        # 3. Shape features
        shape_feat = self._extract_shape_features(vehicle_crop)
        features.append(shape_feat)

        # 4. Edge/texture features
        texture_feat = self._extract_texture_features(vehicle_crop)
        features.append(texture_feat)

        # 5. Deep CNN features (if available)
        if self.deep_model is not None:
            deep_feat = self._extract_deep_features(vehicle_crop)
            features.append(deep_feat)

        # Concatenate all features
        full_feature = np.concatenate(features)

        # Reduce to fixed dimension via PCA-like projection (deterministic)
        feat_len = len(full_feature)
        if feat_len > self.feature_dim:
            if feat_len not in self.proj_matrices:
                rng = np.random.RandomState(42)
                proj = rng.randn(feat_len, self.feature_dim).astype(np.float32)
                proj /= (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-12)
                self.proj_matrices[feat_len] = proj
            full_feature = full_feature.astype(np.float32) @ self.proj_matrices[feat_len]
        elif feat_len < self.feature_dim:
            full_feature = np.pad(full_feature, (0, self.feature_dim - feat_len))

        # L2 normalize
        norm = np.linalg.norm(full_feature)
        if norm > 0:
            full_feature = full_feature / norm

        return full_feature.astype(np.float32)

    def _extract_color_histogram(self, crop):
        """Extract HSV color histogram."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # H channel histogram
        h_hist = cv2.calcHist([hsv], [0], None, [self.h_bins], [0, 180])
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        
        # S channel histogram
        s_hist = cv2.calcHist([hsv], [1], None, [self.s_bins], [0, 256])
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        
        return np.concatenate([h_hist, s_hist])

    def _extract_spatial_color(self, crop):
        """Extract color features from spatial grid regions."""
        resized = cv2.resize(crop, self.standard_size)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        h, w = resized.shape[:2]
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols
        
        spatial_features = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cell = hsv[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                # Mean color per cell
                mean_color = np.mean(cell, axis=(0, 1))
                spatial_features.extend(mean_color)
                # Dominant hue
                h_hist = cv2.calcHist([cell], [0], None, [8], [0, 180])
                h_hist = cv2.normalize(h_hist, h_hist).flatten()
                spatial_features.extend(h_hist)
        
        return np.array(spatial_features, dtype=np.float32)

    def _extract_shape_features(self, crop):
        """Extract shape/geometry features."""
        h, w = crop.shape[:2]
        aspect_ratio = w / max(h, 1)
        area_ratio = (h * w) / (self.standard_size[0] * self.standard_size[1])
        
        # Edge density
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / max(edges.size, 1)
        
        # Contour features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), 100) / 100.0  # Normalized
        
        return np.array([aspect_ratio, area_ratio, edge_density, num_contours], dtype=np.float32)

    def _extract_texture_features(self, crop):
        """Extract texture features using gradient histograms."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        
        # Gradient magnitude and orientation
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi + 180  # 0-360
        
        # HOG-like histogram
        n_bins = 18
        hist = np.zeros(n_bins, dtype=np.float32)
        bin_width = 360.0 / n_bins
        
        for i in range(n_bins):
            mask = ((angle >= i * bin_width) & (angle < (i + 1) * bin_width))
            hist[i] = np.sum(mag[mask])
        
        # Normalize
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        
        return hist

    def _extract_deep_features(self, crop):
        """Extract deep CNN features and ensure a 1‑D vector.

        MobileNetV2 without its classifier produces a spatial feature map (e.g.,
        shape (1280, 7, 7)).  The previous implementation attempted to ``squeeze``
        the tensor, which leaves a 3‑D array when the spatial dimensions are not
        size‑1, causing a dimension mismatch when concatenating with other
        handcrafted features.  We now flatten the tensor (or optionally average
        across spatial dimensions) so the result is always 1‑D.
        """
        import torch
        
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            input_tensor = self.deep_transform(rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.deep_model(input_tensor)
            
            # Convert to NumPy and ensure 1‑D
            features = features.squeeze().cpu().numpy()
            if features.ndim > 1:
                # Collapse spatial dimensions by taking the mean across them
                features = features.mean(axis=tuple(range(1, features.ndim)))
            
            # Truncate or pad to a reasonable length (128)
            if features.shape[0] > 128:
                features = features[:128]
            elif features.shape[0] < 128:
                features = np.pad(features, (0, 128 - features.shape[0]))
            
            return features.astype(np.float32)
        except Exception as e:
            # Fallback to zero vector on any error
            return np.zeros(128, dtype=np.float32)


    def compute_similarity(self, feat1, feat2):
        """
        Compute cosine similarity between two feature vectors.
        
        Returns:
            float: similarity score in [0, 1]
        """
        if feat1 is None or feat2 is None:
            return 0.0
        
        dot = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot / (norm1 * norm2)
        return float(max(0.0, min(1.0, (similarity + 1) / 2)))  # Map from [-1,1] to [0,1]

    def compute_distance_matrix(self, features_list1, features_list2):
        """
        Compute pairwise distance matrix between two sets of features.
        
        Returns:
            numpy array: distance matrix (lower = more similar)
        """
        n1 = len(features_list1)
        n2 = len(features_list2)
        dist_matrix = np.zeros((n1, n2), dtype=np.float32)
        
        for i in range(n1):
            for j in range(n2):
                sim = self.compute_similarity(features_list1[i], features_list2[j])
                dist_matrix[i, j] = 1.0 - sim
        
        return dist_matrix


def get_dominant_color_name(crop):
    """
    Robust vehicle color classification designed for outdoor surveillance.
    Filters out windshield/sky reflections and accurately distinguishes neutral
    tones (white, silver, black) from chromatic colors.
    """
    if crop is None or crop.size == 0:
        return 'unknown'

    h, w = crop.shape[:2]
    if h < 15 or w < 15:
        return 'unknown'

    # Focus on the lower 60% of the vehicle (doors, bumpers, hood/trunk)
    # avoiding the upper 35% where windows and sky reflections dominate.
    body = crop[int(h * 0.35):int(h * 0.92), int(w * 0.12):int(w * 0.88)]
    if body.size == 0:
        body = crop

    # Convert to float BGR
    b = body[:, :, 0].astype(np.float32)
    g = body[:, :, 1].astype(np.float32)
    r = body[:, :, 2].astype(np.float32)

    # Convert to HSV
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # Calculate statistics
    median_v = float(np.median(v_chan))
    median_s = float(np.median(s_chan))
    v75 = float(np.percentile(v_chan, 75))

    # Fraction of pixels in different brightness bands
    frac_white = float(np.mean(v_chan > 155))   # bright paint pixels (white car bodies)
    frac_dark  = float(np.mean(v_chan < 70))    # shadow / dark body pixels

    # Chroma: difference between max and min color channels
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    chroma = max_c - min_c
    mean_chroma = float(np.mean(chroma))
    median_chroma = float(np.median(chroma))

    # Effective brightness (perceptual luminance)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    mean_lum = float(np.mean(luminance))

    # 1. Neutral / Achromatic detection:
    # In outdoor daylight, camera auto white-balance imparts a slight blue cast
    # (Hue 105-125) with saturation up to 60 even on neutral white/silver/black surfaces.
    # True chromatic colors have high chroma (> 35) or high saturation (> 75).
    is_neutral = False

    if median_chroma < 26 and mean_chroma < 30:
        is_neutral = True
    elif median_s < 45 and median_chroma < 32:
        is_neutral = True
    elif 95 <= np.median(h_chan) <= 130:
        # Daylight skylight cast: real blue cars have strong blue channel dominance
        b_dominance = float(np.mean(b - np.maximum(r, g)))
        if b_dominance < 22 or median_chroma < 36 or median_s < 75:
            is_neutral = True

    if is_neutral:
        # White: many bright paint pixels even when dark windows/tyres pull median down
        if (frac_white > 0.25 and median_s < 55) or (frac_white > 0.40) or (v75 > 165 and median_s < 60):
            return 'white'
        # Black: majority of pixels are dark, or clearly low V
        elif frac_dark > 0.45 or (median_v < 78 and v75 < 120) or (mean_lum < 92 and frac_white < 0.12 and frac_dark > 0.12):
            return 'black'
        # White fallback: high overall brightness
        elif mean_lum > 148 or (median_v > 155 and frac_white > 0.15):
            return 'white'
        else:
            return 'silver'

    # 2. Chromatic colors (strong color presence)
    # Mask out shadows (very low V) and specular highlights (very low S)
    valid_mask = (v_chan > 40) & (s_chan > 40)
    if np.sum(valid_mask) > 10:
        dominant_h = float(np.median(h_chan[valid_mask]))
    else:
        dominant_h = float(np.median(h_chan))

    if dominant_h < 10 or dominant_h > 170:
        return 'red'
    elif dominant_h < 25:
        return 'orange'
    elif dominant_h < 35:
        return 'yellow'
    elif dominant_h < 85:
        return 'green'
    elif dominant_h < 132:
        return 'blue'
    elif dominant_h < 165:
        return 'purple'

    return 'unknown'
