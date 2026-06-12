#!/usr/bin/env python3

"""
=============================================================
Streak Detection: Canny Edges + Hough Lines
=============================================================
Goal: Detect satellite trails, Starlink chains, and airplane
      streaks in FITS images using classical OpenCV methods.

Pipeline:
  FITS → stretch → preprocess → Canny → HoughLinesP → filter → annotate

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch
import json
import argparse
import h5py
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Streak Detection"
        )
    
    parser.add_argument(
        '--image',
        type=str,
        help="Path to the FITS file to process"
        )
    
    parser.add_argument(
        '--dr',
        type=str,
        help="Path to the fistar DR file for star masking (optional)"
        )
    
    return parser.parse_args()


def load_and_stretch(filepath, hdu_index=1):
    """Load a FITS file and return both raw float32 and stretched uint8."""
    with fits.open(filepath) as hdul:
        data = hdul[hdu_index].data.astype(np.float32)

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    stretched = AsinhStretch(a=0.1)(norm)
    uint8 = (stretched * 255).astype(np.uint8)

    print(
        f"─────────────────────────────────────────────────\
        \n Loaded: {filepath}  shape={data.shape}  vmin={vmin:.1f}  vmax={vmax:.1f}\n"
        )
    return data, uint8


# ─────────────────────────────────────────────────────────────
# Preprocessing pipeline for streak detection
# ─────────────────────────────────────────────────────────────

def preprocess_for_streaks(uint8_img):
    """
    Prepare the image so that streaks are maximally visible
    and stars/noise are suppressed before edge detection.

    Returns dict of intermediate stages
    """
    stages = {}
    stages['input'] = uint8_img.copy()

    # ── Step 1: Median filter ─────────────────────────────────
    # Removes cosmic rays and hot pixels without blurring streak edges.
    # Use 3×3 — enough to kill isolated spikes.
    denoised = cv2.medianBlur(uint8_img, 3)
    stages['denoised'] = denoised

    # ── Step 2: Background subtraction ───────────────────────
    # Large Gaussian (kernel 61×61) captures the slowly varying
    # sky background and vignetting. Subtracting it flattens the field
    # so thresholding works uniformly everywhere in the image.
    background = cv2.GaussianBlur(denoised, (61, 61), sigmaX=25)
    bg_sub = cv2.subtract(denoised, background)
    bg_sub = cv2.normalize(bg_sub, None, 0, 255, cv2.NORM_MINMAX)
    stages['bg_subtracted'] = bg_sub

    # ── Step 3: CLAHE — local contrast enhancement ────────────
    # Boosts faint streaks in regions where the background is dark.
    # clipLimit=2 prevents over-amplifying noise.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bg_sub)
    stages['clahe'] = enhanced

    # ── Step 4: Unsharp mask ──────────────────────────────────
    # Sharpens edges — makes streak boundaries crisper for Canny.
    # Formula: sharpened = original + weight * (original - blurred)
    blur_for_sharp = cv2.GaussianBlur(enhanced, (9, 9), sigmaX=3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blur_for_sharp, -0.5, 0)
    stages['sharpened'] = sharpened

    return stages


# ─────────────────────────────────────────────────────────────
# Canny Edge Detector (deep dive)
# ─────────────────────────────────────────────────────────────

def apply_canny(img, low_thresh=30, high_thresh=90):
    """
    Canny works in 4 internal steps:
      1. Gaussian blur (built-in) — smooths noise
      2. Sobel gradient — finds intensity changes in X and Y
      3. Non-maximum suppression — thins edges to 1px wide
      4. Hysteresis thresholding — two thresholds:
            - pixels ABOVE high_thresh   → definitely an edge
            - pixels BETWEEN low & high  → edge only if connected to a strong edge
            - pixels BELOW low_thresh    → discarded

    For astronomy streaks:
      - Too low threshold  → stars produce false edges everywhere
      - Too high threshold → faint streaks get missed
      - Ratio 1:3 (low:high) is a good starting point

    HOW TO TUNE:
      - Raise high_thresh if too many noise edges appear
      - Lower low_thresh  if streak edges are broken/gappy
    """
    edges = cv2.Canny(img, low_thresh, high_thresh, apertureSize=3, L2gradient=True)
    return edges


def explore_canny_thresholds(prepped_img):
    """Show how different Canny thresholds affect detection. Study this!"""

    # ~ (250, 750)
    configs = [
        (75,  150),
        (75,  200),
        (75,  250),
        (75, 300),

        (100,  200),
        (100,  250),
        (100,  350),
        (100, 450),

        (150,  300),
        (150,  350),
        (150,  450),
        (150, 550),

        (200,  300),
        (200,  400),
        (200,  500),
        (200, 600)
    ]

    fig, axes = plt.subplots(2, 8, figsize=(30, 20))
    fig.suptitle('Canny Threshold Exploration\n'
                 'low:high ratio stays ~1:3 — only sensitivity changes',
                 fontsize=12, fontweight='bold')

    for ax, (lo, hi) in zip(axes.flatten(), configs):
        edges = apply_canny(prepped_img, lo, hi)
        edge_px = np.sum(edges > 0)
        ax.imshow(edges, cmap='gray', origin='lower')
        ax.set_title(f'low={lo}, high={hi}  |  edge pixels={edge_px}', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('module2_canny_exploration.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("[+] Saved → module2_canny_exploration.png")


# ─────────────────────────────────────────────────────────────
# Hough Line Transform (the core detector)
# ─────────────────────────────────────────────────────────────

def detect_streaks_hough(edge_img, original_uint8, config=None):
    """
    HoughLinesP — Probabilistic Hough Line Transform.

    How it works:
      Every edge pixel "votes" for all lines that pass through it
      in a (rho, theta) parameter space called the accumulator.
      Where many votes accumulate → a line exists.

      'Probabilistic' = only samples a random subset of edge pixels
      → faster, returns line SEGMENTS (x1,y1,x2,y2) not infinite lines.

    Key parameters:
      rho        — distance resolution in pixels (1 = precise)
      theta      — angle resolution in radians (np.pi/180 = 1°)
      threshold  — minimum votes for a line to be accepted
                   higher → fewer, more certain detections
      minLineLength — shortest segment to report (pixels)
                      set to ~10% of image diagonal for real streaks
      maxLineGap    — max gap in pixels between collinear segments
                      to merge them into one line
                      larger → connects broken streaks

    TUNING GUIDE:
      Too many false positives → raise threshold or minLineLength
      Missing faint streaks   → lower threshold, raise maxLineGap
    """

    lines = cv2.HoughLinesP(
        edge_img,
        rho           = config['rho'],
        theta         = config['theta'],
        threshold     = config['threshold'],
        minLineLength = config['minLineLength'],
        maxLineGap    = config['maxLineGap'],
    )

    return lines


# ─────────────────────────────────────────────────────────────
# Merge collinear Hough segments into cleaner detections
# ─────────────────────────────────────────────────────────────

def merge_collinear_segments(lines, angle_tol=3.0, dist_tol=30.0):
    """
    Merge Hough segments that belong to the same physical streak.

    Two segments are merged if:
      1. Their angles agree within angle_tol degrees
      2. The perpendicular distance between their midpoints
         is within dist_tol pixels (same track across the image)

    Returns a reduced list of lines where each entry represents
    one merged streak, keeping the outermost endpoints.
    """
    if lines is None:
        return None

    # Convert to list of [x1,y1,x2,y2] for easier handling
    segs = [line[0].tolist() for line in lines]

    def seg_angle(s):
        return np.degrees(np.arctan2(s[3]-s[1], s[2]-s[0])) % 180

    def perp_distance(s1, s2):
        """Distance from s2's midpoint to the infinite line through s1."""
        x1,y1,x2,y2 = s1
        mx,my = (s2[0]+s2[2])/2, (s2[1]+s2[3])/2
        # Line direction vector
        dx,dy = x2-x1, y2-y1
        length = np.sqrt(dx*dx + dy*dy) + 1e-9
        # Perpendicular distance formula
        return abs(dy*mx - dx*my + x2*y1 - y2*x1) / length

    def merge_two(s1, s2):
        """Return the segment spanning the outermost endpoints of s1 and s2."""
        pts = [(s1[0],s1[1]), (s1[2],s1[3]),
               (s2[0],s2[1]), (s2[2],s2[3])]
        # Project all points onto the direction of s1
        dx = s1[2]-s1[0]; dy = s1[3]-s1[1]
        length = np.sqrt(dx*dx+dy*dy)+1e-9
        projs = [(dx*p[0]+dy*p[1])/length for p in pts]
        i_min = np.argmin(projs); i_max = np.argmax(projs)
        return [pts[i_min][0], pts[i_min][1],
                pts[i_max][0], pts[i_max][1]]

    merged = True
    while merged:
        merged = False
        used = set()
        new_segs = []
        for i in range(len(segs)):
            if i in used:
                continue
            current = segs[i]
            for j in range(i+1, len(segs)):
                if j in used:
                    continue
                a1 = seg_angle(current)
                a2 = seg_angle(segs[j])
                angle_diff = abs(a1-a2)
                angle_diff = min(angle_diff, 180-angle_diff)
                if angle_diff < angle_tol and perp_distance(current, segs[j]) < dist_tol:
                    current = merge_two(current, segs[j])
                    used.add(j)
                    merged = True
            used.add(i)
            new_segs.append(current)
        segs = new_segs

    # Repack into numpy format HoughLinesP returns
    return np.array([[[int(s[0]),int(s[1]),int(s[2]),int(s[3])]] for s in segs])

# ─────────────────────────────────────────────────────────────
# Filter and classify detected lines
# ─────────────────────────────────────────────────────────────

def compute_line_properties(x1, y1, x2, y2):
    """Compute length, angle, and midpoint of a line segment."""
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    angle  = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180   # 0–180°
    mid_x  = (x1 + x2) / 2
    mid_y  = (y1 + y2) / 2
    return length, angle, mid_x, mid_y


def group_parallel_lines(line_props, angle_tol=8.0, dist_tol=50.0):
    """
    Identify groups of parallel lines — the signature of Starlink chains.

    Two lines are 'parallel' if:
      - Their angles differ by < angle_tol degrees
      - Their midpoints are within dist_tol pixels

    Returns list of groups; groups with 2+ lines → Starlink candidate.
    """
    groups = []
    used = set()

    for i, p1 in enumerate(line_props):
        if i in used:
            continue
        group = [i]
        for j, p2 in enumerate(line_props):
            if j <= i or j in used:
                continue
            angle_diff = abs(p1['angle'] - p2['angle'])
            angle_diff = min(angle_diff, 180 - angle_diff)   # handle wrap-around
            mid_dist   = np.sqrt((p1['mid_x'] - p2['mid_x'])**2 +
                                 (p1['mid_y'] - p2['mid_y'])**2)
            if angle_diff < angle_tol and mid_dist < dist_tol:
                group.append(j)
                used.add(j)
        used.add(i)
        groups.append(group)

    return groups


def measure_streak_profile(img, detection, n_samples=10):
    """
    Sample the brightness profile perpendicular to the streak
    at n_samples points along its length.
    Returns the mean peak width (in pixels) of the cross-section.
    A wide smooth profile = real trail (Gaussian PSF convolved with streak)
    A narrow sharp profile = detector artifact
    """
    x1,y1,x2,y2 = detection['x1'],detection['y1'],detection['x2'],detection['y2']
    
    # Unit vector along the streak
    dx, dy = x2-x1, y2-y1
    length = np.sqrt(dx*dx + dy*dy)
    ux, uy = dx/length, dy/length
    
    # Perpendicular unit vector
    px, py = -uy, ux
    
    widths = []
    for i in range(n_samples):
        # Sample point along the streak
        t = (i + 1) / (n_samples + 1)
        cx = x1 + t*dx
        cy = y1 + t*dy
        
        # Sample 20px either side perpendicularly
        profile = []
        for offset in range(-10, 11):
            sx = int(cx + offset*px)
            sy = int(cy + offset*py)
            if 0 <= sy < img.shape[0] and 0 <= sx < img.shape[1]:
                profile.append(float(img[sy, sx]))
        
        if len(profile) < 5:
            continue
            
        profile = np.array(profile)
        peak = profile.max()
        background = profile.min()
        if peak - background < 5:
            continue
            
        # FWHM: how many pixels are above half the peak height
        half_max = background + (peak - background) * 0.5
        fwhm = np.sum(profile > half_max)
        widths.append(fwhm)
    
    return np.mean(widths) if widths else 0.0


def classify_detections(img, lines, min_length=200):
    """
    Given raw HoughLinesP output, classify each line as:
      - 'satellite'  : single long streak
      - 'starlink'   : member of a parallel group (2+ parallel lines)
      - 'airplane'   : shorter streak, moderate brightness
      - 'noise'      : too short, reject

    Returns list of detection dicts with classification + properties.
    """
    if lines is None:
        return []

    img_diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    detections = []

    # Compute properties for all accepted lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length, angle, mid_x, mid_y = compute_line_properties(x1, y1, x2, y2)

        print(length)
        if length < min_length:
            continue   # too short → noise

        d = {
            'x1': int(x1), 'y1': int(y1),
            'x2': int(x2), 'y2': int(y2),
            'length': float(length),
            'angle':  float(angle),
            'mid_x':  float(mid_x),
            'mid_y':  float(mid_y),
            'label':  'satellite',
        }

        # detections.append({
        #     'x1': int(x1), 'y1': int(y1),
        #     'x2': int(x2), 'y2': int(y2),
        #     'length': float(length),
        #     'angle':  float(angle),
        #     'mid_x':  float(mid_x),
        #     'mid_y':  float(mid_y),
        #     'label':  'satellite',      # default, may be updated below
        # })

        profile_width = measure_streak_profile(img, d)
        d['profile_width'] = profile_width

        # Real trails: PSF-broadened, typically 4–15px wide
        # Artifacts: 1–2px sharp step
        if profile_width < 3.0:
            continue   # reject as detector artifact

        detections.append(d)

    # Check for Starlink groups
    if len(detections) >= 2:
        groups = group_parallel_lines(detections, angle_tol=10, dist_tol=80)
        for group in groups:
            if len(group) >= 2:
                for idx in group:
                    detections[idx]['label'] = 'starlink'

    # Reclassify shorter single streaks as airplane candidates
    for d in detections:
        if d['label'] == 'satellite' and d['length'] < img_diagonal * 0.25:
            d['label'] = 'airplane'

    return detections


def build_star_mask_from_dr(dr_path, img_shape, radius=4):
    """
    Build a star mask from fistar extractions using circular masks.
    Radius per star = s * psf_scale, where s is the Gaussian sigma.

    psf_scale=2.0 ≈ 1 FWHM. Raise to 3.0 for safety margin.
    """

    mask = np.zeros(img_shape, dtype=np.uint8)

    with h5py.File(dr_path, 'r') as f:
        base = '/SourceExtraction/Version000/Sources'
        xs = f[f'{base}/x'][:]
        ys = f[f'{base}/y'][:]
        # ss = f[f'{base}/s'][:]

    print(f"[+] fistar sources: {len(xs)}")
    # print(f"[+] s range: {ss.min():.2f} – {ss.max():.2f}  mean={ss.mean():.2f}")

    for x, y in zip(xs, ys):
        # radius = max(3, int(round(s * psf_scale)))
        cv2.circle(mask, (int(round(x)), int(round(y))), radius, 255, -1)

    masked_fraction = np.sum(mask > 0) / mask.size
    print(f"[+] Mask coverage: {masked_fraction*100:.1f}% of image")

    return mask.astype(bool), zip(xs, ys)


def filter_by_star_overlap(detections, star_mask, overlap_threshold=0.85):
    """
    Reject detections where most pixels along the line land on stars.
    
    overlap_threshold=0.85 means: if 85% or more of the sampled
    pixels along this line are within a star's PSF, reject it.
    
    Real trails pass through open sky between stars.
    False positives from star halos/diffraction are almost entirely
    within star PSF regions.
    """
    kept = []
    for d in detections:
        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
        
        # Sample ~200 points along the line
        n_samples = min(200, int(d['length']))
        xs = np.linspace(x1, x2, n_samples).astype(int)
        ys = np.linspace(y1, y2, n_samples).astype(int)
        
        # Clip to image bounds
        valid = ((xs >= 0) & (xs < star_mask.shape[1]) &
                 (ys >= 0) & (ys < star_mask.shape[0]))
        xs, ys = xs[valid], ys[valid]
        
        if len(xs) == 0:
            continue
        
        # What fraction of line pixels fall on a star?
        on_star = np.sum(star_mask[ys, xs])
        overlap = on_star / len(xs)
        d['star_overlap'] = float(overlap)
        print(f"  overlap={overlap:.2f}  label={d['label']}  length={d['length']:.0f}px")
        if overlap >= overlap_threshold:
            print(f"  [rejected] {d['label']} {d['length']:.0f}px — "
                  f"{overlap*100:.0f}% on stars")
            continue
        
        kept.append(d)

    return kept

# ─────────────────────────────────────────────────────────────
# Annotate and visualize results
# ─────────────────────────────────────────────────────────────

# Color scheme (BGR for OpenCV drawing, RGB for matplotlib)
LABEL_COLORS = {
    'satellite': {'bgr': (0, 255, 100),   'rgb': (0,   1,   0.4), 'name': 'Satellite'},
    'starlink':  {'bgr': (0, 180, 255),   'rgb': (0,   0.7, 1),   'name': 'Starlink'},
    'airplane':  {'bgr': (50, 50, 255),   'rgb': (0.2, 0.2, 1),   'name': 'Airplane'},
    'noise':     {'bgr': (128, 128, 128), 'rgb': (0.5, 0.5, 0.5), 'name': 'Noise'},
}


def annotate_image(uint8_img, detections):
    """
    Draw detection results on the image:
      - Colored line over the streak
      - Label + length text
      - Returns BGR color image
    """
    # Convert grayscale to BGR so we can draw colored lines
    annotated = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)

    for d in detections:
        color = LABEL_COLORS[d['label']]['bgr']
        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']

        # font_scale = max(0.4, img_shape[1] / 1200)   # scales with image width
        # thickness   = max(1, int(img_shape[1] / 800))

        # Draw the streak line (thick)
        cv2.line(annotated, (x1, y1), (x2, y2), color, thickness=2)

        # Draw endpoints
        cv2.circle(annotated, (x1, y1), 4, color, -1)
        cv2.circle(annotated, (x2, y2), 4, color, -1)

        # Label text
        label_text = f"{d['label']}  {d['length']:.0f}px  {d['angle']:.1f}°"
        text_x = int(d['mid_x']) - 40
        text_y = int(d['mid_y']) - 8

        # Black background behind text for readability
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 10)
        cv2.rectangle(annotated,
                      (text_x - 2, text_y - th - 2),
                      (text_x + tw + 2, text_y + 2),
                      (0, 0, 0), -1)
        cv2.putText(annotated, label_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, color, 10, cv2.LINE_AA)

    return annotated


def plot_full_pipeline(stages, edge_img, detections, annotated_bgr, uint8_img):
    """Show the complete detection pipeline in one figure."""
    fig = plt.figure(figsize=(32, 24))
    fig.suptitle('Module 2 — Full Streak Detection Pipeline', fontsize=14, fontweight='bold')

    # Define subplot layout
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.imshow(stages['input'],        cmap='gray', origin='lower')
    ax1.set_title('1. Input (asinh stretched)', fontsize=9)
    ax1.axis('off')

    ax2.imshow(stages['bg_subtracted'], cmap='gray', origin='lower')
    ax2.set_title('2. Background subtracted', fontsize=9)
    ax2.axis('off')

    ax3.imshow(stages['sharpened'],     cmap='gray', origin='lower')
    ax3.set_title('3. Sharpened (unsharp mask)', fontsize=9)
    ax3.axis('off')

    ax4.imshow(edge_img,               cmap='gray', origin='lower')
    ax4.set_title('4. Canny edges (lo=30, hi=90)', fontsize=9)
    ax4.axis('off')

    # Hough raw lines on dark background
    hough_vis = np.zeros((*uint8_img.shape, 3), dtype=np.uint8)
    for d in detections:
        color = LABEL_COLORS[d['label']]['bgr']
        cv2.line(hough_vis, (d['x1'], d['y1']), (d['x2'], d['y2']), color, 2)
    ax5.imshow(cv2.cvtColor(hough_vis, cv2.COLOR_BGR2RGB), origin='lower')
    ax5.set_title(f'5. Hough detections ({len(detections)} found)', fontsize=9)
    ax5.axis('off')

    # Final annotated result
    ax6.imshow(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), origin='upper')
    ax6.set_title('6. Final annotated result', fontsize=9)
    ax6.axis('off')

    # Legend
    patches = [mpatches.Patch(color=v['rgb'], label=v['name'])
               for k, v in LABEL_COLORS.items() if k != 'noise']
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=10, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('module2_pipeline.png', dpi=130, bbox_inches='tight')
    plt.show()
    print("[+] Saved → module2_pipeline.png")


def plot_detection_stats(detections):
    """Visualize statistics of what was detected — useful for tuning."""
    if not detections:
        print("No detections to plot stats for.")
        return

    lengths = [d['length'] for d in detections]
    angles  = [d['angle']  for d in detections]
    labels  = [d['label']  for d in detections]

    label_counts = {k: labels.count(k) for k in set(labels)}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Module 2 — Detection Statistics', fontsize=12, fontweight='bold')

    # Streak length distribution
    axes[0].hist(lengths, bins=20, color='steelblue', edgecolor='white')
    axes[0].set_title('Streak lengths (pixels)')
    axes[0].set_xlabel('Length')
    axes[0].set_ylabel('Count')

    # Angle distribution (0–180°)
    axes[1].hist(angles, bins=36, range=(0, 180), color='coral', edgecolor='white')
    axes[1].set_title('Streak angles (degrees)')
    axes[1].set_xlabel('Angle')
    axes[1].set_ylabel('Count')

    # Class breakdown
    bar_colors = [LABEL_COLORS.get(k, LABEL_COLORS['noise'])['rgb']
                  for k in label_counts.keys()]
    bars = axes[2].bar(label_counts.keys(), label_counts.values(), color=bar_colors)
    axes[2].set_title('Detection class breakdown')
    axes[2].set_ylabel('Count')
    for bar, val in zip(bars, label_counts.values()):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(val), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('module2_stats.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("[+] Saved → module2_stats.png")


# ─────────────────────────────────────────────────────────────
# Detection report (JSON output for later pipeline)
# ─────────────────────────────────────────────────────────────

def save_detection_report(detections, fits_path, output_path="detections.json"):
    """
    Save detections as JSON — lays the groundwork for Module 6's
    full pipeline where you'll process folders of FITS files.
    """
    report = {
        'fits_file':   fits_path,
        'n_detections': len(detections),
        'Clean?':     len(detections) == 0,
        'detections':  detections,
        'summary': {
            'satellite': sum(1 for d in detections if d['label'] == 'satellite'),
            'starlink':  sum(1 for d in detections if d['label'] == 'starlink'),
            'airplane':  sum(1 for d in detections if d['label'] == 'airplane'),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n── Detection Report ──")
    print(f"  Clean?   : {report['Clean?']}")
    print(f"  Total    : {report['n_detections']} detections")
    for label, count in report['summary'].items():
        if count > 0:
            print(f"  {label:<12}: {count}")
    # print(f"  Saved    : {output_path}")

    return report


def save_ds9_regions(detections, star_positions, output_path='region.reg', coord_system='image'):
    """
    Save detections as a DS9 region file.
    
    coord_system: 'image' — pixel coordinates (use if no WCS in your FITS)
                  'fk5'   — RA/Dec (requires WCS, more work)
    """
    color_map = {
        'satellite': 'green',
        'starlink':  'cyan',
        'airplane':  'red',
    }

    with open(output_path, 'w') as f:
        # Header
        f.write('# Region file format: DS9 version 4.1\n')
        f.write(f'global color=green width=2 font="helvetica 10 normal" '
                f'select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1\n')
        f.write(f'{coord_system}\n')

        for d in detections:
            color  = color_map.get(d['label'], 'green')
            x1, y1 = d['x1'], d['y1']
            x2, y2 = d['x2'], d['y2']

            # DS9 image coords are 1-indexed — add 1
            f.write(
                f'line({x1+1},{y1+1},{x2+1},{y2+1}) '
                f'# color={color} width=2 '
                f'text={{  {d["label"]} {d["length"]:.0f}px '
                f'{d["angle"]:.1f}deg}}\n'
            )
        
        for x, y in star_positions:
            f.write(
                f'circle({x+1},{y+1},4) '
                f'# color=red width=1\n'
            )


    print(f"[+] Saved {len(detections)} regions → {output_path}")

# ─────────────────────────────────────────────────────────────
# MAIN — Run the full pipeline
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    args = parse_args()
    fits_path = args.image
    dr_path   = args.dr

    # ── Loading FITS...
    raw_data, uint8_img = load_and_stretch(fits_path)

    # ── Preprocessing...
    stages = preprocess_for_streaks(uint8_img)

    # ── Explore Canny thresholds (educational) ──
    # Exploring Canny thresholds...
    # explore_canny_thresholds(stages['sharpened'])

    # ── Running Canny edge detector...
    edge_img = apply_canny(
        stages['sharpened'],
        low_thresh=100,
        high_thresh=450
        )
    edge_pixels = np.sum(edge_img > 0)
    print(f"    Edge pixels found: {edge_pixels} ({100*edge_pixels/edge_img.size:.2f}% of image)")

    # ── Running Hough Line Transform...
    hough_config = {
        'rho':           1,
        'theta':         np.pi / 360,
        'threshold':     10,   # minimum votes to accept a line
        'minLineLength': 150,   # minimum length of line segments to report
        'maxLineGap':    12,   # max gap to connect collinear segments into one line
    }
    raw_lines = detect_streaks_hough(edge_img, uint8_img, hough_config)

    ############ ADDED: Merge collinear segments to get cleaner detections ############
    raw_lines = merge_collinear_segments(
        raw_lines,
        angle_tol=3.0,
        dist_tol=30.0
        )

    n_raw = len(raw_lines) if raw_lines is not None else 0
    print(f"    Raw Hough lines found: {n_raw}")

    # ── Filter and classify ──
    detections = classify_detections(
        uint8_img,
        raw_lines,
        min_length=200
        )
    
    mask, star_positions = build_star_mask_from_dr(dr_path, img_shape=raw_data.shape) if dr_path else (None, None)
    # sns.heatmap(mask.astype(float), cmap='gray', cbar=False)

    if mask is not None:
        detections = filter_by_star_overlap(detections, mask, overlap_threshold=0.85)


    save_ds9_regions(detections,
                     star_positions,
                     fits_path[fits_path.rfind("/") + 1:].replace(".fits.fz", ".reg")
                     )
    # for d in detections:
    #     print(f"      {d['label']:<12}  length={d['length']:.0f}px  angle={d['angle']:.1f}°")


    annotated_bgr = annotate_image(uint8_img, detections)
    # plot_full_pipeline(stages, edge_img, detections, annotated_bgr, uint8_img)
    # plot_detection_stats(detections)
    report = save_detection_report(detections, fits_path)


    # print(f"\n******** Clean image?: {len(detections) == 0} ********\n")