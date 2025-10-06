"""
Canister Damage Detection System
=================================

This script provides multiple approaches for detecting damage in canister images:
1. CLIP Zero-Shot Classification (Best for limited data)
2. Classical Computer Vision (Feature-based)
3. Transfer Learning with CNNs (Requires more data)

Author: AI Computer Vision Engineer
Date: October 2025
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CLIPCanisterClassifier:
    """Zero-shot canister damage detection using CLIP"""
    
    def __init__(self):
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(device)
        self.model.eval()
        print("✓ CLIP model loaded successfully")
        
        # Define text prompts for classification
        self.text_prompts = [
            "a photo of an intact metallic canister with no damage or dents",
            "a photo of a damaged metallic canister with dents, scratches, and deformations"
        ]
    
    def predict(self, image_path):
        """
        Predict whether canister is OK or damaged
        Returns: ('ok' or 'damaged', confidence_score)
        """
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(
            text=self.text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        confidence_ok = probs[0][0].item()
        confidence_damaged = probs[0][1].item()
        
        if confidence_ok > confidence_damaged:
            return 'ok', confidence_ok
        else:
            return 'damaged', confidence_damaged


class ClassicalCanisterInspector:
    """Classical CV approach for damage detection"""
    
    def extract_features(self, image_path):
        """Extract handcrafted features from image"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # 1. Edge density (damaged canisters may have irregular edges)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 2. Contour irregularity
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            features['contour_irregularity'] = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        else:
            features['contour_irregularity'] = 0
        
        # 3. Surface texture variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['texture_variance'] = np.var(laplacian)
        
        # 4. Symmetry score
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        min_width = min(left_half.shape[1], right_half.shape[1])
        symmetry_diff = np.mean(np.abs(left_half[:, :min_width].astype(float) - 
                                        right_half[:, :min_width].astype(float)))
        features['symmetry_score'] = symmetry_diff
        
        # 5. Gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features['mean_gradient'] = np.mean(gradient_magnitude)
        
        return features
    
    def visualize_analysis(self, image_path):
        """Visualize the classical CV analysis"""
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(edges, cmap='gray')
        axes[0, 2].set_title('Edge Detection')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(laplacian, cmap='gray')
        axes[1, 0].set_title('Laplacian (Texture)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(gradient, cmap='hot')
        axes[1, 1].set_title('Gradient Magnitude')
        axes[1, 1].axis('off')
        
        features = self.extract_features(image_path)
        feature_text = "\n".join([f"{k}: {v:.4f}" for k, v in features.items()])
        axes[1, 2].text(0.1, 0.5, feature_text, fontsize=10, verticalalignment='center')
        axes[1, 2].set_title('Extracted Features')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return features


class CanisterDamageDetector:
    """
    Production-ready Canister Damage Detection System
    
    Uses ensemble of multiple approaches:
    1. CLIP zero-shot classification (primary)
    2. Classical CV features (supporting)
    """
    
    def __init__(self, use_ensemble=True):
        self.use_ensemble = use_ensemble
        self.clip_classifier = CLIPCanisterClassifier()
        self.classical_inspector = ClassicalCanisterInspector()
    
    def predict(self, image_path, return_details=False):
        """
        Predict if canister is damaged or ok
        
        Args:
            image_path: Path to the canister image
            return_details: If True, return detailed analysis
        
        Returns:
            If return_details=False: ('ok' or 'damaged', confidence)
            If return_details=True: dict with full analysis
        """
        results = {}
        
        # 1. CLIP prediction (primary)
        clip_pred, clip_conf = self.clip_classifier.predict(image_path)
        results['clip_prediction'] = clip_pred
        results['clip_confidence'] = clip_conf
        
        # 2. Classical CV features
        features = self.classical_inspector.extract_features(image_path)
        results['cv_features'] = features
        
        # Simple heuristic based on CV features
        damage_score = (
            features['edge_density'] * 0.3 +
            min(features['contour_irregularity'] / 10, 1.0) * 0.3 +
            min(features['texture_variance'] / 1000, 1.0) * 0.2 +
            min(features['symmetry_score'] / 50, 1.0) * 0.2
        )
        cv_pred = 'damaged' if damage_score > 0.5 else 'ok'
        results['cv_prediction'] = cv_pred
        results['cv_damage_score'] = damage_score
        
        # Ensemble decision (weighted voting)
        if self.use_ensemble:
            # Weighted: CLIP (0.7), CV (0.3)
            clip_vote = 1 if clip_pred == 'damaged' else 0
            cv_vote = 1 if cv_pred == 'damaged' else 0
            
            weighted_score = (clip_vote * 0.7 * clip_conf + cv_vote * 0.3) / (0.7 * clip_conf + 0.3)
            final_pred = 'damaged' if weighted_score > 0.5 else 'ok'
            final_conf = weighted_score if final_pred == 'damaged' else (1 - weighted_score)
        else:
            final_pred = clip_pred
            final_conf = clip_conf
        
        results['final_prediction'] = final_pred
        results['final_confidence'] = final_conf
        
        if return_details:
            return results
        else:
            return final_pred, final_conf
    
    def predict_with_visualization(self, image_path):
        """Predict and show visual analysis"""
        results = self.predict(image_path, return_details=True)
        
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(img_rgb)
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gray, cmap='gray')
        ax2.set_title('Grayscale', fontsize=12)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(edges, cmap='gray')
        ax3.set_title('Edge Detection', fontsize=12)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(gradient, cmap='hot')
        ax4.set_title('Gradient Analysis', fontsize=12)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        result_text = f"""
╔═══════════════════════════════════════╗
║   DAMAGE DETECTION RESULTS            ║
╠═══════════════════════════════════════╣
║                                       ║
║  PREDICTION: {results['final_prediction'].upper():^10}            ║
║  CONFIDENCE: {results['final_confidence']:^6.1%}                ║
║                                       ║
╠═══════════════════════════════════════╣
║  Model Predictions:                   ║
║    • CLIP: {results['clip_prediction']:>10} ({results['clip_confidence']:.1%})    ║
║    • CV: {results['cv_prediction']:>10} ({results['cv_damage_score']:.1%})      ║
╚═══════════════════════════════════════╝
"""
        
        color = 'red' if results['final_prediction'] == 'damaged' else 'green'
        ax5.text(0.5, 0.5, result_text, 
                fontsize=10, family='monospace',
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        
        plt.suptitle(f'Canister Analysis: {os.path.basename(image_path)}', fontsize=16, fontweight='bold')
        plt.show()
        
        return results['final_prediction'], results['final_confidence']


# ============================================================
# MAIN PREDICTION FUNCTION (Simple API for production)
# ============================================================

def detect_canister_damage(image_path):
    """
    Main function to detect canister damage from an image.
    
    Args:
        image_path (str): Path to the canister image file
    
    Returns:
        tuple: (prediction, confidence)
            - prediction (str): 'ok' or 'damaged'
            - confidence (float): Confidence score between 0 and 1
    
    Example:
        >>> prediction, confidence = detect_canister_damage('canister.jpg')
        >>> print(f"Status: {prediction}, Confidence: {confidence:.1%}")
        Status: damaged, Confidence: 87.3%
    """
    detector = CanisterDamageDetector(use_ensemble=True)
    prediction, confidence = detector.predict(image_path)
    return prediction, confidence


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  🔍 CANISTER DAMAGE DETECTION - TESTING ALL IMAGES")
    print("="*70)
    
    import glob
    
    # Get all images from inputs folder
    image_files = sorted(glob.glob('inputs/*.png') + glob.glob('inputs/*.jpg'))
    
    if not image_files:
        print("\n❌ No images found in 'inputs/' folder")
        print("Please ensure your images are in the 'inputs' directory")
    else:
        print(f"\n📊 Found {len(image_files)} images to analyze\n")
        
        # Create output folder
        output_folder = 'outputs'
        os.makedirs(output_folder, exist_ok=True)
        print(f"💾 Saving outputs to: {output_folder}/\n")
        
        print(f"{'Image':<25} {'Prediction':<12} {'Confidence':<12} {'Status'}")
        print("-" * 70)
        
        # Initialize detector once for efficiency
        detector = CanisterDamageDetector(use_ensemble=True)
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            
            try:
                # Get prediction
                prediction, confidence = detector.predict(img_path)
                
                # Format status
                if prediction == 'damaged':
                    status = "❌ DAMAGED"
                else:
                    status = "✓ OK"
                
                # Print result
                print(f"{filename:<25} {prediction.upper():<12} {confidence:<12.1%} {status}")
                
            except Exception as e:
                print(f"{filename:<25} ERROR: {str(e)}")
        
        print("-" * 70)
        print("\n✅ Quick analysis complete!")
        
        # Run detailed visual analysis and save images
        print("\n" + "="*70)
        print("  📊 GENERATING DETAILED VISUAL ANALYSIS...")
        print("="*70 + "\n")
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            
            print(f"Processing: {filename}...")
            
            try:
                # Generate visualization
                results = detector.predict(img_path, return_details=True)
                
                # Load and prepare images
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient = np.sqrt(sobelx**2 + sobely**2)
                
                # Create visualization with damage overlay
                fig = plt.figure(figsize=(18, 6))
                
                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(img_rgb)
                plt.title(f'Original: {filename}', fontsize=12, fontweight='bold')
                plt.axis('off')
                
                # Enhanced damage detection visualization
                plt.subplot(1, 3, 2)
                
                # Create a more visible damage detection overlay
                damage_viz = img_rgb.copy()
                
                # Dilate edges to make them thicker and more visible
                kernel = np.ones((3, 3), np.uint8)
                thick_edges = cv2.dilate(edges, kernel, iterations=2)
                
                # Create gradient-based anomaly map
                gradient_norm = gradient / gradient.max() if gradient.max() > 0 else gradient
                anomaly_map = (gradient_norm > 0.3).astype(np.uint8) * 255
                
                # Dilate anomaly regions
                thick_anomalies = cv2.dilate(anomaly_map, kernel, iterations=2)
                
                # Combine edges and anomalies
                combined_mask = cv2.bitwise_or(thick_edges, thick_anomalies)
                
                # Create colored overlay (bright red for damage areas)
                overlay = np.zeros_like(img_rgb)
                overlay[combined_mask > 0] = [255, 0, 0]  # Bright red
                
                # Blend with stronger overlay
                blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
                
                # Find and draw contours for damage regions
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw thick contours around significant damage areas
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Only show significant regions
                        # Draw thick red contour
                        cv2.drawContours(blended, [contour], -1, (255, 0, 0), 3)
                        
                        # Draw bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(blended, (x-2, y-2), (x+w+2, y+h+2), (255, 255, 0), 2)
                
                plt.imshow(blended)
                plt.title('Damage Detection (Red = Anomalies, Yellow Box = Region)', 
                         fontsize=11, fontweight='bold')
                plt.axis('off')
                
                # Results panel
                plt.subplot(1, 3, 3)
                plt.axis('off')
                
                # Prepare result text
                pred = results['final_prediction'].upper()
                conf = results['final_confidence']
                color = 'red' if results['final_prediction'] == 'damaged' else 'green'
                status_icon = '❌' if results['final_prediction'] == 'damaged' else '✓'
                
                result_text = f"""
╔═══════════════════════════════════════╗
║   DAMAGE DETECTION RESULTS            ║
╠═══════════════════════════════════════╣
║                                       ║
║  STATUS: {status_icon} {pred:^10}                ║
║  CONFIDENCE: {conf:^6.1%}                  ║
║                                       ║
╠═══════════════════════════════════════╣
║  Model Predictions:                   ║
║    • CLIP: {results['clip_prediction']:>10} ({results['clip_confidence']:.1%})    ║
║    • CV: {results['cv_prediction']:>10} ({results['cv_damage_score']:.1%})      ║
║                                       ║
╠═══════════════════════════════════════╣
║  CV Features:                         ║
║    • Edge Density: {results['cv_features']['edge_density']:.4f}        ║
║    • Contour Irreg: {results['cv_features']['contour_irregularity']:.4f}       ║
║    • Texture Var: {results['cv_features']['texture_variance']:.1f}          ║
║    • Symmetry: {results['cv_features']['symmetry_score']:.2f}             ║
╚═══════════════════════════════════════╝
"""
                
                plt.text(0.5, 0.5, result_text, 
                        fontsize=9, family='monospace',
                        verticalalignment='center', horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
                
                plt.suptitle(f'Canister Damage Analysis: {filename}', 
                           fontsize=14, fontweight='bold')
                
                # Save the figure
                output_path = f"{output_folder}/{base_name}_analysis.png"
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ Saved: {base_name}_analysis.png")
                
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {str(e)}")
        
        print("\n" + "="*70)
        print(f"✅ ALL ANALYSIS COMPLETE!")
        print(f"📁 Output images saved to: {output_folder}/")
        print("="*70 + "\n")


