# SFD-DETR: An End-to-End Spatial–Frequency Dual-Guided Multi-Scale Network for Tiny Object Detection in UAV Aerial Images
## Algorithm Pseudocode

This document provides the pseudocode for the SFD-DETR algorithm for technical review purposes.

---

## 1. Core Modules

### 1.1 MSCS Block 

```
Algorithm 1: MSCS 

1: // First-level convolution
2: F1 ← Conv_3×3(X)
3: 
4: // Channel splitting (50/50 ratio)
5: c1 ← floor(C × α)
6: F1_p ← F1[0:c1, :, :]          // Progressive branch
7: F1_r ← F1[c1:C, :, :]          // Retained branch
8: 
9: // Second-level multi-scale convolution
10: F2 ← DWConv_5×5(F1_p)
11: c2 ← floor(c1 × α)
12: F2_p ← F2[0:c2, :, :]          // Progressive branch
13: F2_r ← F2[c2:c1, :, :]         // Retained branch
14: 
15: // Third-level multi-scale convolution
16: F3 ← DWConv_7×7(F2_p)
17: 
18: // Feature concatenation
19: F_concat ← Concat(F3, F2_r, F1_r, dim=1)
20: 
21: // Final projection with residual connection
22: Y ← Conv_1×1(F_concat) + X
23: 
24: return Y
```

### 1.2 CAC 

```
Algorithm 2: CAC Module

1: // Spatial neighborhood sampling with stride 2
2: for c = 0 to C-1 do
3:     for i = 0 to H/2-1 do
4:         for j = 0 to W/2-1 do
5:             X_00[c, i, j] ← X[c, 2i, 2j]         // Top-left pixel
6:             X_10[c, i, j] ← X[c, 2i+1, 2j]       // Top-right pixel
7:             X_01[c, i, j] ← X[c, 2i, 2j+1]       // Bottom-left pixel
8:             X_11[c, i, j] ← X[c, 2i+1, 2j+1]     // Bottom-right pixel
9:         end for
10:    end for
11: end for
12:
13: // Channel concatenation (4C channels)
14: X_concat ← Concat(X_00, X_10, X_01, X_11, dim=1)
15:
16: // Convolution for feature fusion
17: Y ← Conv_3×3(X_concat)           // Output C channels
18:
19: return Y
```

### 1.3 POF

```
Algorithm 3: POF Module

1: // Feature concatenation
2: F_concat ← Concat(F1, F2, F3, dim=1)
3: C ← F_concat.shape[1]
4: 
5: // Channel projection
6: F_proj ← Conv_1×1(F_concat)
7: 
8: // Channel splitting
9: C_e ← floor(C × e)               // Enhancement branch channels
10: C_r ← C - C_e                    // Identity branch channels
11: 
12: F_E ← F_proj[:, 0:C_e, :, :]     // Enhancement branch
13: F_I ← F_proj[:, C_e:C, :, :]     // Identity branch
14: 
15: // Omni-Kernel Module for enhancement
16: F_OKM ← OKM(F_E)
17: 
18: // Feature fusion
19: Y ← Conv_1×1(Concat(F_OKM, F_I, dim=1))
20: 
21: return Y
```

### 1.4 OKM 

```
Algorithm 4: OKM 

1: // Input projection
2: X_in ← Conv_1×1(X) + GELU(X)
3: 
4: // Multi-scale Spatial Convolution (MSC) branch
5: F_MSC ← MSC(X_in)
6: 
7: // Global Frequency Attention (GFA) branch
8: F_GFA ← GFA(X_in)
9: 
10: // Multi-domain enhancement
11: Y ← Conv_1×1(X_in + F_MSC + F_GFA)
12: 
13: return Y
```

### 1.5 MSC 

```
Algorithm 5: MSC (Multi-scale Spatial Convolution)
Input: Feature map X ∈ R^(C×H×W)
Output: Multi-scale feature F_MSC

1: // Input projection
2: X_in ← Conv_1×1(X) + GELU(X)
3: 
4: // Parallel depth-wise convolutions with different kernels
5: F_1×1 ← DWConv_1×1(X_in)        // Local details
6: F_31×1 ← DWConv_31×1(X_in)      // Horizontal strip
7: F_1×31 ← DWConv_1×31(X_in)      // Vertical strip
8: F_31×31 ← DWConv_31×31(X_in)    // Ultra-wide receptive field
9: 
10: // Multi-scale aggregation
11: F_MSC ← X_in + F_1×1 + F_31×1 + F_1×31 + F_31×31
12: F_MSC ← ReLU(F_MSC)
13: 
14: return F_MSC
```

### 1.6 GFA 

```
Algorithm 6: GFA Module
Input: Feature map X ∈ R^(C×H×W)
Output: Frequency-enhanced feature Y

1: // Dual-domain Channel Attention (DCA)
2: X_spatial ← Conv(GAP(X))         // Spatial descriptor
3: X_fft ← FFT2D(X)
4: X_freq ← X_fft ⊙ X_spatial        // Frequency modulation
5: X' ← IFFT2D(X_freq)
6: X' ← |X'|
7: X_DCA ← X' ⊙ Conv(GAP(X'))
8: 
9: // Frequency Spatial Attention (FSA)
10: Z ← FFT2D(Conv(X_DCA))
11: X_FSA ← IFFT2D(Z ⊙ Conv(X_DCA))
12: X_FSA ← |X_FSA|
13: 
14: // Frequency-domain gating (FGM)
15: X1 ← Conv_1×1(X_FSA)
16: X2 ← Conv_1×1(X_FSA)
17: X2_fft ← FFT2D(X2)
18: X_FGM ← |IFFT2D(X1 ⊙ X2_fft)| × α + X_FSA × β
19: 
20: return X_FGM
```

### 1.7 C2D 

```
Algorithm 7: Converse2D Operator
Input: Feature map X ∈ R^(C×H×W), kernel size K, scale factor S = 2
Output: Upsampled feature Y ∈ R^(C×(H×S)×(W×S))

1: // Circular padding
2: X_pad ← Pad(X, padding=K-1, mode='circular')
3: 
4: // Sparse upsampling target
5: STy ← Sparse_Upsample(X_pad, scale=S)
6: X_up ← Interpolate(X, scale=S, mode='nearest')
7: 
8: // Frequency domain convolution
9: FB ← FFT2D(Weight)               // Convolution kernel FFT
10: FBC ← Conjugate(FB)
11: F2B ← |FB|²
12: 
13: // Inverse problem solving
14: FBFy ← FBC ⊙ FFT2D(STy)
15: bias_eps ← Sigmoid(bias - 9.0) + ε
16: FR ← FBFy + FFT2D(bias_eps ⊙ X_up)
17: 
18: // Deconvolution operation
19: X1 ← FB ⊙ FR
20: FBR ← Mean(Split(X1, S), dim=-1)
21: invW ← Mean(Split(F2B, S), dim=-1)
22: invWBR ← FBR / (invW + bias_eps)
23: 
24: // Final reconstruction
25: FCBinvWBR ← FBC ⊙ Repeat(invWBR, S×S)
26: FX ← (FR - FCBinvWBR) / bias_eps
27: Y ← Real(IFFT2D(FX))
28: 
29: // Remove padding
30: Y ← Crop(Y, padding=(K-1)×S)
31: 
32: return Y
```

---

## 2. Training Procedure

```
Algorithm 8: SFD-DETR Training Process
Input: Dataset D, epochs E, batch size B
Output: Trained model M

1: // Initialize model
2: M ← InitializeModel(SFD-DETR_Config)
3: Optimizer ← AdamW(M.parameters(), lr=1e-4, weight_decay=1e-4)
4: 
5: for epoch = 1 to E do
6:     for batch in DataLoader(D, batch_size=B, shuffle=True) do
7:         // Data preprocessing
8:         images, gt_bboxes, gt_labels ← Preprocess(batch)
9:         
10:        // Forward propagation
11:        predictions ← M(images)
12:        
13:        // Loss calculation
14:        L_giou ← GIoU_Loss(predictions.bboxes, gt_bboxes)
15:        L_cls ← Focal_Loss(predictions.cls, gt_labels)
16:        L_l1 ← L1_Loss(predictions.bboxes, gt_bboxes)
17:        
18:        // Shape-IoU loss for bounding box regression
19:        L_shape ← Shape_IoU_Loss(predictions.bboxes, gt_bboxes)
20:        
21:        L_total ← L_giou + λ_cls × L_cls + λ_l1 × L_l1 + λ_shape × L_shape
22:        
23:        // Backward propagation
24:        Optimizer.zero_grad()
25:        L_total.backward()
26:        Optimizer.step()
27:    end for
28:    
29:    // Validation
30:    mAP ← Validate(M, Val_Dataset)
31:    Print(epoch, L_total, mAP)
32:    
33:    // Learning rate scheduling
34:    Scheduler.step()
35: end for
36: 
37: return M
```

### 2.1 Shape-IoU Loss

```
Algorithm 9: Shape-IoU Loss Calculation
Input: Predicted bbox B_pred, Ground truth bbox B_gt, scale factor s = 1.5
Output: Shape-IoU loss value

1: // Calculate IoU
2: IoU ← Intersection_Over_Union(B_pred, B_gt)
3: 
4: // Calculate shape-adaptive weights
5: w_gt, h_gt ← B_gt.width, B_gt.height
6: ww ← 2 × w_gt^s / (w_gt^s + h_gt^s)
7: hh ← 2 × h_gt^s / (w_gt^s + h_gt^s)
8: 
9: // Calculate shape distance
10: Δx ← (B_pred.center_x - B_gt.center_x)
11: Δy ← (B_pred.center_y - B_gt.center_y)
12: distance_shape ← hh × Δx² + ww × Δy²
13: 
14: // Calculate shape penalty
15: Ω_shape ← ww × (B_pred.width - B_gt.width)² + 
16:              hh × (B_pred.height - B_gt.height)²
17: 
18: // Shape-IoU loss
19: L_Shape_IoU ← 1 - IoU + distance_shape + 0.5 × Ω_shape
20: 
21: return L_Shape_IoU
```

## 3. Notation

- `X`: Input feature map
- `Y`: Output feature map
- `C`: Number of channels
- `H, W`: Feature map height and width
- `K`: Kernel size
- `FFT2D`: 2D Fast Fourier Transform
- `IFFT2D`: 2D Inverse Fast Fourier Transform
- `Conv_k×k`: k×k convolution operation
- `DWConv`: Depth-wise separable convolution
- `Concat`: Feature concatenation
- `NMS`: Non-Maximum Suppression
- `GIoU`: Generalized Intersection over Union
- `⊙`: Element-wise multiplication
- `GAP`: Global Average Pooling

---

**Note**: This pseudocode describes the core algorithmic concepts and workflows of SFD-DETR. The complete implementation will be open-sourced after the paper is accepted.
