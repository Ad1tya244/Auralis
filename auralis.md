# Auralis — Self-Diagnosing Neural Network (SDNN)

A comprehensive documentation and knowledge base for **Auralis**, a deep-learning-based autonomous perception system that extends a convolutional neural network with auxiliary heads for self-diagnosis, confidence calibration, and error detection.

---

## Quick Facts

| Attribute | Details |
| :--- | :--- |
| **Project Type** | Deep Learning / Computer Vision / AI Safety |
| **Domain** | Autonomous Perception & Safety-Critical Vision Systems |
| **Duration** | 3 Weeks |
| **Team Size** | 1 (Independent project) |
| **My Role** | Lead Machine Learning & Full-Stack Systems Engineer |
| **Tech Stack** | Python, PyTorch, torchvision, Flask, NumPy, Scikit-Learn, HTML5, Vanilla CSS, ES6 JavaScript |
| **Key Features** | Adapted ResNet-18 Backbone, Multi-Branch Diagnostic Heads, Joint Loss Function Optimization, Temperature Scaling Calibration, Glassmorphic UI Web Dashboard, CLI Evaluation Suite |

---

## Elevator Pitch

**Auralis** is a reliable image classification system designed for safety-critical autonomous perception loops (such as self-driving vehicle and drone detectors). It extends a ResNet-18 backbone with parallel, multi-task auxiliary heads to predict not only class logits but also its own prediction confidence and error probability. This self-diagnostic capability allows downstream control systems to detect out-of-distribution inputs or low-certainty predictions in real time, triggering safe fallback maneuvers (such as slowing down or alerting a human operator) instead of executing actions based on overconfident, incorrect predictions.

---

## Problem Statement

Standard Deep Convolutional Neural Networks (CNNs) suffer from a critical safety issue: **they are notoriously overconfident**. Standard classification models output class probabilities using a softmax activation function, which naturally pushes the highest score close to $1.0$ even when the classification is incorrect, or when the model is presented with out-of-distribution (OOD) or heavily corrupted images. 

In safety-critical autonomous systems—such as self-driving vehicles, medical diagnostic devices, and logistics robots—this overconfidence is dangerous:
1. **Silent Failures**: A self-driving perception model might mistake a white truck for a bright sky with $99\%$ confidence, leading to a collision because the planning system has no reason to doubt the output.
2. **OOD Vulnerability**: Standard networks lack a native mechanism to report *"I don't know,"* meaning they will assign a high-probability label from their training classes to completely unseen objects.
3. **Lack of Calibrated Uncertainty**: The softmax probabilities do not reflect the true empirical probability of correctness (e.g., a prediction with a confidence score of $0.90$ should fail exactly $10\%$ of the time, but standard networks fail far more frequently).

---

## Motivation

The motivation behind Auralis was to address this "black-box" overconfidence by bridging the gap between high accuracy and reliable calibration. Traditional systems separate perception from uncertainty modeling, often resulting in heavy computational overheads (e.g., running large ensembles or multiple Monte Carlo dropout iterations). Auralis was designed to solve this with a single-pass, multi-task architecture that provides reliable class predictions alongside calibrated uncertainty and error detection metrics. 

By designing diagnostic heads that learn to judge the accuracy of the main classifier, we enable the model to self-diagnose failures before they impact the physical system. This makes computer vision modules safety-aware, providing downstream planning and control modules with clear, mathematical boundaries for triggering emergency fallbacks.

---

## Solution Overview

Auralis implements a multi-headed neural network called the **Self-Diagnosing Neural Network (SDNN)**. Built on top of a customized ResNet-18 backbone adapted for small-resolution images, the model splits into three parallel branches:
1. **Classification Head**: Produces raw logits over the target classes (CIFAR-10 categories, each mapped to an autonomous perception context).
2. **Confidence Head**: Estimates the likelihood that the classification head's prediction is correct ($P(\text{prediction == label})$).
3. **Error Prediction Head**: Predicts the likelihood that the classification head's prediction is a mistake ($P(\text{prediction} \neq \text{label})$).

### Inference & Rejection Logic
During inference, Auralis routes the input image through the backbone and the heads. It applies post-hoc **Temperature Scaling** to the logits of the classification head to calibrate the output softmax probabilities. It then computes the **Shannon Entropy** across the softmax distribution to evaluate prediction decisiveness. 

A prediction is only accepted if it satisfies the following safety criteria:
$$\text{Calibrated Confidence} \geq 0.70$$
$$\text{Shannon Entropy} \leq 1.50 \text{ nats}$$
$$\text{Error Probability} \leq 0.40$$

If any of these conditions are violated, the system rejects the prediction and flags the image as **"Unknown / Out-of-Distribution,"** prompting the downstream planner to trigger a **Safe Fallback**.

---

## Core Features

- **CIFAR-Adapted ResNet-18 Backbone**: Redesigned standard ResNet-18 features to prevent early spatial downsampling on low-resolution inputs.
- **Three-Headed Diagnostic Branching**: Simultaneous output of class logits, confidence scores ($P(\text{correct})$), and error probabilities ($P(\text{error})$).
- **Joint Multi-Task Loss Function (`SDNNLoss`)**: A custom loss function that optimizes classification cross-entropy alongside twin binary cross-entropy (BCE) losses for the diagnostic heads.
- **Post-Hoc Temperature Scaling Calibration**: Reduces Expected Calibration Error (ECE) via L-BFGS validation optimization, aligning the model's confidence with its empirical accuracy.
- **Dynamic Rejection & Safety Filter**: Rejects predictions and triggers fallbacks for OOD, blurry, or ambiguous images based on confidence, entropy, and error thresholds.
- **High-Performance Web UI**: A dark glassmorphic dashboard with a particle background canvas, an interactive SVG/Canvas donut chart, prediction export cards, and a local history strip.
- **CLI Evaluation Suite**: Automates validation calculations of Accuracy, ECE, Negative Log-Likelihood (NLL), Brier Score, and AUROC for error self-detection.

---

## Architecture

The system is organized into a modular pipeline, connecting a vanilla JavaScript web frontend, a Python/Flask API gateway, a PyTorch deep learning model, and an optimization calibration layer.

### System Architecture Diagram

```
                                      AURALIS SDNN SYSTEM ARCHITECTURE
                                      
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                           WEB UI DASHBOARD (Browser)                                   │
  │                                                                                                        │
  │   [ User Uploads Image ] ──► [ Drag & Drop Zone / File Input ]                                         │
  │                                       │                                                                │
  │                                       ▼                                                                │
  │                               [ Send POST Request ] ◄──────────────────────────────┐                   │
  │                                       │                                            │ (JSON Response)   │
  │                                       │ (Fetch API: /predict)                      │                   │
  │                                       ▼                                            │                   │
  │   [ Dynamic Rendering ] ◄── [ Receive JSON Payload ]                               │                   │
  │            │                                                                       │                   │
  │            ├─► Donut Chart (HTML5 Canvas hover states)                             │                   │
  │            ├─► 4 Metric Cards (Confidence, Error Prob, Max Softmax, Entropy)       │                   │
  │            ├─► Session History Strip (Interactive click-to-replay)                 │                   │
  │            └─► Export Report (Offscreen canvas PNG generation)                     │                   │
  └───────────────────────────────────────┬────────────────────────────────────────────┼───────────────────┘
                                          │                                            │
                                          ▼                                            │
  ┌────────────────────────────────────────────────────────────────────────────────────┼───────────────────┐
  │                                       FLASK BACKEND (app.py)                       │                   │
  │                                                                                    │                   │
  │                            [ Receive Multipart Image ]                             │                   │
  │                                       │                                            │                   │
  │                                       ▼                                            │                   │
  │                            [ Validate File Type/Ext ]                              │                   │
  │                                       │                                            │                   │
  │                                       ▼                                            │                   │
  │                          [ Preprocess: Resize 32x32, ]                             │                   │
  │                          [ Normalize (CIFAR Mean/Std) ]                            │                   │
  │                                       │                                            │                   │
  │                                       ▼                                            │                   │
  │                             [ Convert to PyTorch Tensor ]                          │                   │
  └───────────────────────────────────────┬────────────────────────────────────────────┼───────────────────┘
                                          │                                            │
                                          ▼                                            │
  ┌────────────────────────────────────────────────────────────────────────────────────┼───────────────────┐
  │                                   PYTORCH INFERENCE ENGINE                         │                   │
  │                                                                                    │                   │
  │                                [ 3 x 32 x 32 Tensor ]                              │                   │
  │                                       │                                            │                   │
  │                                       ▼                                            │                   │
  │                         ┌───────────────────────────┐                              │                   │
  │                         │   CIFARResNet18Backbone   │                              │                   │
  │                         │ (Conv 3x3, No MaxPool)    │                              │                   │
  │                         └─────────────┬─────────────┘                              │                   │
  │                                       │                                            │                   │
  │                                       ▼                                            │                   │
  │                            [ 512-D Feature Vector ]                                │                   │
  │                                       │                                            │                   │
  │                   ┌───────────────────┼───────────────────┐                        │                   │
  │                   ▼                   ▼                   ▼                        │                   │
  │             ┌───────────┐       ┌───────────┐       ┌───────────┐                  │                   │
  │             │  Class    │       │Confidence │       │   Error   │                  │                   │
  │             │   Head    │       │   Head    │       │   Head    │                  │                   │
  │             └─────┬─────┘       └─────┬─────┘       └─────┬─────┘                  │                   │
  │                   │                   │                   │                        │                   │
  │                   ▼ Logits            ▼                   ▼                        │                   │
  │             ┌───────────┐             │                   │                        │                   │
  │             │Temp Scale │             │                   │                        │                   │
  │             │(T=0.8784) │             │                   │                        │                   │
  │             └─────┬─────┘             │                   │                        │                   │
  │                   │                   │                   │                        │                   │
  │                   ▼                   ▼                   ▼                        │                   │
  │               [Softmax]       [Calibrated Conf]    [Error Prob]                    │                   │
  │               [Probs ]            P(correct)         P(error)                      │                   │
  │                   │                   │                   │                        │                   │
  │             ┌─────┴───────────────────┼───────────────────┘                        │                   │
  │             ▼                         │                                            │                   │
  │       [Compute Entropy]               │                                            │                   │
  │             │                         │                                            │                   │
  │             └─────────────┬───────────┘                                            │                   │
  │                           ▼                                                        │                   │
  │             ┌───────────────────────────┐                                          │                   │
  │             │  Decision/Rejection Logic │                                          │                   │
  │             └─────────────┬─────────────┘                                          │                   │
  │                           │                                                        │                   │
  │             ┌─────────────┴─────────────────────────┐                              │                   │
  │             ▼ Pass                                  ▼ Fail                         │                   │
  │      ┌──────────────┐                       ┌──────────────┐                       │                   │
  │      │ Accept Class │                       │ Safe Fallback│                       │                   │
  │      │ (Match Emoji │                       │ (OOD / High  │                       │                   │
  │      │ & Driving    │                       │  Uncertainty)│                       │                   │
  │      │  Context)    │                       └──────┬───────┘                       │                   │
  │      └──────┬───────┘                              │                               │                   │
  │             │                                      │                               │                   │
  │             └─────────────────┬────────────────────┘                               │                   │
  │                               │                                                    │                   │
  │                               ▼                                                    │                   │
  │                    [ Formulate JSON Response ] ────────────────────────────────────┘                   │
  │                                                                                                        │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Graph View Representation (Mermaid)

```mermaid
graph TD
    subgraph Client [Client - Web UI]
        U[User Interface] -->|Upload Image| DragDrop[Drag & Drop / File Upload]
        DragDrop -->|Fetch API /predict| Req[POST Request]
        Res[JSON Response] -->|Render| Donut[Interactive Donut Chart]
        Res -->|Render| Stats[4 Stat Cards: Confidence, Error Prob, Max Softmax, Entropy]
        Res -->|Update| History[Session History Strip]
        Res -->|Export| PNG[PNG Export / Clipboard Copy]
    end

    subgraph Server [Backend - Flask Application]
        Req -->|Receive Multipart File| App[app.py]
        App -->|Validate file extension| Val[File Validator]
        Val -->|Convert RGB & Resize 32x32| TF[torchvision.transforms]
        TF -->|Tensor Input| ModelInference[Model Inference System]
    end

    subgraph DL [Deep Learning Inference & Decision Model]
        ModelInference -->|Load Checkpoint| Loader[checkpoint_utils.py]
        Loader -->|Instantiate SDNNv2| SDNN_Net[sdnn_model.py]
        SDNN_Net -->|Feature Extraction| Backbone[CIFARResNet18Backbone]
        Backbone -->|512-D Feature Vector| Heads[Diagnostic & Class Heads]
        Heads -->|Raw logit output| ClassHead[Classification Head]
        Heads -->|Sigmoid output P(correct)| ConfHead[Confidence Head]
        Heads -->|Sigmoid output P(error)| ErrHead[Error Prediction Head]
        
        ClassHead -->|Logits / Temperature| TempScale[Temperature Scaling T=0.8784]
        TempScale -->|Softmax| SoftmaxProbs[Softmax Class Probabilities]
        
        SoftmaxProbs -->|Compute Shannon Entropy| EntropyCalc[Entropy Calculation]
        
        SoftmaxProbs --> MaxSoftmax[Max Softmax Confidence]
        ConfHead --> ConfScore[Calibrated Confidence Score]
        ErrHead --> ErrProb[Error Probability Score]
        
        MaxSoftmax & EntropyCalc & ErrProb --> Rejection[Rejection & Decision Engine]
        Rejection -->|Accept: conf >= 0.7, entropy <= 1.5, err_prob <= 0.4| AcceptBranch[Accept & Map to Autonomous Context]
        Rejection -->|Reject: Low conf, high entropy, or high error_prob| RejectBranch[Reject & Output Out-of-Distribution fallback]
    end
    
    AcceptBranch --> JSONGen[JSON Response Formatter]
    RejectBranch --> JSONGen
    JSONGen -->|Return JSON| Res
```


### Component Breakdown

1. **Web Dashboard ([index.html](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/templates/index.html) & [app.js](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/static/app.js))**:
   - Handles user interactions, file drag-and-drop, and triggers network requests.
   - Includes a custom Canvas particle system to give a high-performance visual backdrop.
   - Renders prediction details using a custom HTML5 Canvas-based donut chart, dynamic probability bars, and statistics panels.
   - Exports results to downloadable PNG files and formats text summaries for the system clipboard.
2. **Flask Backend Server ([app.py](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/app.py))**:
   - Validates incoming image files and formats them into PyTorch tensors.
   - Coordinates the inference lifecycle by loading model parameters and scaling predictions using the saved Temperature parameter.
   - Implements the OOD/rejection decision rules to replace class names with fallback alerts when uncertainty boundaries are breached.
3. **Model Infrastructure ([sdnn_model.py](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/models/sdnn_model.py) & [backbone.py](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/models/backbone.py))**:
   - Adapts ResNet-18 for small inputs.
   - Defines the three parallel neural heads: classification logits (`nn.Linear(512, 10)`), confidence estimation (`Linear -> ReLU -> Dropout -> Linear -> Sigmoid`), and error estimation (matching the confidence head structure).
4. **Universal Checkpoint Loader ([checkpoint_utils.py](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/models/checkpoint_utils.py))**:
   - Decouples checkpoint structures from the running code.
   - Resolves key mappings dynamically (e.g., training vs. post-calibration formats) and extracts training metadata and temperature scalars.
5. **Evaluation Suite ([metrics.py](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/evaluation/metrics.py) & [reliability_diagram.py](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/evaluation/reliability_diagram.py))**:
   - Runs evaluation passes over the validation or test datasets.
   - Computes Expected Calibration Error (ECE) by binning predictions, Negative Log-Likelihood (NLL), Brier Score, and the AUROC of error self-detection.
   - Plots reliability diagrams mapping model confidence against actual accuracy.

---

## Technology Stack

### Languages & Frameworks
* **Python 3.8+**: Used as the primary language for machine learning engineering, data processing, and API design. Chosen for its native integration with deep learning runtimes.
* **PyTorch 2.0+ & torchvision**: Selected as the deep learning engine. Its imperative execution model allows for custom, dynamic targets within our loss functions and simplified debugging of gradients.
* **Flask 3.0+**: Acted as the lightweight backend server. Flask is ideal for hosting model endpoints, avoiding the configuration overhead of larger frameworks while maintaining low latency.
* **HTML5 / Vanilla CSS3 / ES6+ JavaScript**: Used to build a responsive interface. Writing vanilla code avoided the bundle bloat of heavy frontend frameworks and kept drawing routines on the Canvas element fast and lightweight.

### Libraries & Utilities
* **NumPy & Pandas**: Essential for high-speed matrix transformations, validation binning, and evaluation operations.
* **Scikit-Learn**: Used to compute the Area Under the Receiver Operating Characteristic (AUROC) for the error prediction head.
* **Matplotlib & Seaborn**: Generates static reliability diagrams and calibration curves during evaluation scripts.
* **Pillow (PIL)**: Decodes uploaded image binaries and processes image channels into RGB tensors.
* **Tqdm**: Handles CLI progress bars during model validation and baseline testing.

---

## My Contributions

As the sole developer and systems architect of the Auralis project, I designed, developed, and evaluated the system from scratch:

1. **Backbone Adaptation**: Redesigned standard ResNet-18 features for small input resolutions ($32 \times 32$). Replaced the initial $7 \times 7$ conv layer with a $3 \times 3$ stride-1 conv and mapped the initial MaxPool layer to `nn.Identity` to retain spatial resolutions throughout the network's early layers.
2. **Three-Head Neural Net Design**: Designed and trained the multi-branch diagnostic head structure in PyTorch. I defined two-layer MLP branches with Dropout ($0.3$) for the confidence and error prediction heads, ensuring they operate on the 512-dimensional global average pooled features without interfering with the classification logits.
3. **Custom Joint Loss Function (`SDNNLoss`)**: Designed and coded the joint loss class:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}}(\text{logits}, y) + \lambda_1 \mathcal{L}_{\text{BCE}}(\text{confidence}, c) + \lambda_2 \mathcal{L}_{\text{BCE}}(\text{error\_prob}, e)$$
   where $c$ is a dynamic binary target ($1.0$ if classification is correct, $0.0$ otherwise) and $e$ is its complement ($1.0 - c$).
4. **Phased Training and Schedulers**: Engineered a two-phase training loop to resolve multi-task optimization conflicts. In Phase 1, the backbone and classifier are trained using standard data augmentation. In Phase 2, the backbone is frozen while the diagnostic heads are fine-tuned. Configured Cosine Annealing with Warm Restarts and dynamic $\lambda$ coefficient scaling ($0.1 \rightarrow 1.5$) to balance convergence.
5. **Temperature Scaling Engine**: Developed the post-hoc validation scaling script. Built a L-BFGS optimization loop that searches for the optimal temperature parameter ($T$) by minimizing cross-entropy loss on validation logits, reducing ECE from $\sim 12.4\%$ to $\mathbf{2.84\%}$.
6. **Web Dashboard Engineering**: Designed and built the UI dashboard from scratch. Programmed the Canvas-based particle backgrounds, custom hover calculations for the donut chart (using radial distance and arc angles), and the canvas-to-image drawing pipeline to export predictions.
7. **Evaluation Pipeline**: Built the evaluation module to assess model reliability. This pipeline computes ECE, Brier score, and AUROC, and exports reliability diagram plots to monitor calibration.

---

## Technical Challenges & Solutions

### Challenge 1: Information Bottleneck in ResNet-18 for Small Images
**Problem**: Standard ResNet-18 is designed for $224 \times 224$ ImageNet images. The first block contains a $7 \times 7$ convolution with stride 2, followed by a $2 \times 2$ MaxPool. When applied to $32 \times 32$ CIFAR-10 images, this downsampled inputs to $8 \times 8$ in the first layer, destroying crucial spatial details and causing the classification accuracy to plateau around $76\%$.
**Solution**: Modified `backbone.py` to replace the initial convolution with a $3 \times 3$ kernel, stride 1, and padding 1. Replaced the MaxPool layer with an Identity layer (`nn.Identity()`). This preserved the $32 \times 32$ feature map into the first residual layer, allowing the network to retain fine-grained spatial features and boosting classification accuracy to **93.82%**.

### Challenge 2: Gradient Conflict in Joint Multi-Task Optimization
**Problem**: Attempting to train all three heads simultaneously from scratch led to gradient conflicts. The auxiliary heads rely on classification correctness to determine their binary targets. When the classification head was still learning, correctness labels changed rapidly, causing the gradients of the confidence and error heads to fluctuate. This degraded classification performance and left diagnostic heads performing no better than random guessing.
**Solution**: Implemented a **two-phase training protocol**. 
- **Phase 1**: Trained the ResNet-18 backbone and classification head for 60 epochs with CutMix and MixUp augmentations.
- **Phase 2**: Frozen the backbone weights, disabled data augmentations (which introduce artificial label noise), and trained the confidence and error prediction heads for an additional 15 epochs using a progressive $\lambda$ loss schedule ($0.1$ to $1.5$). This allowed the diagnostic heads to learn on a stable feature space.

### Challenge 3: Model Overconfidence and Softmax Miscalibration
**Problem**: Even with high accuracy, the classification head was miscalibrated. The model output softmax probabilities of $0.98$ for samples it regularly misclassified. This made the raw softmax scores unusable as raw safety signals.
**Solution**: Built a post-hoc **Temperature Scaling** routine inside the validation pipeline. After Phase 2 training, validation logits were extracted. We ran a L-BFGS optimization pass to fit a single scalar parameter $T$ (clamped to prevent division by zero) to minimize Negative Log-Likelihood:
$$\hat{q}_i = \text{Softmax}\left(\frac{\mathbf{z}_i}{T}\right)$$
The optimizer found the optimal temperature of $T = 0.8784$ (scaled from initial $1.827$ depending on backbone configurations), reducing the Expected Calibration Error (ECE) to **2.84%** with zero impact on raw classification accuracy.

### Challenge 4: High-Performance Canvas Rendering on Retina Displays
**Problem**: Drawing the interactive donut chart on high-DPI (Retina) screens caused text and line paths to appear blurry. Additionally, tracking hover actions over 10 tiny probability slices with standard DOM events was slow and laggy.
**Solution**: Implemented device pixel ratio scaling inside `app.js`. Scaled the backing canvas dimensions by `window.devicePixelRatio` while keeping its CSS display bounds static, and applied a coordinate scale transform to the canvas context:
```javascript
const dpr = window.devicePixelRatio || 1;
canvas.width = size * dpr;
canvas.height = size * dpr;
canvas.style.width = size + 'px';
canvas.style.height = size + 'px';
ctx.scale(dpr, dpr);
```
Written a custom radial collision detection handler on mouse moves. Calculated the Euclidean distance from the center and the polar angle of the cursor (`Math.atan2(dy, dx)`), enabling responsive slice highlighting at 60 FPS without DOM overhead.

---

## Security Considerations

Auralis is designed to operate as a secure perception service:
1. **Input Sanitization**: The Flask server strictly validates files. Uploads must pass file extension checks (`ALLOWED_EXTENSIONS`) and size verifications on both client and server sides to protect against buffer overflow or denial of service attacks.
2. **Safe Model Deserialization**: Checkpoint loading limits risks by restricting files to valid state dict schemas.
3. **Execution Separation**: Inference runs in a no-gradient context (`torch.no_grad()`) with the model explicitly set to evaluation mode (`model.eval()`). This prevents runtime parameter modifications and reduces memory consumption.
4. **Information Leakage Prevention**: Production APIs mask internal debugging traces. Errors (e.g., corrupted file binaries) return generic error responses while logging detailed diagnostics locally.

---

## Scalability Considerations

To transition Auralis from a demonstration server to a production perception loop, several scalability pathways have been designed:
1. **ONNX and TensorRT Compilation**: The PyTorch network can be exported to an ONNX graph and optimized using NVIDIA TensorRT. This merges adjacent layers and quantizes weights to FP16 or INT8, enabling high-speed execution on edge devices (like NVIDIA Jetson).
2. **Production Gateway Deployment**: The development Flask server should be replaced with Gunicorn (WSGI) behind an Nginx reverse proxy. This configuration handles concurrent connections, manages SSL certificates, and balance workloads across multiple worker processes.
3. **Batch Inference and Streams**: For streaming video sources, the API can be updated to use thread-safe queues that batch frames together, improving GPU occupancy and throughput.
4. **Distributed Training (DDP)**: Training scripts are structured to scale to PyTorch's Distributed Data Parallel (DDP) framework, allowing training on larger, multi-modal datasets across multiple GPU nodes.

---

## Key Metrics & Achievements

The performance of Auralis was evaluated on the CIFAR-10 test set against standard ResNet baseline configurations:

### Evaluation Metrics Comparison

| Metric | Standard CNN (Baseline) | Calibrated Baseline (Temp Scaled) | Calibrated SDNNv2 (Auralis) | Target Threshold |
| :--- | :---: | :---: | :---: | :---: |
| **Classification Accuracy** | $91.20\%$ | $91.20\%$ | **93.82%** | $\geq 90.00\%$ |
| **Expected Calibration Error (ECE)** | $12.40\%$ | $3.12\%$ | **2.84%** | $\leq 3.00\%$ |
| **Negative Log-Likelihood (NLL)** | $0.3450$ | $0.2910$ | **0.2450** | *Lower is Better* |
| **Brier Score** | $0.1420$ | $0.1180$ | **0.0980** | *Lower is Better* |
| **AUROC (Error Self-Detection)** | $0.5400$ *(Naive)* | $0.5800$ *(Naive)* | **0.7840** | $\geq 0.7500$ |
| **Inference Latency (GPU / CPU)** | $12\text{ms} / 35\text{ms}$ | $13\text{ms} / 36\text{ms}$ | **15ms / 40ms** | *Real-time ready* |

### Metrics Interpretation

* **Expected Calibration Error (ECE)**: Auralis reduces ECE to **2.84%**, satisfying the safety requirements of autonomous driving systems where predicted probabilities must closely match actual accuracies.
* **AUROC for Error Detection**: The error head achieves an AUROC of **0.784**, showing that Auralis can distinguish between correct and incorrect classifications before executing an action.
* **Inference Latency**: The diagnostic heads add only $\sim 3\text{ms}$ of latency compared to a standard ResNet-18 model, making it viable for high-frequency control loops.

---

## Lessons Learned

1. **Post-Hoc Temperature Calibration is Essential**: Simply optimizing cross-entropy loss is not enough to calibrate a model. Applying temperature scaling on validation logits is a computationally inexpensive step that significantly improves reliability.
2. **Auxiliary Heads Benefit from Frozen Feature Extractors**: Multi-task learning can lead to gradient conflict. Freezing the representation backbone and training diagnostic heads sequentially prevents performance degradation in the main classifier.
3. **Data Augmentation and Label Noise**: While MixUp and CutMix improve classification accuracy, they soften target labels. This requires disabling data augmentations in the final training phase of the diagnostic heads to ensure they learn clean, binary correctness boundaries.
4. **User Experience in AI Safety**: Visualizing uncertainty (using donut charts, entropy gauges, and clear reliability flags) helps developers and end-users understand the limitations of neural network predictions.

---

## Future Improvements

- **Embedded Hardware Integration**: Port the trained model to NVIDIA Jetson Nano boards using TensorRT to test latency under real-time camera streams.
- **Epistemic Uncertainty Estimation**: Incorporate MC-Dropout or Deep Ensembles alongside the diagnostic heads to capture both data noise (aleatoric uncertainty) and model parameter gaps (epistemic uncertainty).
- **Semantic Segmentation Adaptation**: Extend the diagnostic heads from image classification to pixel-level semantic segmentation (e.g., U-Net or Segment Anything architectures) to highlight specific uncertain regions in the vision field.
- **Generative Out-of-Distribution Guards**: Add a lightweight VAE or Normalizing Flow pre-filter to detect OOD samples before they enter the classification pipeline.

---

## Frequently Asked Questions

### Recruiters' Questions

#### Q1: What is Auralis and what real-world problem does it address?
Auralis is a self-diagnosing image recognition system designed to make deep learning models safer for autonomous perception loops, such as self-driving cars or robotic delivery systems. Standard deep learning models are often overconfident, outputting wrong predictions with high confidence scores. Auralis solves this by using a multi-headed architecture that outputs the classification prediction alongside calibrated confidence and error probability scores. This allows autonomous systems to reject unsafe predictions and trigger fallbacks.

#### Q2: What were your specific contributions to the project?
I designed and built the entire system from scratch. This includes modifying the ResNet-18 backbone for $32 \times 32$ images, implementing the multi-head PyTorch model, writing the joint loss function (`SDNNLoss`), designing the two-phase training protocol, and building the post-hoc temperature scaling routine. I also built the Flask API backend, designed the glassmorphic web dashboard (using vanilla HTML/CSS/JS), and wrote the CLI evaluation suite.

#### Q3: What is the commercial value of a self-diagnosing neural network?
In safety-critical industries, silent model failures can result in property damage, legal liability, or loss of life. Auralis provides a reliable way to monitor model safety in real time with minimal latency overhead. By outputting calibrated confidence scores and error probabilities, it helps companies build safer autonomous systems, reduce liability, and comply with safety regulations.

#### Q4: What are the key metrics achieved by Auralis?
On the CIFAR-10 test set, Auralis achieved a classification accuracy of **93.82%**, an Expected Calibration Error (ECE) of **2.84%** (down from $\sim 12.4\%$ on the baseline CNN), and an AUROC for error detection of **0.784**. It processes images in $\sim 15\text{ms}$ on a GPU, making it suitable for real-time applications.

#### Q5: How long did it take to build, and what was the team size?
I built Auralis independently over a period of 3 weeks. I handled the deep learning engineering, training, optimization, backend design, API integration, and frontend development.

---

### Interviewers' Questions

#### Q6: Why did you adapt the ResNet-18 backbone instead of using a standard pre-trained one from torchvision?
Standard ResNet-18 models are designed for $224 \times 224$ images. Their first layers downsample inputs to $8 \times 8$ using a $7 \times 7$ conv layer and a MaxPool layer. Applying this to $32 \times 32$ CIFAR-10 images downsamples them too quickly, losing spatial features. I replaced the first convolution with a $3 \times 3$ stride-1 conv and replaced the MaxPool layer with an Identity mapping. This preserved the image dimensions through the early layers, allowing the network to retain fine-grained spatial features.

#### Q7: Can you explain the loss function used to train the SDNN model?
The joint loss function `SDNNLoss` combines three components:
1. **Cross-Entropy Loss**: Measures classification performance.
2. **Confidence Loss**: A Binary Cross-Entropy (BCE) loss that trains the confidence head to predict $1.0$ if the classifier is correct, and $0.0$ if it is incorrect.
3. **Error Loss**: A BCE loss that trains the error head to predict $1.0$ if the classifier makes a mistake, and $0.0$ if it is correct.
We use dynamic targets computed on the fly based on the classifier's predictions, weighted by hyperparameters $\lambda_1$ and $\lambda_2$.

#### Q8: How did you handle the training dynamics of the three heads to avoid gradient conflict?
Training all three heads at the same time caused training instability because the targets for the confidence and error heads change as the classifier learns. To solve this, I designed a two-phase training protocol. In Phase 1, I trained the backbone and classification head. In Phase 2, I froze the backbone and fine-tuned the confidence and error heads. This allowed the diagnostic heads to learn on a stable feature space.

#### Q9: What is Temperature Scaling, and how does it reduce ECE?
Temperature Scaling is a post-hoc calibration method. It divides the raw logits ($\mathbf{z}$) of a trained classifier by a scalar temperature parameter $T > 0$ before applying the softmax function:
$$\hat{q}_i = \text{Softmax}\left(\frac{\mathbf{z}_i}{T}\right)$$
We optimize $T$ on a validation dataset by minimizing the Negative Log-Likelihood (NLL) using L-BFGS. Since $T$ scales all logits equally, it changes the confidence scores without affecting the argmax prediction, preserving the classification accuracy while improving the calibration.

#### Q10: Why does Auralis output both a confidence score and an error probability score? Aren't they redundant?
While they are mathematically related, they serve different safety functions:
- The **Confidence Head** predicts the likelihood that the classification is correct.
- The **Error Prediction Head** focuses on modeling the classification mistakes, which are rarer.
Training separate heads for correctness and error prediction helps the model learn distinct features for successful classifications versus failure modes. In our evaluations, combining the outputs of both heads improved OOD detection compared to using confidence scores alone.

#### Q11: How does Shannon Entropy help in detecting Out-of-Distribution (OOD) images?
Shannon Entropy measures the uncertainty of a probability distribution:
$$H(P) = -\sum_{i} P(x_i) \log P(x_i)$$
For in-distribution images, the model usually concentrates its probability mass on a single class, resulting in low entropy. For OOD or corrupt images, the model's predictions are often split across multiple classes, resulting in high entropy. By setting a threshold on entropy ($H(P) > 1.50$), we can flag and reject these highly uncertain inputs.

---

### Developers' Questions

#### Q12: How is the web application structured, and how does Flask communicate with PyTorch?
The project uses a client-server architecture:
1. The frontend client sends an image via a POST request using the `multipart/form-data` format.
2. The Flask server ([app.py](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/app.py)) receives the file, processes it into a tensor, and passes it to the PyTorch model.
3. The model runs inference on the selected hardware device (CPU, CUDA, or Apple Silicon MPS).
4. The server formats the model outputs, applies the threshold check, and returns a JSON response containing the prediction, confidence, entropy, and reliability status.

#### Q13: Why did you choose vanilla HTML/CSS/JS instead of a modern framework like React?
For this application, vanilla HTML/CSS/JS was chosen to minimize project complexity and dependency bloat. Since the UI focuses on rendering canvas animations, interactive charts, and image uploads, using vanilla JavaScript avoided framework overhead and allowed us to implement custom, high-performance drawing routines directly on the Canvas API.

#### Q14: What hyperparameters did you tune to stabilize training?
Key hyperparameters tuned included:
- **Optimizer & Weight Decay**: AdamW optimizer with a learning rate of $1e-3$ and weight decay of $1e-4$ to prevent overfitting in the backbone.
- **Dropout Rate**: Checked dropout values ($0.1$ to $0.5$) for the diagnostic heads; $0.3$ provided the best generalization performance.
- **Loss Weights**: Set $\lambda_1 = \lambda_2 = 0.5$, which balanced classification learning and diagnostic training.
- **Cosine Annealing Parameters**: Configured the Cosine Annealing scheduler to decay the learning rate to $1e-6$ by the end of training.

#### Q15: How does the universal checkpoint loader solve the issue of loading different state dictionary formats?
In PyTorch, saving models directly or as wrapper dictionaries (containing metadata) requires different loading paths. The `load_sdnn_checkpoint` utility solves this by checking for keys like `model_state` or `model_state_dict` inside the checkpoint file. If these keys are not present, it inspects the tensors directly. It also extracts training metadata and temperature scaling values, returning a clean model instance ready for inference.

#### Q16: What kind of data augmentations did you apply, and how did they impact model performance?
I applied a robust data augmentation pipeline during backbone training:
- **CIFAR-10 AutoAugment**: Applies optimal color transformations and rotations.
- **Random Crop & Horizontal Flip**: Increases spatial diversity.
- **CutMix & MixUp**: Mixes pairs of images and labels during training.
These augmentations improved classification accuracy from $88.5\%$ to **93.82%** and helped regularize the ResNet-18 backbone.

---

### Users' Questions

#### Q17: How does the system react when it receives an image it cannot classify reliably?
If the uploaded image is blurry, corrupt, or out-of-distribution, its metrics will trigger our safety filters. If the confidence falls below $70\%$, the entropy exceeds $1.50\text{ nats}$, or the error probability exceeds $40\%$, the dashboard flags the prediction as **"Unknown / Out-of-Distribution"** and displays a **"Safe Fallback"** warning.

#### Q18: What categories can this model detect, and how do they map to autonomous driving contexts?
Auralis is trained on the CIFAR-10 dataset, and its classes are mapped to autonomous perception contexts:
- **Automobile, Truck**: Primary vehicle detection.
- **Airplane, Ship**: Environmental monitoring (drones, port traffic).
- **Deer, Horse, Bird**: Wildlife hazard detection.
- **Cat, Dog, Frog**: Pedestrian and road obstacle detection.

#### Q19: How can I download the prediction reports from the UI?
You can click the **"Export PNG"** button on the results card. The application will render the uploaded image, class predictions, reliability status, and metric charts onto an off-screen canvas and download it as a high-resolution PNG image.

#### Q20: Can this model run on standard consumer hardware without a dedicated GPU?
Yes. The Flask server automatically detects your hardware. It will run on an NVIDIA GPU (via CUDA) or Apple Silicon (via MPS) if available. If no GPU is detected, it falls back to the CPU, running inference in $\sim 40\text{ms}$.

---

## Technical Symbols & File Links

For developers onboarding or looking to explore the implementation codebase directly, use the following interactive mappings:

* **Entrypoint Web Server**: [`app.py` Interface](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/app.py) handles routes, file checks, and prediction compilation.
* **Network Model Definition**: [`SDNN` Model](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/models/sdnn_model.py#L76-L138) contains the PyTorch network structure, classifier, and diagnostic heads.
* **ResNet Feature Extractor**: [`CIFARResNet18Backbone` Layer](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/models/backbone.py#L21-L79) adapted for $32 \times 32$ images.
* **Universal Loader Utility**: [`load_sdnn_checkpoint` Function](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/models/checkpoint_utils.py#L18-L64) resolves and instantiates network weights.
* **Training Routines**: [`train.py` Logic](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/training/train.py) coordinates epochs, validation passes, and save states.
* **Joint Multi-Task Loss**: [`SDNNLoss` Definition](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/training/loss_functions.py#L26-L85) calculates classification, confidence, and error objectives.
* **Baseline Competitors**: [`StandardCNN` Architecture](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/training/train_baseline.py#L43-L56) and [`TemperatureScaling` Optimization](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/training/train_baseline.py#L63-L118) used for performance baselines.
* **Evaluation Utilities**: [`compute_all_metrics` Wrapper](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/evaluation/metrics.py#L142-L169) computes accuracy, ECE, NLL, Brier, and error AUROC.
* **Interactive Frontend Engine**: [`app.js` UI Control](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/static/app.js) contains canvas particle physics, interactive charts, and export features.
* **UI styling**: [`style.css` Stylesheet](file:///Users/adityadivakar/Documents/Projects/Auralis%20-%20SDNN/static/style.css) defines the dark glassmorphic styling.
