from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import io
import base64
import json
from pathlib import Path


app = FastAPI(title="Explain My Model API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ImageNet classes
with open(Path(__file__).parent / "imagenet-simple-labels.json", "r") as f:
    IMAGENET_CLASSES = json.load(f)

# Available models
AVAILABLE_MODELS = {
    'resnet50': {'name': 'ResNet-50', 'layers': ['layer1', 'layer2', 'layer3', 'layer4']},
    'resnet18': {'name': 'ResNet-18', 'layers': ['layer1', 'layer2', 'layer3', 'layer4']},
    'resnet34': {'name': 'ResNet-34', 'layers': ['layer1', 'layer2', 'layer3', 'layer4']},
    'resnet101': {'name': 'ResNet-101', 'layers': ['layer1', 'layer2', 'layer3', 'layer4']},
    'densenet121': {'name': 'DenseNet-121', 'layers': ['features.denseblock1', 'features.denseblock2', 'features.denseblock3', 'features.denseblock4']},
    'densenet169': {'name': 'DenseNet-169', 'layers': ['features.denseblock1', 'features.denseblock2', 'features.denseblock3', 'features.denseblock4']},
    'mobilenet_v2': {'name': 'MobileNet V2', 'layers': ['features.3', 'features.6', 'features.13', 'features.17']},
    'mobilenet_v3_large': {'name': 'MobileNet V3 Large', 'layers': ['features.3', 'features.6', 'features.12', 'features.16']},
    'efficientnet_b0': {'name': 'EfficientNet-B0', 'layers': ['features.2', 'features.4', 'features.6', 'features.8']},
}

# Cache for loaded models
model_cache = {}

def get_model(model_name):
    """Get or load a model from cache"""
    if model_name not in model_cache:
        if model_name == 'resnet50':
            model_cache[model_name] = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet18':
            model_cache[model_name] = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif model_name == 'resnet34':
            model_cache[model_name] = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif model_name == 'resnet101':
            model_cache[model_name] = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif model_name == 'densenet121':
            model_cache[model_name] = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        elif model_name == 'densenet169':
            model_cache[model_name] = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        elif model_name == 'mobilenet_v2':
            model_cache[model_name] = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        elif model_name == 'mobilenet_v3_large':
            model_cache[model_name] = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        elif model_name == 'efficientnet_b0':
            model_cache[model_name] = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            model_cache[model_name] = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        model_cache[model_name].eval()
    
    return model_cache[model_name]

def get_target_layer(model, model_name, layer_id):
    """Get the target layer from model"""
    if model_name in ['resnet50', 'resnet18', 'resnet34', 'resnet101']:
        layer_map = {
            'layer1': model.layer1[-1],
            'layer2': model.layer2[-1],
            'layer3': model.layer3[-1],
            'layer4': model.layer4[-1]
        }
        return layer_map.get(layer_id, model.layer4[-1])
    
    elif model_name in ['densenet121', 'densenet169']:
        layer_map = {
            'features.denseblock1': model.features.denseblock1,
            'features.denseblock2': model.features.denseblock2,
            'features.denseblock3': model.features.denseblock3,
            'features.denseblock4': model.features.denseblock4
        }
        return layer_map.get(layer_id, model.features.denseblock4)
    
    elif model_name == 'mobilenet_v2':
        layer_map = {
            'features.3': model.features[3],
            'features.6': model.features[6],
            'features.13': model.features[13],
            'features.17': model.features[17]
        }
        return layer_map.get(layer_id, model.features[17])
    
    elif model_name == 'mobilenet_v3_large':
        layer_map = {
            'features.3': model.features[3],
            'features.6': model.features[6],
            'features.12': model.features[12],
            'features.16': model.features[16]
        }
        return layer_map.get(layer_id, model.features[16])
    
    elif model_name == 'efficientnet_b0':
        layer_map = {
            'features.2': model.features[2],
            'features.4': model.features[4],
            'features.6': model.features[6],
            'features.8': model.features[8]
        }
        return layer_map.get(layer_id, model.features[8])
    
    return None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GradCAM:
    """Grad-CAM implementation"""
    def __init__(self, model, target_layer, model_name):
        self.model = model
        self.target_layer = target_layer
        self.model_name = model_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass - handle VGG separately
        if self.model_name == 'vgg16':
            # VGG has separate features and classifier
            # Pass through features (this will trigger our hooks)
            features = self.model.features(input_tensor)
            # Apply adaptive average pooling
            x = self.model.avgpool(features)
            # Flatten for classifier
            x = torch.flatten(x, 1)
            # Pass through classifier
            output = self.model.classifier(x)
        else:
            # Standard forward pass for other models
            output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Convert to integer if it's a tensor
        if isinstance(target_class, torch.Tensor):
            target_class = target_class.item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Check if gradients were captured
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured. Check hook registration.")
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def cleanup(self):
        """Remove hooks"""
        self.forward_handle.remove()
        self.backward_handle.remove()

def get_gradcam_heatmap(model, image_tensor, original_image, model_name, layer_id):
    """Generate Grad-CAM heatmap overlay"""
    target_layer = get_target_layer(model, model_name, layer_id)
    
    if target_layer is None:
        return None
    
    # Ensure model is in training mode for gradient computation
    model.train()
    
    # Create a new tensor that requires grad
    input_for_gradcam = image_tensor.clone().detach().requires_grad_(True)
    
    # Generate Grad-CAM - pass model_name to handle VGG
    gradcam = GradCAM(model, target_layer, model_name)
    cam = gradcam.generate_cam(input_for_gradcam)
    gradcam.cleanup()
    
    # Set model back to eval mode
    model.eval()
    
    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (original_image.width, original_image.height))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    original_np = np.array(original_image)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
    
    return overlay

def get_activation_maps(model, image_tensor, model_name, layer_id):
    """Extract and visualize activation maps"""
    activations = {}
    
    def hook_fn(module, input, output):
        activations['output'] = output
    
    target_layer = get_target_layer(model, model_name, layer_id)
    
    if target_layer is None:
        return None
    
    # Register hook
    handle = target_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor)
    
    handle.remove()
    
    # Get activations
    acts = activations['output'][0].cpu().numpy()
    
    # Visualize first 16 channels in a grid
    n_channels = min(16, acts.shape[0])
    grid_size = 4
    
    # Create grid
    act_grid = np.zeros((acts.shape[1] * grid_size, acts.shape[2] * grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < n_channels:
                act = acts[idx]
                act = (act - act.min()) / (act.max() - act.min() + 1e-8)
                act_grid[i*acts.shape[1]:(i+1)*acts.shape[1], 
                         j*acts.shape[2]:(j+1)*acts.shape[2]] = act
    
    # Convert to color image
    act_colored = cv2.applyColorMap(np.uint8(255 * act_grid), cv2.COLORMAP_VIRIDIS)
    act_colored = cv2.cvtColor(act_colored, cv2.COLOR_BGR2RGB)
    
    return act_colored

def numpy_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(img_array.astype('uint8'))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    model_name: str = Form("resnet50"),
    layer: str = Form("layer4")
):
    try:
        # Validate model
        if model_name not in AVAILABLE_MODELS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid model: {model_name}"}
            )
        
        # Get model
        model = get_model(model_name)
        
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess for model
        input_tensor = transform(image).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
        
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        predictions = []
        for prob, idx in zip(top5_prob[0], top5_idx[0]):
            class_name = IMAGENET_CLASSES[int(idx)] if int(idx) < len(IMAGENET_CLASSES) else f"Class {int(idx)}"
            predictions.append({
                "class": class_name,
                "confidence": float(prob)
            })
        
        # Generate Grad-CAM with error handling
        try:
            gradcam_overlay = get_gradcam_heatmap(model, input_tensor, image, model_name, layer)
            gradcam_base64 = numpy_to_base64(gradcam_overlay) if gradcam_overlay is not None else None
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            gradcam_base64 = None
        
        # Generate activation maps with error handling
        try:
            activation_maps = get_activation_maps(model, input_tensor, model_name, layer)
            activations_base64 = numpy_to_base64(activation_maps) if activation_maps is not None else None
        except Exception as e:
            print(f"Activation maps error: {e}")
            activations_base64 = None
        
        return JSONResponse({
            "predictions": predictions,
            "gradcam": gradcam_base64,
            "activations": activations_base64,
            "layer": layer,
            "model": model_name
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/")
async def root():
    return {"message": "Explain My Model API is running"}

@app.get("/models")
async def get_models():
    return {
        "models": [
            {
                "id": key,
                "name": value['name'],
                "layers": [
                    {"id": layer, "name": f"Layer {i+1}"}
                    for i, layer in enumerate(value['layers'])
                ]
            }
            for key, value in AVAILABLE_MODELS.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)