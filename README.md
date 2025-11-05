# 🧠 Explain My Model

**Explain My Model** is an interactive web app that visualizes what convolutional neural networks (CNNs) "see" inside images.
Using **Grad-CAM** (Gradient-weighted Class Activation Mapping) and **feature activation maps**, it provides an intuitive view into how deep learning models make decisions.

---

## 🚀 Features

* **Upload any image** (drag & drop or file selector)
* **Choose a pretrained model** (e.g. ResNet-50)
* **Select any convolutional layer** for visualization
* **View Grad-CAM heatmaps** showing which areas most influence predictions
* **See feature activations** for deeper insight into learned patterns
* **Adjust heatmap opacity** for clearer overlays
* Clean, modern **React + TailwindCSS** interface

---

## 🧩 Tech Stack

**Frontend:**

* React (Functional Components + Hooks)
* TailwindCSS for styling
* Lucide Icons
* Fetch API for backend communication

**Backend (expected):**

* FastAPI (Python)
* PyTorch or TensorFlow for model inference
* Torchvision pre-trained models (ResNet, VGG, etc.)
* Grad-CAM implementation
* CORS enabled for localhost development

---

## 🖼️ Example Workflow

1. Start the **backend** server (FastAPI, running on port `8000`).
2. Run the **frontend** development server with `npm start`.
3. Upload an image (e.g., a cat or car photo).
4. Choose the model (`ResNet-50`) and layer (`layer4`).
5. Click **Analyze Image** to visualize:

   * **Top predictions** (class + confidence)
   * **Grad-CAM heatmap**
   * **Feature activations**

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/explain-my-model.git
cd explain-my-model
```

### 2️⃣ Install Dependencies

For the frontend:

```bash
cd explain-my-model-frontend
npm install
```

For the backend (Python 3.10+):

```bash
cd explain-my-model-backend
pip install -r requirements.txt
```

### 3️⃣ Run the App

**Start backend (FastAPI):**

```bash
uvicorn main:app --reload
```

**Start frontend:**

```bash
npm start
```

The app will be available at **[http://localhost:3000](http://localhost:3000)**

---

## 🧠 API Endpoints (Backend)

| Endpoint   | Method | Description                                                                 |
| ---------- | ------ | --------------------------------------------------------------------------- |
| `/models`  | GET    | Returns available models and their layers                                   |
| `/analyze` | POST   | Upload an image, model name, and layer to generate Grad-CAM and activations |

**Example request (multipart/form-data):**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@dog.jpg" \
  -F "model_name=resnet50" \
  -F "layer=layer4"
```

---

## 🧪 Example Response

```json
{
  "predictions": [
    {"class": "golden retriever", "confidence": 0.87},
    {"class": "Labrador retriever", "confidence": 0.09}
  ],
  "gradcam": "data:image/png;base64,...",
  "activations": "data:image/png;base64,..."
}
```

---

## 🌈 UI Preview

| Upload & Model Selection          | Grad-CAM Visualization       | Activation Maps                      |
| --------------------------------- | ---------------------------- | ------------------------------------ |
| ![Upload Screen](docs/upload.png) | ![GradCAM](docs/gradcam.png) | ![Activations](docs/activations.png) |

---

## 🧭 Project Structure

```
explain-my-model/
├── explain-my-model-frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── ExplainMyModelDashboard.jsx
│   │   ├── index.js
│   │   └── ...
│   ├── public/
│   ├── package.json
│   └── tailwind.config.js
└── explain-my-model-backend/
    ├── main.py
    ├── models/
    ├── utils/
    └── requirements.txt
```

---

## 💡 Future Enhancements

* Add model upload support for custom networks
* Integrate layer-wise filter visualization
* Add real-time Grad-CAM overlays
* Compare multiple models side-by-side
* Deploy frontend + backend via Docker

---

## 🧑‍💻 Author

**Mattia Jorgen Prugnoli**
AI & Software Engineer passionate about explainable AI, neural visualization, and user-centered design.
📫 Connect on [GitHub](https://github.com/<your-username>) or [LinkedIn](https://linkedin.com/in/<your-handle>)

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify it.

---

**"The best way to understand a neural network is to *see* what it sees."**
