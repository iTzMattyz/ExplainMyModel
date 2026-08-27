# 🧠 Explain My Model

**Explain My Model** is an interactive web app that visualizes what convolutional neural networks (CNNs) "see" inside images.
Using **Grad-CAM** (Gradient-weighted Class Activation Mapping) and **feature activation maps**, it provides an intuitive view into how deep learning models make decisions.

> **Runs locally — there is no hosted demo.** Inference needs a Python process
> with PyTorch and ~1 GB of RAM for the model cache, which no free static or
> serverless host will run. See **Setup Instructions** below to run it on your
> own machine; it takes two terminals and about five minutes.

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

**Backend:**

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
git clone https://github.com/iTzMattyz/ExplainMyModel.git
cd ExplainMyModel
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

### 3️⃣ Configure the Backend URL

The frontend reads its backend address from `REACT_APP_API_URL`, falling back to
`http://localhost:8000` when unset — so local development needs no config at all.
For any deployed build, copy the example and point it at your hosted backend:

```bash
cd explain-my-model-frontend
cp .env.example .env
```

Because this is Create React App, the value is baked in at **build time**, not
read at runtime — rebuild after changing it.

### 4️⃣ Run the App

Two processes, two terminals. **Start the backend first** — the frontend fetches
the model list on mount and shows an empty dropdown if the API is unreachable.

**Terminal 1 — backend (FastAPI):**

```bash
cd explain-my-model-backend
uvicorn main:app --reload --port 8000
```

Port `8000` is what `REACT_APP_API_URL` defaults to. The first `/analyze` call
downloads that model's pretrained weights (~100 MB) into `~/.cache/torch`, so
expect a pause; later calls reuse the cache.

**Terminal 2 — frontend:**

```bash
cd explain-my-model-frontend
npm start
```

The app will be available at **[http://localhost:3000](http://localhost:3000)**.
Sanity-check the backend on its own at
**[http://localhost:8000/models](http://localhost:8000/models)** — it should
return the model list as JSON.

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

## 🧭 Project Structure

```
ExplainMyModel/
├── explain-my-model-frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js          # the entire ExplainMyModelDashboard component
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   ├── .env.example
│   ├── package.json
│   ├── postcss.config.js
│   └── tailwind.config.js
└── explain-my-model-backend/
    ├── main.py                      # FastAPI app: Grad-CAM, activations, model cache
    ├── imagenet-simple-labels.json  # 1000 ImageNet class names
    └── requirements.txt
```

The dashboard is a single component in `src/App.js` — there is no `components/`
directory. Torchvision weights are downloaded on first use and cached under
`~/.cache/torch/hub/checkpoints`, so they are not vendored in the repo.

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
📫 Connect on [GitHub](https://github.com/iTzMattyz) or [LinkedIn](https://www.linkedin.com/in/mattia-jorgen-prugnoli-853169180/)

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify it.

---

**"The best way to understand a neural network is to *see* what it sees."**
