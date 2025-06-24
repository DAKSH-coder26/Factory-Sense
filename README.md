# **FactorySense: AI-Powered Anomaly Detection for Production Lines**

**Project Status:** 🚧 Under Active Development & Optimization 🚧

FactorySense is an end-to-end Python project that simulates a bottle production line and uses modern computer vision and AI techniques to monitor it in real-time. It can autonomously detect, track, and diagnose anomalies in the production process, providing human-readable explanations and suggested actions using a Large Language Model.

This project is a powerful demonstration of how synthetic data can be used to train robust AI models for industrial automation and quality control.

## **Key Features**

* **Synthetic Data Generation:** Includes a script to generate thousands of labeled training images from a dynamic simulation, complete with domain randomization (varied backgrounds, blur).  
* **YOLOv8 Object Detection:** Utilizes a custom-trained YOLOv8 model to detect the state of bottles on the conveyor belt (e.g., empty, filling, capped, labeled).  
* **Multi-Object Tracking:** Implements DeepSORT to assign a unique ID to each bottle and track it consistently as it moves through the production line.  
* **Rule-Based Anomaly Detection:** A robust logic module analyzes each bottle's history and position to detect a variety of anomalies:  
  * Stuck bottles  
  * Production stages out of order  
  * Missing process steps (e.g., missing label)  
  * Misalignment within a production zone  
* **LLM-Powered Reasoning:** Connects to the Google Gemini API to translate complex anomaly data into concise, actionable insights for an operator.  
* **Modular & Customizable:** The code is organized into logical modules for simulation, detection, tracking, and reasoning, making it easy to extend and modify.

## **Project Structure**

.  
├── agent/  
│   └── llm\_reasoner.py       \# Handles communication with the Gemini LLM.  
├── tracking/  
│   ├── anomaly\_detector.py   \# Defines the rules for identifying anomalies.  
│   └── bottle\_tracker.py       \# Stores and manages the history of each bottle.  
├── vision/  
│   ├── deep\_sort\_tracker.py  \# Implements the DeepSORT tracking algorithm.  
│   └── yolo\_detector.py      \# Manages the YOLOv8 object detection model.  
├── backgrounds/                \# (Optional) Folder for background images for the simulation.  
├── bottle1.png                 \# Image asset for the bottle simulation.  
├── generate\_data.py            \# Step 1: Script to create the synthetic dataset.  
├── split\_data.py               \# Step 2: Script to split data into train/validation sets.  
├── main.py                     \# Step 4: Main script to run the live simulation and detection.  
├── scenario\_generator.py       \# Creates scripted scenarios with random anomalies for the simulation.  
├── simulation\_elements.py      \# Defines the core components of the virtual factory environment.  
└── dataset.yaml                \# Configuration file for training the YOLO model.

## **Setup and Installation**

### **1\. Clone the Repository**

git clone \<your-repository-url\>  
cd \<repository-folder\>

### **2\. Create a Python Virtual Environment**

It is highly recommended to use a virtual environment to manage project dependencies.

\# Windows  
python \-m venv venv  
.\\venv\\Scripts\\activate

\# macOS / Linux  
python3 \-m venv venv  
source venv/bin/activate

### **3\. Install Dependencies**

Install all the required libraries using the provided requirements.txt file.

pip install \-r requirements.txt

### **4\. Set Up Environment Variables**

This project uses the Google Gemini API for anomaly reasoning. You will need an API key.

1. Create a file named .env in the root of the project directory.  
2. Add your API key to the file as follows:  
   GEMINI\_API\_KEY="YOUR\_API\_KEY\_HERE"

3. You can get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## **How to Run the Project**

Follow these steps in order to generate data, train the model, and run the live simulation.

### **Step 1: Generate Synthetic Data**

Run the generate\_data.py script. This will create a data/ folder containing images/ and labels/ subdirectories filled with thousands of training samples.

python generate\_data.py

### **Step 2: Split the Dataset**

Run the split\_data.py script. This will process the contents of the data/ folder and split them into train and val sets required for training.

python split\_data.py

### **Step 3: Train the YOLOv8 Model**

Train the object detection model using the Ultralytics command-line interface. You can specify the number of epochs and image size.

yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640

This process can take some time depending on your hardware. Once complete, the best-trained model will be saved in the runs/detect/train/weights/ directory as best.pt.

### **Step 4: Run the Live Simulation**

1. Copy the trained model file (runs/detect/train/weights/best.pt) to the root directory of the project.  
2. Run the main.py script to start the FactorySense simulation.

python main.py

The application window will appear, and the scenario will begin, with the AI monitoring the bottles and reporting any detected anomalies in the console and on-screen.