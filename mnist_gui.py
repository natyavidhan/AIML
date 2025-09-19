import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import io
import os
from tempfile import NamedTemporaryFile

weights = np.load("assets/mnist_weights.npz")
W1 = weights["W1"]
b1 = weights["b1"]
W2 = weights["W2"]
b2 = weights["b2"]

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def predict(image):
    image = image.reshape(784, 1).astype(np.float32)
    Z1 = np.dot(W1, image) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return A2

def random_img():
    idx = np.random.randint(0, 10000)
    df = pd.read_csv("assets/mnist.csv", nrows=10000)
    row = df.iloc[idx]
    label = row[0]
    pixels = row[1:].values.reshape(28, 28).astype(np.uint8)
    return pixels, label

class MNISTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognizer")
        
        self.root.geometry("615x340")
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white', cursor="cross", borderwidth=2, relief="ridge")
        self.canvas.place(x=35, y=20)
        self.canvas.bind("<B1-Motion>", self.draw)

        tk.Label(root, text="MNIST Digit Recognizer", font=("Helvetica", 16)).place(x=350, y=30)
        
        result_elements = []
        # a label for each digit and its probability with a progress bar
        for i in range(10):
            tk.Label(root, text=f"{i}", font=("Helvetica", 12)).place(x=340, y=70 + i*25)
            label = tk.Label(root, text=f"0.00%", font=("Helvetica", 12))
            label.place(x=550, y=70 + i*25)
            progress = tk.Canvas(root, width=180, height=20, bg='lightgray')
            progress.place(x=360, y=70 + i*25)
            bar = progress.create_rectangle(0, 0, 0, 20, fill='blue')
            result_elements.append((label, progress, bar))
            
        self.result_elements = result_elements
        self.clear_button = tk.Button(root, text="Clear", command=self.clear)
        self.clear_button.place(x=100, y=310)
        
        self.random_button = tk.Button(root, text="Random", command=self.load_random_image)
        self.random_button.place(x=150, y=310)
        
        self.last_x, self.last_y = None, None
        
    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=8, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
        self.last_x, self.last_y = event.x, event.y
        self.update_prediction()
        
    def clear(self):
        self.canvas.delete("all")
        self.last_x, self.last_y = None, None
        for label, progress, bar in self.result_elements:
            label.config(text=f"0.00%")
            progress.coords(bar, 0, 0, 0, 20)
            
    def update_prediction(self):
        x = self.canvas.winfo_rootx() + 2
        y = self.canvas.winfo_rooty() + 2
        width = self.canvas.winfo_width() - 4
        height = self.canvas.winfo_height() - 4

        img = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        img = img.convert('L')

        img = ImageOps.invert(img)
        img = ImageOps.fit(img, (28, 28), method=Image.LANCZOS, centering=(0.5, 0.5))

        img_data = np.array(img).astype(np.float32) / 255.0
        probs = predict(img_data)

        for i, (label, progress, bar) in enumerate(self.result_elements):
            prob = probs[i, 0]
            label.config(text=f"{prob*100:.2f}%")

            bar_width = int(prob * progress.winfo_width())
            progress.coords(bar, 0, 0, bar_width, 20)
            
    def load_random_image(self):
        # Clear the canvas first
        self.clear()
        
        # Get a random image from the dataset
        img_data, label = random_img()
        
        # Calculate the scaling factor to display the 28x28 image on the 280x280 canvas
        scale = 10
        
        # Draw the image on the canvas pixel by pixel
        for y in range(28):
            for x in range(28):
                # Get the pixel intensity (0-255)
                intensity = img_data[y, x]
                if intensity > 0:
                    # Convert to hex color (grayscale)
                    color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                    # Draw a rectangle for each pixel
                    self.canvas.create_rectangle(
                        x * scale, y * scale, 
                        (x + 1) * scale, (y + 1) * scale, 
                        fill=color, outline=color
                    )
        
        # Update prediction for the loaded image
        # probs = predict(img_data / 255.0)
        
        # # Update the UI with prediction results
        # for i, (label, progress, bar) in enumerate(self.result_elements):
        #     prob = probs[i, 0]
        #     label.config(text=f"{prob*100:.2f}%")
        #     bar_width = int(prob * progress.winfo_width())
        #     progress.coords(bar, 0, 0, bar_width, 20)
        
        # # Display the correct label
        # print(f"Loaded image with label: {label}")
        self.update_prediction()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTApp(root)
    root.mainloop()