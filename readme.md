# 🎨 Neural Style Transfer with PyTorch

This project performs **Neural Style Transfer** using a pre-trained VGG19 model in PyTorch. It blends the *content* of one image with the *style* of another to generate a new artistic image.

---

## 📌 Project Description

Neural Style Transfer is a deep learning technique that applies the artistic style of one image (e.g., a painting) to the content of another image (e.g., a photograph). This is achieved by optimizing a target image to simultaneously match the content representation of the content image and the style representation of the style image using feature maps from a pretrained convolutional neural network (CNN), in this case, VGG19.

---

## 🧠 How It Works

- **VGG19** is used to extract feature maps.
- **Content loss** is computed from one layer (usually deeper layers).
- **Style loss** is computed using **Gram matrices** from multiple layers.
- The target image is initialized as a copy of the content image and optimized using **LBFGS** to reduce total loss (content + style).

---

## Output
![Image](https://github.com/user-attachments/assets/b241b638-7b71-4aa8-8517-4f7fe24770c7)
