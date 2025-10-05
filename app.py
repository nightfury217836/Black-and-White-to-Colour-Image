import numpy as np
import cv2
from cv2 import dnn
import os

# -------- Model file paths -------- #
proto_file = "models/colorization_deploy_v2.prototxt"
model_file = "models/colorization_release_v2.caffemodel"
hull_pts = "models/pts_in_hull.npy"

# -------- Chatbot style -------- #
def chatbot_print(msg):
    print(f"[🤖 Chatbot]: {msg}")

def user_input(msg):
    return input(f"[👤 You]: {msg}")

def main():
    chatbot_print("Hi! I'm your Image Colorization Assistant 🎨")
    chatbot_print("I can convert your black & white images into color.")

    # Ask for input image
    img_path = user_input("Please enter the path of your black & white image: ")

    # Fix path (allow forward/backslashes)
    img_path = img_path.strip().strip('"').replace("\\", "/")

    if not os.path.exists(img_path):
        chatbot_print("❌ Error: File not found. Please check your path.")
        return

    chatbot_print("✅ Loading pre-trained colorization model...")
    net = dnn.readNetFromCaffe(proto_file, model_file)
    kernel = np.load(hull_pts)

    # Read and preprocess image
    chatbot_print("📂 Reading input image...")
    img = cv2.imread(img_path)
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Add cluster centers
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Resize for network
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    chatbot_print("⚡ Running colorization model...")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

    # Recombine channels
    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Save output
    output_path = "colorized_output.jpg"
    cv2.imwrite(output_path, colorized)

    chatbot_print("🎉 Done! Your image has been colorized successfully.")
    chatbot_print(f"📁 Saved output as: {output_path}")

    # Optional: show images side by side
    show = user_input("Do you want to preview the result? (y/n): ")
    if show.lower() == "y":
        img_resized = cv2.resize(img, (400, 400))
        colorized_resized = cv2.resize(colorized, (400, 400))
        result = cv2.hconcat([img_resized, colorized_resized])
        cv2.imshow("Grayscale -> Colour", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        chatbot_print("Okay, preview skipped 👍")

if __name__ == "__main__":
    main()
