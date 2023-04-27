# Import the Images module from pillow
import os

from PIL import Image

# Open the image by specifying the image path.
dir_path = "C:\\Users\\dbelarbia\\pa\\dataset\\en colère"
for img in os.listdir(dir_path):
    print(img)
    image = Image.open(os.path.join(dir_path, img))

    image = image.convert("L")
    image = image.resize((150, 150))

    image.save(os.path.join(dir_path, img), quality=50)
