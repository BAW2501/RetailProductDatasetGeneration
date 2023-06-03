import os
import random
from PIL import Image, ImageDraw

input_folder = "cropped_images"
output_folder = "Generated"
generated_images_folder = os.path.join(output_folder, "X")
generated_labels_folder = os.path.join(output_folder, "Y")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(generated_images_folder, exist_ok=True)
os.makedirs(generated_labels_folder, exist_ok=True)

def resize_image(image_path, output_size):
    image = Image.open(image_path)
    image.thumbnail(output_size)
    return image

def generate_dataset(source_folder, num_images, min_objects=1, max_objects=5, output_size=(640, 640)):
    class_folders = os.listdir(source_folder)

    for i in range(num_images):
        # Choose a random number of objects for the image
        num_objects = random.randint(min_objects, max_objects)

        # Create a blank canvas for the new image
        canvas = Image.new("RGB", output_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        annotations = []  # List to store the annotations for the image

        for _ in range(num_objects):
            # Choose a random class folder
            class_folder = random.choice(class_folders)
            class_path = os.path.join(source_folder, class_folder)

            # Choose a random product image from the class folder
            product_image = random.choice(os.listdir(class_path))
            product_image_path = os.path.join(class_path, product_image)

            # Open the product image and resize it
            product = resize_image(product_image_path, (256, 256))  # Adjust the size as needed

            product_width, product_height = product.size

            # Randomly choose a position on the canvas where the product fits without overlapping
            while True:
                x = random.randint(0, output_size[0] - product_width)
                y = random.randint(0, output_size[1] - product_height)

                # Check if the chosen position overlaps with any existing product
                if all(
                    canvas.getpixel((px, py)) == (255, 255, 255)
                    for px in range(x, x + product_width)
                    for py in range(y, y + product_height)
                ):
                    break

            # Paste the product image on the canvas
            canvas.paste(product, (x, y))

            # Calculate the normalized bounding box coordinates
            xmin = x / output_size[0]
            ymin = y / output_size[1]
            xmax = (x + product_width) / output_size[0]
            ymax = (y + product_height) / output_size[1]

            # Append the annotation to the list
            annotation = f"{class_folder} {xmin} {ymin} {xmax} {ymax}"
            annotations.append(annotation)

        # Save the generated image
        image_filename = f"generated_{i}.jpg"
        image_path = os.path.join(generated_images_folder, image_filename)
        canvas.save(image_path)

        # Save the annotations to a text file
        label_filename = f"generated_{i}.txt"
        label_path = os.path.join(generated_labels_folder, label_filename)
        with open(label_path, "w") as f:
            f.write("\n".join(annotations))

        print(f"Generated image {i+1}/{num_images}")

generate_dataset(input_folder, num_images=10, min_objects=1, max_objects=5, output_size=(640, 640))
