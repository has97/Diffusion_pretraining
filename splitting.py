import os

# Define the root directory of your dataset
root_dir = '/raid/biplab/phduser1/Hassan/diffused-solo-learn-main_1/diffused-solo-learn-main/domainnet/domainnet-sketch/train'

# Initialize a list to store image paths and corresponding class labels
image_paths_with_labels = []

# Walk through the root directory and its subdirectories
for class_label in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, class_label)
    
    # Check if it's a directory
    if os.path.isdir(class_dir):
        for image_filename in os.listdir(class_dir):
            # Get the full path to the image
            image_path = os.path.join(class_dir, image_filename)
            
            # Extract the class label from the folder name
            label = int(class_label)
            
            # Append the image path and label to the list
            image_paths_with_labels.append((image_path, label))

# Define the output file path
output_file = '/raid/biplab/phduser1/Hassan/diffused-solo-learn-main_1/diffused-solo-learn-main/domainnet_splits/labeled_source_images_sketch.txt'

# Write the data to the text file
with open(output_file, 'w') as file:
    for image_path, label in image_paths_with_labels:
        file.write(f"{image_path} {label}\n")

print(f"Data saved to {output_file}")