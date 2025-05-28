import os
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# The path where the dataset was downloaded as specified in download.py
DATASET_PATH = r"C:\Users\frost\.cache\kagglehub\datasets\hajareddagni\shapenetcorev2\versions\1"

def analyze_dataset_structure():
    """Analyze the structure of the ShapeNet Core V2 dataset."""
    print(f"Analyzing ShapeNet Core V2 dataset at: {DATASET_PATH}")
    
    # Check if the path exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path {DATASET_PATH} does not exist.")
        print("Please run download.py first to download the dataset.")
        return
    
    # Count files and directories
    total_dirs = 0
    total_files = 0
    file_extensions = Counter()
    category_counts = {}
    
    # Get the top-level directories (categories)
    try:
        categories = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        print(f"\nFound {len(categories)} top-level directories/categories")
    except Exception as e:
        print(f"Error accessing dataset directory: {e}")
        return
    
    # Go through each category
    for category in categories:
        category_path = os.path.join(DATASET_PATH, category)
        model_count = 0
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(category_path):
            total_dirs += len(dirs)
            total_files += len(files)
            model_count += len([d for d in dirs if os.path.isdir(os.path.join(root, d))])
            
            # Count file extensions
            for file in files:
                _, ext = os.path.splitext(file)
                if ext:
                    file_extensions[ext.lower()] += 1
        
        category_counts[category] = model_count
        print(f"Category '{category}': {model_count} models")
    
    # Print overall statistics
    print(f"\nTotal Statistics:")
    print(f"Total directories: {total_dirs}")
    print(f"Total files: {total_files}")
    print(f"\nFile extensions:")
    for ext, count in file_extensions.most_common(10):
        print(f"  {ext}: {count} files")
    
    # Try to find and load metadata if it exists
    try_load_metadata()
    
    # Plot category distribution if we have categories
    if category_counts:
        plot_category_distribution(category_counts)

def try_load_metadata():
    """Try to find and load any metadata files in the dataset."""
    # Common metadata file names
    metadata_files = ['metadata.json', 'taxonomy.json', 'categories.json']
    
    for filename in metadata_files:
        filepath = os.path.join(DATASET_PATH, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    metadata = json.load(f)
                print(f"\nFound metadata file: {filename}")
                print(f"Metadata contains {len(metadata)} items")
                
                # Print a sample of the metadata structure
                if isinstance(metadata, dict):
                    print("Sample metadata keys:", list(metadata.keys())[:5])
                elif isinstance(metadata, list) and len(metadata) > 0:
                    if isinstance(metadata[0], dict):
                        print("Sample metadata fields:", list(metadata[0].keys()))
                
                return metadata
            except Exception as e:
                print(f"Error loading metadata file {filename}: {e}")
    
    print("\nNo standard metadata files found.")
    return None

def plot_category_distribution(category_counts):
    """Create a bar chart of model counts by category."""
    try:
        # Sort categories by count in descending order
        categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [cat for cat, count in categories]
        counts = [count for cat, count in categories]
        
        # If there are too many categories, limit to top 15
        if len(labels) > 15:
            labels = labels[:15]
            counts = counts[:15]
            title = "Top 15 Categories by Model Count"
        else:
            title = "Model Count by Category"
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.ylabel('Number of Models')
        plt.title(title)
        plt.tight_layout()
        
        # Save the plot to a file
        plt.savefig('category_distribution.png')
        print("\nCategory distribution chart saved as 'category_distribution.png'")
        
    except Exception as e:
        print(f"Error creating category distribution plot: {e}")

def analyze_sample_models():
    """Analyze a few sample models from the dataset."""
    print("\nAnalyzing sample models...")
    
    # Common 3D model file extensions
    model_extensions = ['.obj', '.stl', '.ply', '.off']
    
    # Try to find sample models
    sample_paths = []
    
    try:
        categories = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        
        # Look in up to 3 categories
        for category in categories[:3]:
            category_path = os.path.join(DATASET_PATH, category)
            
            # Get subdirectories (model instances)
            model_dirs = [d for d in os.listdir(category_path) 
                          if os.path.isdir(os.path.join(category_path, d))]
            
            # Look at up to 2 models per category
            for model_dir in model_dirs[:2]:
                model_path = os.path.join(category_path, model_dir)
                
                # Look for model files
                for root, _, files in os.walk(model_path):
                    for file in files:
                        _, ext = os.path.splitext(file)
                        if ext.lower() in model_extensions:
                            sample_paths.append(os.path.join(root, file))
                            break  # Only take one model file per directory
                
                if len(sample_paths) >= 5:  # Limit to 5 samples
                    break
            
            if len(sample_paths) >= 5:
                break
    
    except Exception as e:
        print(f"Error finding sample models: {e}")
    
    # Report on sample models
    if sample_paths:
        print(f"Found {len(sample_paths)} sample model files:")
        for path in sample_paths:
            rel_path = os.path.relpath(path, DATASET_PATH)
            file_size = os.path.getsize(path) / 1024  # size in KB
            print(f"  {rel_path} ({file_size:.1f} KB)")
    else:
        print("No sample model files found with extensions: " + ", ".join(model_extensions))

def print_dataset_examples():
    """Print 3 detailed examples from the dataset with complete file structure."""
    print("\n" + "=" * 40)
    print("DATASET EXAMPLES - 3 DETAILED MODEL SHOWCASES")
    print("=" * 40)
    
    try:
        categories = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        
        if not categories:
            print("No categories found in the dataset.")
            return
            
        # Take 3 different categories if possible
        sample_categories = categories[:3] if len(categories) >= 3 else categories
        
        for i, category in enumerate(sample_categories, 1):
            category_path = os.path.join(DATASET_PATH, category)
            model_dirs = [d for d in os.listdir(category_path) 
                        if os.path.isdir(os.path.join(category_path, d))]
            
            if not model_dirs:
                print(f"\nExample {i}: Category '{category}' has no models.")
                continue
                
            # Take the first model in this category
            model_dir = model_dirs[0]
            model_path = os.path.join(category_path, model_dir)
            
            print(f"\nExample {i}: Category '{category}', Model ID: '{model_dir}'")
            print("-" * 40)
            
            # Print file structure for this model
            file_list = []
            total_size = 0
            
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, model_path)
                    size = os.path.getsize(file_path)
                    total_size += size
                    file_list.append((rel_path, size))
            
            # Print directory structure
            print(f"Model path: {os.path.relpath(model_path, DATASET_PATH)}")
            print(f"Total files: {len(file_list)}")
            print(f"Total size: {total_size/1024:.1f} KB")
            print("\nFiles:")
            
            # Sort files by name for consistent output
            file_list.sort()
            for rel_path, size in file_list:
                print(f"  {rel_path:<30} {size/1024:.1f} KB")
            
            # Try to count vertices and faces if there's an OBJ file
            obj_files = [f for f, _ in file_list if f.lower().endswith('.obj')]
            if obj_files:
                try:
                    obj_path = os.path.join(model_path, obj_files[0])
                    vertex_count = 0
                    face_count = 0
                    
                    with open(obj_path, 'r') as f:
                        for line in f:
                            if line.startswith('v '): # vertex
                                vertex_count += 1
                            elif line.startswith('f '): # face
                                face_count += 1
                    
                    print(f"\nGeometry Information (from {obj_files[0]}):")
                    print(f"  Vertices: {vertex_count}")
                    print(f"  Faces: {face_count}")
                    print(f"  Vertex-to-face ratio: {vertex_count/max(face_count, 1):.2f}")
                except Exception as e:
                    print(f"\nCould not analyze OBJ file: {e}")
            
    except Exception as e:
        print(f"Error printing dataset examples: {e}")

if __name__ == "__main__":
    print("ShapeNet Core V2 Dataset Analysis")
    print("="*40)
    analyze_dataset_structure()
    analyze_sample_models()
    print_dataset_examples()  # Added detailed examples
    print("\nAnalysis complete!")