# cad_agent
Fine-tuning LLM to create good CAD designs

## Dataset Information

### ShapeNet Core V2 Dataset

This project uses the ShapeNet Core V2 dataset, which is a large-scale dataset of 3D shapes. The dataset is downloaded using kagglehub to the following location:

```
C:\Users\frost\.cache\kagglehub\datasets\hajareddagni\shapenetcorev2\versions\1
```

### Dataset Structure

The ShapeNet Core V2 dataset is organized as follows:

```
shapenetcorev2/
├── [category_id_1]/         # Each category has its own folder (e.g., '02691156' for airplanes)
│   ├── [model_id_1]/        # Each model has its own folder with a unique ID
│   │   ├── model.obj        # 3D model in OBJ format
│   │   ├── model.mtl        # Material file for the OBJ model
│   │   ├── texture.png      # Texture image (if available)
│   │   └── model_normalized.obj  # Normalized version of the model (sometimes present)
│   ├── [model_id_2]/
│   └── ...
├── [category_id_2]/
└── ...
```

The dataset contains approximately 51,300 unique 3D models across 55 common object categories. Each category is identified by a synset ID from WordNet, and each model has a unique identifier.

Common categories include:
- Airplanes (02691156)
- Cars (02958343)
- Chairs (03001627)
- Tables (04379243)
- Lamps (03636649)
- And many more

Each model typically includes:
- Geometry data (vertices, faces)
- Material information
- Texture maps (when available)

## Usage
```bash
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## Analysis

The `run.py` script provides analysis of the dataset structure, including:

- Category distributions
- File counts and types
- Sample model information
- Visualization of category distribution
