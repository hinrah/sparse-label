# Sparse Vessel Masks

This repository can be used for the training and evaluation of 3D image segmentation using sparse annotations. It converts centerline and contour annotations into voxel masks. Voxels for which no or conflicting sparse annotations exist are set to an ignore label and do not contribute to the loss.  
We do not provide any network structures or training scripts, but the created sparse voxel masks can be used with network structures and training schemes that support ignore labels (e.g., nnUNet [[1]](#1) [[2]](#2)).

### Installing dependencies
This project was developed with Python 3.10.  

To install dependencies, run:  
```bash
pip install -r requirements.txt
```

# Get Started

### Example Dataset

To get familiar with the library, we recommend running the complete workflow on the provided example dataset with sparse annotations for a subset of the TotalSegmentator dataset.

1. Clone the repository to ```<repository_path>```
2. Create a Python 3.10 environment (this is the version we tested the code with)
3. Install dependencies: ```pip install -r <repository_path>/requirements.txt```
4. Download the TotalSegmentator dataset from here: https://zenodo.org/records/14710732
5. Unpack the dataset
6. Set the environment variable ```sparseVesselMasks_raw``` to ```<repository_path>/example_dataset```
7. Set the environment variable ```PYTHONPATH``` to ```<repository_path>/src```
8. Run:
   ```bash
   python <repository_path>/example_dataset/copy_relevant_image_data.py -i <path_to_totalSegmentator_dataset>
   ```
9. Run:
   ```bash
   python <repository_path>/src/sparselabel/entrypoints/create_sparse_label.py -d 1 -n 4 --no-wall
   ```
   (If you have sufficient RAM and cores, you can set a higher `-n` to speed up the process)
10. You can view the created labels in ```<repository_path>/example_dataset/labelsTr``` and use them for network training.
11. You are now ready to train an nnUNet. Follow their documentation: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
    - Install torch (with CUDA) and nnUNetv2 (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)
    - Set the environment variable ```nnUNet_raw``` to ```<repository_path>/example_dataset```
    - Set the environment variable ```nnUNet_results``` to ```<repository_path>/nnUNet_results```
    - Set the environment variable ```nnUNet_preprocessed``` to ```<repository_path>/nnUNet_preprocessed```
    - Run:
      ```bash
      nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
      ```
    - Run:
      ```bash
      nnUNetv2_train 1 3d_fullres <FOLD>
      ```

## Train on Your Own Data

To train on your own data, you will need to transform your sparse annotations into the expected format. When doing so, keep in mind that all annotations are stored in world space and the coordinate system needs to be RAS (Right-Anterior-Superior), as NIFTI images are stored in this format.

#### Centerline Annotations
The centerline consists of a NetworkX graph with nodes and edges. Each node has a 3D coordinate in the image space. Each edge connects two nodes and has skeletons, which are the paths between the two nodes. This allows the handling of branching structures. An example of the JSON structure can be found in:  
```<repository_path>/example_dataset/Dataset001_totalSegmentatorAorta/centerlinesTr/s0019.json```

To load a centerline, run this code:
```python
import json
from networkx.readwrite import json_graph

with open(centerline_path, "r", encoding="ascii") as file:
    centerline_raw = json.load(file)
centerline = json_graph.node_link_graph(centerline_raw, link="edges")
```

To create centerline annotations, you will need to generate the centerline with a tool of your choice and export it to a JSON file. Here is an example of how to create a centerline with NetworkX:
```python
import json
import networkx as nx
from networkx.readwrite import json_graph

def save(centerline, save_path):
    out_graph = nx.DiGraph()
    out_graph.graph["coordinateSystem"] = "RAS"
    
    for node in centerline.getNodes():
        out_graph.add_node(
            node.id,
            pos=(node.position()).tolist(),
        )
    
    for edge in centerline.getEdges():
        start_node = edge.getStartNode()
        end_node = edge.getEndNode()
        out_graph.add_edge(
            start_node.id,
            end_node.id,
            skeletons=(np.array(edge.get_skeleton_points()))
        )
    
    serialized_graph = json_graph.node_link_data(out_graph, edges="edges")
    with open(save_path, "w") as file:
        json.dump(serialized_graph, file)
```

#### Cross-Section Contour Annotations
All cross-section annotations for one case are stored in a single file. Each cross-section contains an `inner_contour` and optionally an `outer_contour`. The contours are represented as a list of 3D coordinates. An example of the JSON structure can be found in:  
```<repository_path>/example_dataset/Dataset001_totalSegmentatorAorta/contoursTr/s0019.json```

An example contour annotation for one case looks like this:
```json
{
    "cross_section_1": {
        "inner_contour": [
            [x1, y1, z1],
            [x2, y2, z2],
            ...
        ],
        "outer_contour": [
            [x1, y1, z1],
            [x2, y2, z2],
            ...
        ]
    }, 
    "cross_section_2": {
        ...
    }
}
```

### Adding Different Annotations and Strategies

Sparse annotations may come in various forms not yet considered in this library. The code is structured to allow the addition of new strategies to convert them into voxel masks. To add a new strategy, you will need to implement the abstract base class ```sparselabel.label_strategies.labeling_strategies.LabelingStrategy``` and add it to the list of applied strategies.

# References
<a id="1">[1]</a>  
Gotkowski, Karol, et al. *"Embarrassingly simple scribble supervision for 3D medical segmentation."* arXiv preprint arXiv:2403.12834 (2024).  

<a id="2">[2]</a>  
Isensee, Fabian, et al. *"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation."* *Nature Methods* 18.2 (2021): 203-211.

# License
The code in this repository is licensed under the GNU-GPL license.  
The annotations in `/example_dataset/` were created using a subset of the TotalSegmentator MRI dataset (https://zenodo.org/records/14710732) and are licensed under the CC BY-NC-SA 4.0 License.  
(https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)
