This project investigates open-vocabulary affordance prediction by combining 3D reconstruction from SAM3D with semantic context extracted from a Vision-Language Model (VLM). The goal is to predict fine-grained, verb-conditioned affordance scores on a reconstructed 3D representation, enabling the model to reason about interactions such as *grasp*, *sit on*, *pour from*, or *open* in an open-vocabulary setting. The method combines geometric understanding from SAM3D with semantic reasoning from a frozen VLM to produce dense affordance fields over reconstructed objects.

The proposed pipeline consists of the following stages:

---

## 1. Input Processing

The system receives as input a single RGB-D image together with an open-vocabulary verb describing a potential interaction. The RGB-D image provides both appearance and geometric cues, while the verb acts as a semantic query conditioning the affordance prediction process.

The goal is to infer a probabilistic affordance value in the range ([0,1]) for every vertex of the reconstructed object, representing the likelihood that the queried interaction can be performed at that location.

---

## 2. 3D Reconstruction using SAM3D

The RGB-D image is processed using SAM3D to generate a dense 3D representation of the scene or object. The reconstruction consists of:

* a textured mesh representation,
* and a Gaussian splatting representation used for high-quality rendering and view synthesis.

The mesh representation serves as the primary structure for affordance prediction, while Gaussian splats provide improved rendering consistency and richer view-dependent appearance information during multi-view feature extraction.

In addition to geometry, SAM3D produces latent geometric features associated with mesh vertices or reconstructed points. To obtain stronger global geometric context, these local features can optionally be aggregated using a PointNet-style encoder that produces:

* local geometric features per vertex,
* and a global object-level latent representation.

The global latent is intended to capture object-scale semantic and structural information that may be important for affordances requiring holistic reasoning, such as *sit on* or *pour from*.

---

## 3. Novel View Sampling and Rendering

To obtain richer semantic coverage than available from the original observation, multiple novel viewpoints are sampled around the reconstructed object.

The reconstructed mesh and Gaussian splats are rendered from these viewpoints to produce:

* RGB renderings,
* depth maps,
* visibility masks,
* and camera correspondences.

Initially, viewpoint sampling can be implemented using simple uniform spherical sampling around the object. Later iterations may explore more advanced visibility-aware or coverage-optimized view selection strategies.

The Gaussian splatting representation is primarily used during rendering to improve visual fidelity and preserve fine-grained appearance information for downstream VLM processing.

---

## 4. Vision-Language Feature Extraction

Each rendered view is processed by a frozen Vision-Language Model together with the queried verb.

The VLM extracts:

* patch-level visual embeddings from the rendered images,
* and a text embedding representing the semantic meaning of the verb.

The initial implementation uses feature concatenation for conditioning, where the verb embedding is concatenated with projected visual and geometric features. This provides a lightweight and stable baseline for open-vocabulary affordance prediction.

Initially, patch embeddings from the final VLM layers can be used directly. However, since deeper layers may lose spatial precision, later experiments may investigate extracting features from intermediate transformer layers to better preserve:

* local geometry,
* contact regions,
* handles,
* openings,
* and interaction-relevant structures.

Future extensions may additionally explore:

* cross-attention between verb embeddings and visual features,
* multi-scale feature extraction,
* or semantic attention maps from the VLM.

---

## 5. Projection of VLM Features onto the 3D Representation

Using the rendering correspondences between image space and 3D geometry, the VLM patch embeddings are projected back onto the reconstructed 3D object.

Projection is performed by mapping visible 3D points or mesh vertices into image space and associating them with corresponding VLM patch features. Features from multiple rendered views are aggregated per vertex to produce a unified semantic representation over the mesh.

The initial implementation may use simple aggregation strategies such as:

* averaging,
* max pooling,
* or weighted averaging.

Subsequent iterations will investigate attention-based multi-view feature fusion, where the model learns to combine semantic evidence from multiple viewpoints dynamically.

Visibility information obtained during rendering is used to ensure that only geometrically visible vertices receive semantic updates from a given view. More advanced visibility-aware confidence weighting and uncertainty estimation may later be explored to reduce hallucinated affordance predictions in occluded regions.

---

## 6. Vertex Feature Fusion

For every mesh vertex, multiple information sources are fused into a joint feature representation:

* SAM3D geometric latent,
* projected VLM semantic latent,
* optional global PointNet latent,
* and the verb embedding.

This creates a unified geometric-semantic representation capturing:

* local shape information,
* object-scale context,
* visual semantics,
* and interaction intent.

The fused representation forms the basis for downstream affordance prediction.

---

## 7. Affordance Prediction Head

The initial affordance predictor is implemented as a lightweight multilayer perceptron (MLP) operating independently on each vertex feature.

The MLP predicts a probabilistic affordance score in the range ([0,1]) for every mesh vertex.

This lightweight formulation serves as a strong baseline while keeping the overall system computationally efficient and modular.

Future iterations may replace the MLP with more expressive architectures, including:

* transformer-based vertex reasoning,
* graph neural networks over mesh connectivity,
* or geometry-aware message passing networks.

These extensions may improve spatial consistency and enable more sophisticated interaction reasoning across neighboring surface regions.

---

## 8. Training and Supervision

The system is intended to be trained using affordance annotations from datasets such as AGD20K.

Training supervision is formulated as probabilistic vertex-wise affordance prediction conditioned on the queried verb.

The initial training setup will likely use binary cross-entropy loss on per-vertex affordance labels. Later extensions may additionally incorporate:

* spatial smoothness regularization,
* visibility-aware consistency losses,
* multi-view consistency objectives,
* or contrastive language-geometry alignment losses.

---

## 9. Output Representation

The final output is a dense, verb-conditioned affordance field defined over the reconstructed 3D mesh.

Each vertex receives a continuous affordance score representing the predicted likelihood that the queried interaction can be performed at that location. High-scoring regions correspond to semantically and geometrically plausible interaction areas, enabling fine-grained 3D reasoning about object affordances in an open-vocabulary setting.
