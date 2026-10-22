# Source and notices

This package is curated from the local development project `torch-points3d-SiamKPConvVariants-master`.

The framework and Siamese KPConv implementation originate from:

- Torch-Points3D: https://github.com/torch-points3d/torch-points3d
- SiamKPConvVariants: https://github.com/IdeGelis/torch-points3d-SiamKPConvVariants
- de Gélis, Corpetti and Lefèvre, “Change Detection Needs Change Information: Improving Deep 3-D Point Cloud Change Detection,” IEEE TGRS, 2024, DOI 10.1109/TGRS.2024.3359484.

The local project includes CTSC cross-temporal features, feature-fusion modifications and Urb3DSimul loading changes. This packaging operation does not establish authorship of every historical modification.

The original BSD license in LICENSE.md and the copyright/source notices inside retained files are preserved. In particular, shared framework helpers may carry their own source notices. The top-level repository name is changed for organization; the internal torch_points3d package name is retained.
