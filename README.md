# R&D

Repository of ideas in the brewing stage.


# Notes

Networks to try for thing segmentation:

- Xception (Might not be real-time)
- BiSeNet (uses xception (but still very fast??) 60fps!)
- ResNet (turned into fully connected net)

_*Losses*_:

- Start with pixel-wise cross-entropy
- Try Dice-loss (balances out classes and small object but might lead to 
    unstable gradients)  
- Triplet loss for dense correspondence

# Outline

*SceneFlow Dataset for ThingNet*

- Contruct HDF5 dataset from tarfiles 
    - construct image pairs for use at training (rgb, depth, material_image, object_image)
    - maybe down sample images (540, 960) -> (360, 640) or even (270, 480)

- At training time, draw matches and non-matches for each frame (randomly)

*Consideraions*

- Might want to start of with really low resolution image and move to high res later
    
