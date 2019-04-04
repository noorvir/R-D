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


# Outline

*SceneFlow Dataset for ThingNet*

- Contruct HDF5 dataset from tarfiles 
    - draw matches and non-matches for each frame
    
