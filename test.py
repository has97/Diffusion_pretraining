from imagenet_aug import ImageNetDataset
import sys
strength =0.2
guidance=0.3
augment = ImageNetDataset(strength=strength,guidance=guidance)
augment.generate_augmentation()
