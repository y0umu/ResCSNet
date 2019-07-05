# ResCSNet
Code for ResCSNet (and hopefully for my paper to be published) which compresses and restores images using convolutional neural network.

The code is in pytorch. As of coding `torch.utils.tensorboard` wasn't available, so I used `tensorboardX`. (But as it turns out there is no big difference between the two)

It should run in Google Colabotory. The notebook should be self-explanatory.

# Reconstruction demo
| Original                   | Reconstruction at ratio=0.10                         |
|----------------------------|------------------------------------------------------|
|![house](img/house.jpg)     | ![house_rescsnet_r10](img/house_rescsnet_r10.jpg)    |
|![parrot](img/parrot.jpg)   | ![parrot_rescsnet_r10](img/parrot_rescsnet_r10.jpg)  |
|![barbara](img/barbara.jpg) | ![barbara_rescsnet_r10](img/barbara_rescsnet_r10.jpg) |