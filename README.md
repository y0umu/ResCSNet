# ResCSNet
Code for ResCSNet which compresses and restores images using convolutional neural network.

The code is in pytorch. As of coding `torch.utils.tensorboard` wasn't available, so I used `tensorboardX`. (But as it turns out there is no big difference between the two)

It should run in Google Colaboratory. The notebook should be self-explanatory.

**NOTE** During my experiments I notice that certain versions of numpy or pytorch is broken. This will cause the recovered image to have radiculous 'holes'. For example the demo image of parrot in [commit 9b8376fc402160c7b8330b15d9ee75a61822906b](https://github.com/y0umu/ResCSNet/tree/9b8376fc402160c7b8330b15d9ee75a61822906b). More recent versions of numpy (or maybe pytorch) has fixed the bug.

# Reconstruction demo
The 'ratio' in the table below refers to the compression ratio, i.e. ratio=M/N
| Original                   | ratio=0.25                                           | ratio=0.20                                           | ratio=0.15                                           | ratio=0.10                                           | ratio=0.04                                           | ratio=0.01                                           |
|----------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
|![house](img/house.jpg)     | ![house_rescsnet_r25](img/house_rescsnet_r25.jpg)    | ![house_rescsnet_r20](img/house_rescsnet_r20.jpg)    | ![house_rescsnet_r15](img/house_rescsnet_r15.jpg)    | ![house_rescsnet_r10](img/house_rescsnet_r10.jpg)    | ![house_rescsnet_r04](img/house_rescsnet_r04.jpg)    | ![house_rescsnet_r01](img/house_rescsnet_r01.jpg)    |
|![parrot](img/parrot.jpg)   | ![parrot_rescsnet_r25](img/parrot_rescsnet_r25.jpg)  | ![parrot_rescsnet_r20](img/parrot_rescsnet_r20.jpg)  | ![parrot_rescsnet_r15](img/parrot_rescsnet_r15.jpg)  | ![parrot_rescsnet_r10](img/parrot_rescsnet_r10.jpg)  | ![parrot_rescsnet_r04](img/parrot_rescsnet_r04.jpg)  | ![parrot_rescsnet_r01](img/parrot_rescsnet_r01.jpg)  |
|![barbara](img/barbara.jpg) | ![barbara_rescsnet_r25](img/barbara_rescsnet_r25.jpg)| ![barbara_rescsnet_r20](img/barbara_rescsnet_r20.jpg)| ![barbara_rescsnet_r15](img/barbara_rescsnet_r15.jpg)| ![barbara_rescsnet_r10](img/barbara_rescsnet_r10.jpg)| ![barbara_rescsnet_r04](img/barbara_rescsnet_r04.jpg)| ![barbara_rescsnet_r01](img/barbara_rescsnet_r01.jpg)|

# Citation
I am still working on the research article. I would be glad to hear that this repo can help you even without the paper.