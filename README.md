# PushT Imitation Learning

The model is an encoder-decoder transformer model using a linear action head and is written from scratch with PyTorch. Future work should explore a diffusion action head, as it is shown to perform better (see Diffusion Policy, Chi).

The model achieved successful results with 58M parameters trained over 200 epochs for 48973 seconds.

For more details, see the "pusht.ipynb" notebook.

##
These videos demonstrate the model successfully solving the PushT problem on unseen gym environments.

[<img src="https://img.youtube.com/vi/gIHPeBsfaAo/hqdefault.jpg" width="256" height="256"
/>](https://www.youtube.com/embed/gIHPeBsfaAo)


[<img src="https://img.youtube.com/vi/m6ErDz2-Lc8/hqdefault.jpg" width="256" height="256"
/>](https://www.youtube.com/embed/m6ErDz2-Lc8)

This video illustrates how the model sometimes struggles to complete the task, but ultimately succeeds.

[<img src="https://img.youtube.com/vi/aaWEV6jSGlw/hqdefault.jpg" width="256" height="256"
/>](https://www.youtube.com/embed/aaWEV6jSGlw)


Finally, the model has room for improvement, as this example demonstrates, where the model gets "stuck" unable to complete the task.

[<img src="https://img.youtube.com/vi/p6sua3svdFM/hqdefault.jpg" width="256" height="256"
/>](https://www.youtube.com/embed/p6sua3svdFM)
