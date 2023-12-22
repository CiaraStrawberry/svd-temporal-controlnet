# sdv_controlnet

Stable Video Diffusion Temporal Controlnet!

This is an implementation of the controlnet style encoder over the svd base.

To run this simply add a controlnet model folder from here: 
into  this directory, install the requirements and run "run_inference.py" with your folders  added

![combined_with_square_image_new_gif](https://github.com/CiaraStrawberry/sdv_controlnet/assets/13116982/e39192af-c31e-448d-975b-95fcecd34150)

Some notes for getting a good result:

This will tend to latch on to a central object to extract motion features from + sometimes a background, so it may ignore overly complex motion or the moving object if it isn't distinct enough within the object.
Keep to motion that svd could generate without the controlnet.


Thanks to diffusers for the svd implimentation.

Thanks to pixeli99 for the working example of a svd training script: https://github.com/pixeli99/SVD_Xtend
