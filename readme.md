# Kaggle Competition: Invasive Species

Identifying invasive hyacinth flowers in images of forest scenes using computer vision [(competition link)](https://www.kaggle.com/c/invasive-species-monitoring)

First attempt: convolutional neural net (Keras with Theano backend) using images resized to 150 x 200 pixels: 68% accuracy in 2 epochs. Very slow with the larger images. Definitely room for improvement.

Second attempt (code is in /src): CNN using images centered and resized to square (256x256), then cropped to 224x224. Used keras image generator to generate images that are rotated, random height/width shifts, flipped on horizontal axis). Trained neural net on top of VGG16. 98% accuracy in 40 epochs. This code was adapted from [(work by kaggle user fujisan)](https://www.kaggle.com/fujisan/use-keras-pre-trained-vgg16-acc-98/versions).

Things to try next:
1. Train on larger images (Using images resized to 512x512 and cropped to 488x488 so far is not working very well.)
2. Try multiscale model
3. Identify regions/partitions of the image where flowers are located
