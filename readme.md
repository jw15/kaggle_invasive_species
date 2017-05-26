# Kaggle Competition: Invasive Species

Identifying invasive hyacinth flowers in images of forest scenes using computer vision [(competition link)](https://www.kaggle.com/c/invasive-species-monitoring)

First attempt: convolutional neural net (Keras with Theano backend) using images resized to 150 x 200 pixels: 68% accuracy in 2 epochs. Very slow with the larger images. Definitely room for improvement.

Things to try next:
1. Add VGG16 convolutional layer as first layer
2. Train on larger images
3. Identify regions/partitions of the image where flowers are located
