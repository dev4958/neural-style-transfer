# Neural Style Transfer

Neural style transfer implementation in TensorFlow with a couple improvements and quirks.  Based on [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf) and improvements in [Improving the Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1605.04603.pdf).

### Improvements

* Geometric weighting scheme for content and style layers.
* Utilizes all 16 convolutional layers in the VGG-19 model.
* Calculates Gram matrices using shifted activations.
* Feature comparisons with blurred chain correlations between local layers.

### Quirks

* Weighs feature comparisons within the same layer, *l*, and *l* and *l - 1* differently.
* Blurs *l* and *l - 1* correlation using a gaussian blur instead of a box blur.
* Hyperparameters...

### Usage

Download some images to work with and the pretrained [VGG-19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) model (save the model file in the project's root directory, don't change it's name).  Install [NumPy](https://www.scipy.org/install.html), [SciPy](https://www.scipy.org/install.html), and [TensorFlow](https://www.tensorflow.org/install/).  If possible, install the version of TensorFlow that runs on your GPU, this speeds things up considerably.  Then run the following in your terminal:

$`python3 neural-style-transfer.py [content path] [style path] [width]`

Generated images are saved inside the "output" directory when the process is completed.  Inside the "iterations" folder there will be generated images from every 100 iterations processed.  Inside the "setup" folder will be the resized content and style images used during processing.

### Examples

Content | Style | Output
------- | ----- | ------
![Man in a Forest Landscape](/readme-files/man-in-a-forest-landscape.jpg) | ![Cypresses](/readme-files/cypresses.jpg) | ![Man in a Forest Landscape + Cypresses](/readme-files/man-in-a-forest-landscape+cypresses.jpg)

Content: [Man in a Forest Landscape](https://images.metmuseum.org/CRDImages/ph/original/DT223780.jpg) (Constant Alexandre Famin, ca. 1870)

Style: [Cypresses](https://images.metmuseum.org/CRDImages/ep/original/DP130999.jpg) (Vincent van Gogh, 1889)

Iterations: 1000

</br>

Content | Style | Output
------- | ----- | ------
![Medinet Habu](/readme-files/medinet-habu.jpg) | ![Rochishin Chopping Off the Head of Nio](/readme-files/rochishin-chopping-off-the-head-of-nio.jpg) | ![Medinet Habu + Rochishin Chopping Off the Head of Nio](/readme-files/medinet-habu+rochishin-chopping-off-the-head-of-nio.jpg)

Content: [Medinet-Habu](https://images.metmuseum.org/CRDImages/ph/original/DT1163.jpg) (John Beasley Greene, 1854)

Style: [Rochishin Chopping Off the Head of Nio](https://images.metmuseum.org/CRDImages/as/original/56_121_40_162330.jpg) (School of Katsushika Hokusai, 18th-19th century)

Iterations: 1000

</br>

Content | Style | Output
------- | ----- | ------
![Sentinel Rock Yosemite](/readme-files/sentinel-rock-yosemite.jpg) | ![The Beeches](/readme-files/the-beeches.jpg) | ![Sentinel Rock Yosemite + The Beeches](/readme-files/sentinel-rock-yosemite+the-beeches.jpg)

Content: [Sentinel Rock, Yosemite](https://images.metmuseum.org/CRDImages/ph/original/DP152226.jpg) (attributed to Carleton E. Watkins, ca. 1872, printed ca. 1876)

Style: [The Beeches](https://images.metmuseum.org/CRDImages/ap/original/DT75.jpg) (Asher Brown Durand, 1845)

Iterations: 1000

</br>

Content | Style | Output
------- | ----- | ------
![11:00 A.M. Monday, May 9th, 1910. Newsies at Skeeter's Branch, Jefferson near Franklin. They were all smoking. Location: St. Louis, Missouri.](/readme-files/11-00am-monday-may-9th-1910-newsies-at-skeeters-branch-jefferson-near-franklin-they-were-all-smoking-location-st-louis-missouri.jpg) | ![Fleur de Lis](/readme-files/fleur-de-lis.jpg) | ![11:00 A.M. Monday, May 9th, 1910. Newsies at Skeeter's Branch, Jefferson near Franklin. They were all smoking. Location: St. Louis, Missouri. + Fleur de Lis](/readme-files/11-00am-monday-may-9th-1910-newsies+fleur-de-lis.jpg)

Content: [11:00 A.M. Monday, May 9th, 1910. Newsies at Skeeter's Branch, Jefferson near Franklin. They were all smoking. Location: St. Louis, Missouri.](https://images.metmuseum.org/CRDImages/ph/original/DP352686.jpg) (Lewis Hine, May 9, 1910)

Style: [Fleur de Lis](https://images.metmuseum.org/CRDImages/ap/original/DP167061.jpg) (Robert Reid, ca. 1895-1900)

Iterations: 1000
