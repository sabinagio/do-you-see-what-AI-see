# Glaucoma detection from fundus images
For a detailed introduction into eye diseases, glaucoma, and previous literature, please refer to [this presentation](https://docs.google.com/presentation/d/1F_ygRbjZ08RPUSjtxJwbNJWhHZWojSwe-Ly8sdUpPnI/edit?usp=sharing).

<img src="https://github.com/sabinagio/do-you-see-what-AI-see/blob/master/data/kaggle_dataset/glaucoma/Glaucoma_051.png" alt="Retinal fundus image of a glaucomatous eye" width=500 height=auto>

## About the project
This project is part of the Ironhack Data Analytics Remote Part Time bootcamp and was completed in 3 weeks. The purpose of the project is to build a neural network that detects glaucoma from retinal fundus images and compare its results on a novel dataset to already established architectures.

## Technical requirements
Please refer to the `requirements.txt`.

## Machine learning process
The project was separated in multiple Jupyter notebooks explaining each specific part of the project, which you can find below:

1. [Data Collection & Exploration](https://github.com/sabinagio/do-you-see-what-AI-see/blob/master/scripts/1-data-collection-and-exploration.ipynb)
2. [Data Splitting, Augmentation, and Preprocessing](https://github.com/sabinagio/do-you-see-what-AI-see/blob/master/scripts/2.data-splitting-augmentation-and-preprocessing.ipynb)
3. [Simple Neural Network Training](https://github.com/sabinagio/do-you-see-what-AI-see/blob/master/scripts/3-simple-CNN-model-acrima-training.ipynb)  
4. [Evaluation of Multiple Neural Networks on a New Dataset](https://github.com/sabinagio/do-you-see-what-AI-see/blob/master/scripts/4-acrima-models-evaluation.ipynb)  
5. [Learning Resources](https://github.com/sabinagio/do-you-see-what-AI-see/blob/master/learning-resources.md)

<details>
<summary><h2> Project Challenges </h2></summary>
<ol>
<li> Selecting a suitable project </li>
<li> Acquiring the knowledge to finish the project </li>
<li> Choosing an effective training strategy </li>
<li> Improving simple model prediction accuracy </li>
<li> Cropping the test images for evaluation </li>
</ol>
</details>

<details>
<summary><h2> Lessons Learned </h2></summary>
<ol> 
<li> Do your research before choosing the project - you might realize that what you set out to do is not as impactful as you initially thought. </li>
<li> Collect data early - it takes a while for images to download. </li>
<li> Keep backups for your data before you start processing it - e.g. `shutil.move()` can be a dangerous command if used improperly. </li>
<li> Ensure you're able to run your project in the cloud (either via Google Colab or a different cloud provider), as training might be too taxing for your local machine. </li>
<li> Try to test your data on a real life scenario as soon as possible, so you can discard poorly trained models. </li>
<li> Check the changes you would undo before resetting to a previous git version - and if you don't, make sure you have backups. </li>
</ol>
</details>
 
## Limitations
1. The deep learning model might have trouble with images where the optic nerve head is damaged.
2. There are few publicly available labelled datasets.

## Results
The simple CNN model (G: 94%, N: 3.6%) is significantly outperformed by the more complex architectures (avg G: 63%, avg N: 97.7%), which have performed relatively well on an unseen dataset. 

## Conclusions
We can conclude that model architecture, data preprocessing, and an extensive training dataset are key to the performance of a disease detection algorithm, as shown by the difference in performance between a minimum viable model and more complex models.

## References
[1] [A global review of publicly available datasets for ophthalmological imaging: barriers to access, usability, and generalisability](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(20)30240-5/fulltext) - *Saad M Khan, Xiaoxuan Liu, Siddharth Nath, Edward Korot, Livia Faes, Siegfried K Wagner et al.*  
[2] [ACRIMA dataset (cropped images)](https://figshare.com/s/c2d31f850af14c5b5232)  
[3] [Cataract dataset (full images)](https://www.kaggle.com/datasets/jr2ngb/cataractdataset)  
[4] [Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation](https://arxiv.org/abs/1801.00926) - *Huazhu Fu, Jun Cheng, Yanwu Xu, Damon Wing Kee Wong, Jiang Liu, and Xiaochun Cao*  
[5] [CNNs for Automatic Glaucoma Assessment using Fundus Images: An Extensive Validation](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-019-0649-y#Sec3) - *Diaz-Pinto, A., Morales, S., Naranjo, V. et al.*  

## Acknowledgements
Thank you Xisca & Arek for the support, inspiration, and encouragement throughout the bootcamp and especially during the final project :star:
