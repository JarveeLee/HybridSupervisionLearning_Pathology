# Hybrid Supervision Learning for Pathology Whole Slide Image Classification

This is the code of 
Hybrid Supervision Learning for Pathology Whole Slide Image Classification, accepted in MICCAI2021
Jiahui Li, Wen Chen, Xiaodi Huang, Shuang Yang, Zhiqiang Hu, Qi Duan, Dimitris N. Metaxas, Hongsheng Li and Shaoting Zhang


## Description

Hybrid supervision learning in computational pathology this is a difficult task for this reason:
High resolution of whole slide images makes it difficult to do end-to-end
classification model training. To handle this problem, we
propose a hybrid supervision learning framework for this kind of high resolution images with sufficient image-level coarse annotations and a few
pixel-level fine labels. This framework, when applied in training patch
model, can carefully make use of coarse image-level labels to refine generated pixel-level pseudo labels. Complete strategy is proposed to suppress
pixel-level false positives and false negatives.

This code need to run third times to ensemble pixel-level prediction, then run stage2 to 
get 0.9243, 4th in Camelyon17 challenge, the submission.csv is my submission file.  

![alt text](git_shows/pipeline.png)
![alt text](git_shows/strategy.png)
![alt text](git_shows/formula.png)

## Getting Started

### Dependencies

* I have a GPU cluster, using 'srun python' to submit task. Modify code in orders/ to fit your machine.
* Pytorch, opencv, numpy, scikit-learn, scikit-image, PIL. 

### Installing

* 'pip install' all those packages 

### Executing program

* 1,  Replace paths of your whole slide images in 'data/train_pos_1_wsi.txt' et al.
* 2,  Firstly go to data/, 'python crop_segmentation.py' to generate finegrain pixel labels from xmls.
* 3,  'python orders/Stage1_M_step_init.py' to train initial models.
* 4,  'python orders/Stage1_E_step_init_generate_pseudo_label.py' to generate pseudo labels for all whole slide images.
* 5,  'python oswalk.py' to generate paths indexes, replace keys.
* 6,  Repeat step 3~5 from 'init' to 'round3'
* 7,  'python orders/Stage1_E_step_round3_generate_pseudo_label_for_testdata.py' to generate pseudo labels for all whole slide images.
* 8,  'python Stage2_classification_generate_heatmap.py' to generate down sampled heatmaps.
* 9,  ensemble heatmaps from several trail.
* 10, 'python Stage2_classifier.py' to generate submission file from ensembled heatmaps.


```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)


