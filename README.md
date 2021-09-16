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

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
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


