# Crystal-Cluster-and-Surface-Network
A model for predicting properties for 2D materials


## Development Environment
### Package dependencies

- Please make sure about your running enviornment and package version.
- Linux (ubantu)>=18.04
- python==3.8
- pymatgen==2022.4.19
- matplotlib>=3.0.3
- pandas>=1.0.5
- numpy>=1.16.2
- pytorch>=1.11.0
- torchvision>=0.12.0
- torch-geometric>=2.0.4


### Package Installation
We can install all neccessary packages according to 'pip' or 'conda' with short time. 
All data can be downloaded from Materials Project (https://materialsproject.org/),  Materials Cloud(https://www.materialscloud.org/home), Computational 2D Materials Database ( https://cmr.fysik.dtu.dk/c2db/c2db.html#c2db) . 


## About the Code
- The training and testing process are defined in ` ./Model/CCSN_model/train.py`.
- Training results will be automatically saved in `./Model/CCSN_model/` file.
- We provide pretrain model for bandgap task in `./Pretrain_model/` file , which can be used to perform transfer learning.
- Able to choose using both top and bottom surface.
- Able to use E2CNN or CNN


## License
This project is covered under the Apache 2.0 License.

## Thanks for your time and attention.

