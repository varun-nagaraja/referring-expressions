Localizing objects using referring expressions
==============================================
This repository contains code for detecting objects in images mentioned by referring expressions. The code is an implementation of the technique presented in our [paper](http://www.umiacs.umd.edu/~varun/files/refexp-ECCV16.pdf). We have also included links to pretrained models, our split of the Google RefExp dataset and also the processed version of the UNC RefExp dataset.
```
@inproceedings{nagaraja16refexp,
  title={Modeling Context Between Objects for Referring Expression Understanding},
  author={Varun K. Nagaraja and Vlad I. Morariu and Larry S. Davis},
  booktitle={ECCV},
  year={2016}
}
```

We have also implemented the baseline and max-margin techniques proposed by Mao et al. in their CVPR 2016 [paper](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Mao_Generation_and_Comprehension_CVPR_2016_paper.pdf). If you use the Google RefExp dataset, please cite this paper
```
@inproceedings{google_refexp,
  title={Generation and Comprehension of Unambiguous Object Descriptions},
  author={Mao, Junhua and Huang, Jonathan and Toshev, Alexander and Camburu, Oana and Yuille, Alan and Murphy, Kevin},
  booktitle={CVPR},
  year={2016}
}
```

If you use the UNC RefExp dataset, please cite the following paper
```
@inproceedings{unc_refexp,
  title={Modeling Context in Referring Expressions},
  author={Licheng Yu and Patric Poirson and Shan Yang and Alexander C. Berg and Tamara L. Berg},
  booktitle={ECCV},
  year={2016}
}
```

Setup
=====
##### Clone this repository
```Shell
git clone --recursive https://github.com/varun-nagaraja/referring-expressions.git
```
We will call the directory that you cloned this repository into as `$RefExp_ROOT`

##### Build external components
* Build caffe and pycaffe. The instructions for installing caffe are [here](http://caffe.berkeleyvision.org/installation.html).
	```Shell
	cd $RefExp_ROOT/caffe
	make -j8 && make pycaffe
	```

* Download VGGnet
	```Shell
	mkdir $RefExp_ROOT/caffe/models/vggnet
	cd $RefExp_ROOT/caffe/models/vggnet
	wget https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt
	wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
	```

* Build COCO toolbox
	```Shell
	cd $RefExp_ROOT/coco/PythonAPI
	make
	```

##### Download the datasets

* The MSCOCO dataset can be download from [here](http://mscoco.org/dataset/#download). You will need the training images, validation images and object instance annotations.

    Our code uses the following directory structure
	```
	$COCO_PATH
	├── annotations
	│   ├── instances_train2014.json
	│   └── instances_val2014.json
	├── images
	│   ├── train2014
	│   └── val2014
	├── google_refexp
	└── unc_refexp
	```
* Download Google RefExp dataset with our split and MCG region candidates
	```Shell
	cd $COCO_PATH
	wget https://obj.umiacs.umd.edu/referring-expressions/google_refexp_umd_split.tar.gz
	tar -xzf google_refexp_umd_split.tar.gz
	rm google_refexp_umd_split.tar.gz
	```
	**Note:** If you want the original split of the Google RefExp dataset, follow the instructions at this [link](https://github.com/mjhucla/Google_Refexp_toolbox). Then move the dataset files to the appropriate folder as indicated above.

* Download UNC RefExp dataset with MCG candidates
	```Shell
	cd $COCO_PATH
	wget https://obj.umiacs.umd.edu/referring-expressions/unc_refexp.tar.gz 
	tar -xzf unc_refexp.tar.gz
	rm unc_refexp.tar.gz
	```

Testing
=======
* Create cache directories where we will store the model and vocabulary files
	```Shell
	python lib/experiment_settings.py --coco_path $COCO_PATH
	```

* Download pre-trained vocabulary and model files
	```Shell
	cd $COCO_PATH/cache_dir
	cd h5_data
	wget https://obj.umiacs.umd.edu/referring-expressions/Google_RefExp_vocabulary.txt
	wget https://obj.umiacs.umd.edu/referring-expressions/UNC_RefExp_vocabulary.txt
	cd ..
	cd models
	# baseline models trained on Google RefExp and UNC RefExp
	wget https://obj.umiacs.umd.edu/referring-expressions/baseline_models.tar.gz
	tar -xzf baseline_models.tar.gz
	rm baseline_models.tar.gz
	# max-margin models
	wget https://obj.umiacs.umd.edu/referring-expressions/max_margin_models.tar.gz
	tar -xzf max_margin_models.tar.gz
	rm max_margin_models.tar.gz
	# context models with negative bag margin
	wget https://obj.umiacs.umd.edu/referring-expressions/mil_context_withNegMargin_models.tar.gz
	tar -xzf mil_context_withNegMargin_models.tar.gz
	rm mil_context_withNegMargin_models.tar.gz
	# context models with positive bag margin
	wget https://obj.umiacs.umd.edu/referring-expressions/mil_context_withPosNegMargin_models.tar.gz
	tar -xzf mil_context_withPosNegMargin_models.tar.gz
	rm mil_context_withPosNegMargin_models.tar.gz
	```
	**Note**: In the paper, for Google RefExp experiments, we report numbers from models trained on a subset of the training set since we use the remaining training set for validation. However, these pretrained models were trained on the entire training set and hence provide slighty better numbers than those reported in the paper.

* Evaluate on a dataset split
	```Shell
	python lib/comprehension_experiments.py --coco_path $COCO_PATH --dataset Google_RefExp --exp_name baseline --split_name val --proposal_source gt
	```
	Use `--visualize` option to pause at every image and display localization results

Training
========
* Create files for training
	```
	python lib/process_dataset.py --coco_path $COCO_PATH --dataset Google_RefExp --exp_name baseline
	```
	This will first extract region features for all images in the dataset and dump them in a format suitable for loading in caffe. It will require a lot of space on disk depending on the experiment type you want to run.

* If you are working with Google RefExp, we will split the training data to create a validation partition of our own. The test set of the Google RefExp dataset is not yet released.
	```
	cd $COCO_PATH/cache_dir/h5_data/buffer_16/Google_RefExp_baseline_20
	head -n 5038 hdf5_chunk_list.txt > hdf5_chunk_list_part1.txt
	tail -n 300 hdf5_chunk_list.txt > hdf5_chunk_list_part2.txt
	```

* Edit training prototxt file (Ex.: `proto_files/google_refexp/google_refexp.baseline.prototxt`) and set the correct source in hdf5_data_param. For example, 
	```
	hdf5_data_param {
	    source: "$COCO_PATH/cache_dir/h5_data/buffer_16/Google_RefExp_train_baseline_20/hdf5_chunk_list_part1.txt"
	    batch_size: 16
	}
	```

* Edit solver.prototxt file (Ex.: `proto_files/google_refexp/google_refexp.baseline.solver.prototxt`) and set snapshot_prefix appropriately. For example,
	```
	snapshot_prefix: "$COCO_PATH/cache_dir/models/Google_RefExp_baseline/google_refexp.baseline"
	```
	**Important** Create the required model directory (Ex.: `$COCO_PATH/cache_dir/models/Google_RefExp_baseline`)

* Edit training script to use the experiment name you are interested in. Then run the script from the `$RefExp_ROOT` directory.
	```
	./proto_files/google_refexp/google_refexp_train.sh
	```
	The script will print the log to the screen and also write to a file.

* When the training is complete, choose the iteration number of the model snapshot with the lowest cross entropy loss on the validation set. Set this iteration number in `lib/experiment_settings.py` file to test the trained model. The following command will extract the lines which contain the cross entropy loss.
	```
	grep "Testing net (#1)" -A 4 google_refexp.baseline.log
	```

