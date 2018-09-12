# Adapt-ReID
This repository contains codes for AAAI-18 unsupervised domain adaptation for person re-identification.

### Download the Dataset
We prepare the dataset for Market1501, DukeMTMC, MSMT17, CUHK03, Caviar, Viper
``` 
wget https://www.dropbox.com/s/b30wcqjb1hwy7o8/reid_dataset.zip
```


### Setup Dataset Root Directory
``` 
python setup.py --dataset-dir path/to/dataset
```

It will automatically generate a ```config.py``` under the current directory.


