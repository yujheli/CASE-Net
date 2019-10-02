# Var-HR-ReID
This repository contains codes for CVPR-19 Variational HRCNN for person re-identification.

### Download the Dataset
We prepare the dataset for Market1501, DukeMTMC, MSMT17, CUHK03, Caviar, Viper
``` 
wget https://www.dropbox.com/s/2vhployyys67vbd/reid_dataset.zip  
```


### Setup Dataset Root Directory
``` 
python setup.py --dataset-dir path/to/dataset
```

It will automatically generate a ```config.py``` under the current directory.


