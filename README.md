# biosim
Supervised biological similarity models built for distribution with biobricks

```sh
biobricks install biosim
biobricks import biosim ./imports
```

R
```r
mod <- keras::load_model("imports/biosim/biosim.hdf5")
mod$predict("c1ccccc1")
```

python
```py
from keras.models import load_model
model = load_model('imports/biosim/biosim/hdf5')
model.predict('c1ccccc1')
```

# Development
get dev environment with docker
need nvidia-docker2 to run this container ([install directions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker))
```
docker run --gpus -v $(pwd):/biosim insilica/biobricks:3.0-cuda /bin/bash
```
