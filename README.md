# biosim
Supervised biological similarity models built for distribution with biobricks

```sh
biobricks install biosim
biobricks import biosim ./imports
```

```r
mod <- keras::load_model("imports/biosim/biosim.hdf5")
mod$predict("c1ccccc1")
```

```py
from keras.models import load_model
model = load_model('imports/biosim/biosim/hdf5')
model.predict('c1ccccc1')
```

# Development
get dev environment with docker

```
docker run -v $(pwd):/biosim insilica/biobricks:3.0-cuda /bin/bash
```